#pragma once

// Online drafter arbiter (Strata DraftPolicy idea, port-notes/T5B-REPORT.md, audit card A2)
//
// A draft round is not free: the drafter here is a full MoE layer (a nextn block with
// its own experts and an lm_head read on every step), so a rejected draft is expensive.
// The arbiter measures the realized economics of each round and decides, per sequence,
// whether to draft at all and how deep. The goal is to keep the big spec win on
// predictable text while recovering the no-draft floor on text where acceptance is low
// (where a fixed draft config loses).
//
// THE GAIN
//
// A draft round produces `tokens` tokens (accepted draft tokens + the bonus token the
// verify always yields) and takes `t_round` wall time. Without drafting, the same verify
// pass yields 1 token in `t_verify` wall time, so the no-draft floor is
//   floor = 1 / t_verify
// and the drafted rate is
//   rate  = tokens / t_round
// Drafting pays iff rate > floor, i.e. iff
//   gain = tokens * t_verify / t_round > 1
// t_verify is estimated as t_round - t_draft: the target verify pass is not instrumented
// inside speculative.cpp, so it is taken by subtraction (it includes the server gap,
// which is present in both paths and largely cancels in the ratio). This is exact under
// the assumption that the verify cost is ~independent of the draft depth, which holds on
// this box: the target verify is one batched pass dominated by the dense-trunk weight
// read (~3.4 GB/token at depth 0, per the T5 roofline), so verifying 4 positions costs
// almost the same as verifying 1.
//
// Hysteresis
//
// gain is noisy (a single round is a small sample), so it is filtered through an EWMA and
// the state flips only outside a band: engage when ewma_gain > engage, disengage when
// ewma_gain < disengage, with disengage < engage (default 0.95 / 1.10). The band is the
// hysteresis that stops a naive per-step threshold at 1.0 from chattering.
//
// Re-engagement (the probe)
//
// While disengaged the arbiter emits no draft, so it receives no accept() callbacks and
// cannot see the gain. To notice a regime switch (prose -> repeaty mid-run) it probes:
// every `probe_interval` no-draft rounds it issues one short draft of `probe_len` tokens.
// The probe's accept updates the EWMA; if the EWMA then exceeds the engage threshold the
// arbiter re-engages. Because the EWMA starts low from the previous (bad) regime, the
// first probe after a switch usually only lifts it part-way and the second crosses the
// engage band: the worst-case re-engagement latency is two probe intervals (default 2 x
// 16 = 32 rounds), and probes are infrequent and bounded so there is no oscillation.
//
// The core is a self-contained, dependency-free state machine so it can be unit-tested
// offline by replaying synthetic acceptance streams (tests/test-spec-arbiter.cpp) and by
// the python mirror (port-tools/arbiter-sim.py in the workspace root).

#include <cstdint>
#include <cstddef>

namespace spec_arbiter {

struct config {
    // state thresholds; keep 0 < disengage < engage (engage is typically just above 1)
    float engage    = 1.10f;
    float disengage = 0.95f;

    // ewma alpha in (0, 1]; higher reacts faster, less smoothing
    float alpha = 0.30f;

    // while disengaged: probe every probe_interval no-draft rounds with a probe_len draft
    int32_t probe_interval = 16;
    int32_t probe_len      = 2;

    // number of measured drafted rounds before the state may flip; the first rounds are a
    // warm-up at full depth so a predictable prefix is not throttled from step one
    int32_t warmup = 4;

    bool valid() const {
        return alpha > 0.0f && alpha <= 1.0f &&
               engage > disengage && disengage > 0.0f &&
               probe_interval >= 1 && probe_len >= 1 &&
               warmup >= 0;
    }
};

// decision for the upcoming round
struct decision {
    bool    draft   = true; // whether to draft at all this round
    int32_t max_len = 0;    // depth cap when drafting (0 = no cap, use the configured n_max)
};

// one round's realized outcome
struct observation {
    int32_t n_accepted = 0; // accepted draft tokens (0 for a no-draft round)
    int64_t t_round_us = 0; // start of the draft round -> end of its accept
    int64_t t_draft_us = 0; // wall time inside the draft call (the drafter's cost)
};

class core {
public:
    explicit core(const config & cfg) : cfg_(cfg) { reset(); }

    // start a new generation; re-optimistic (engaged, no warm-up credit)
    void reset() {
        engaged_     = true;
        gain_ema_    = 1.0f;
        n_measured_  = 0;
        n_nodraft_   = 0;
        n_drafts_    = 0;
        n_tokens_    = 0;
        n_rounds_    = 0;
        n_engage_    = 0;
        n_disengage_ = 0;
        n_probes_    = 0;
    }

    // decision for the upcoming round; also issues the probe if one is due
    decision decide() {
        ++n_rounds_;

        if (engaged_) {
            // warm-up and normal drafting both draft; the depth is owned by the caller's
            // acceptance ema (common_speculative_impl::arbiter_n_draft), not by the arbiter
            return { /*draft=*/ true, /*max_len=*/ 0 };
        }

        // disengaged: count no-draft rounds and probe when due
        if (++n_nodraft_ >= cfg_.probe_interval) {
            n_nodraft_ = 0;
            ++n_probes_;
            return { /*draft=*/ true, /*max_len=*/ cfg_.probe_len };
        }
        return { /*draft=*/ false, /*max_len=*/ 0 };
    }

    // a drafted round finished with `n_accepted` accepted tokens; updates the EWMA and the
    // engage/disengage state. call once per drafted round (probes included).
    void on_drafted_round(int32_t n_accepted, int64_t t_round_us, int64_t t_draft_us) {
        ++n_drafts_;
        n_tokens_ += n_accepted;

        if (t_round_us <= 0 || t_draft_us < 0 || t_draft_us > t_round_us) {
            return; // degenerate timing; keep the previous state
        }

        const int64_t t_verify_us = t_round_us - t_draft_us;
        if (t_verify_us <= 0) {
            // degenerate verify estimate (the draft ate the whole round): full loss
            update_state(0.0f);
            return;
        }

        // even a fully rejected draft yields the bonus verify token, so the round's
        // output is always n_accepted + 1 tokens and the gain is always well defined
        const float tokens = (float) (n_accepted + 1); // accepted + the bonus verify token
        const float gain   = tokens * (float) t_verify_us / (float) t_round_us;

        update_state(gain);
    }

    // convenience wrapper
    void observe(const observation & o) {
        on_drafted_round(o.n_accepted, o.t_round_us, o.t_draft_us);
    }

    bool engaged() const { return engaged_; }

    // diagnostics (logging / stats / the simulator)
    float   gain_ema()  const { return gain_ema_; }
    int32_t n_rounds()    const { return n_rounds_; }
    int32_t n_drafts()    const { return n_drafts_; }
    int32_t n_tokens()    const { return n_tokens_; }
    int32_t n_engage()    const { return n_engage_; }
    int32_t n_disengage() const { return n_disengage_; }
    int32_t n_probes()    const { return n_probes_; }

private:
    void update_state(float gain) {
        ++n_measured_;
        gain_ema_ = (1.0f - cfg_.alpha) * gain_ema_ + cfg_.alpha * gain;

        // no flipping during warm-up: the first rounds establish the measurement
        if (n_measured_ <= cfg_.warmup) {
            return;
        }

        if (engaged_) {
            if (gain_ema_ < cfg_.disengage) {
                engaged_     = false;
                n_nodraft_   = 0;
                ++n_disengage_;
            }
        } else {
            if (gain_ema_ > cfg_.engage) {
                engaged_     = true;
                n_nodraft_   = 0;
                ++n_engage_;
            }
        }
    }

    config cfg_;

    bool    engaged_     = true;
    float   gain_ema_    = 1.0f;
    int32_t n_measured_  = 0; // drafted rounds with valid timing (warm-up counter)
    int32_t n_nodraft_   = 0; // no-draft rounds since the last probe (disengaged only)
    int32_t n_drafts_    = 0;
    int32_t n_tokens_    = 0;
    int32_t n_rounds_    = 0;
    int32_t n_engage_    = 0;
    int32_t n_disengage_ = 0;
    int32_t n_probes_    = 0;
};

} // namespace spec_arbiter
