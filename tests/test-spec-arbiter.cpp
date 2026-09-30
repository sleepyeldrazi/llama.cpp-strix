// Offline unit test for the online drafter arbiter (spec_arbiter::core).
//
// No model is loaded and nothing GPU-related is touched. The arbiter core is a pure
// state machine, so this replays synthetic acceptance streams (each round = a drafted
// round with an accepted-token count and a measured draft/round time) and checks the
// engage/disengage decisions over time.
//
// This proves the control loop (engage in a high-acceptance regime, disengage in a
// low-acceptance regime, re-engage after a mid-run regime switch, and no oscillation);
// it does NOT prove a real speedup, which requires the GPU A/B in the T5B report.

#include "spec-arbiter.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

// ---- a tiny test harness -------------------------------------------------------

static int g_assertions = 0;
static int g_failures   = 0;

static void check(bool ok, const std::string & what) {
    ++g_assertions;
    if (!ok) {
        ++g_failures;
        std::printf("    [FAIL] %s\n", what.c_str());
    }
}

// run a synthetic round stream through a fresh arbiter. each round i describes the
// content: if a draft were issued, accs[i] tokens would be accepted and the draft/round
// times would be draft_us[i]/round_us[i]. Faithful to the real system: a no-draft round
// produces no accept callback, so the EWMA is updated only on drafted rounds (probes
// included); a probe round's acceptance is capped by the probe length.
struct run_result {
    std::vector<bool>  drafted;  // per round: did the arbiter draft
    std::vector<float> gain;     // per round: ewma gain (diagnostic)
    bool               engaged_end = false;
};

static run_result run_stream(const std::vector<int32_t> & accs, const std::vector<int64_t> & draft_us,
                             const std::vector<int64_t> & round_us, spec_arbiter::config cfg) {
    spec_arbiter::core a(cfg);
    run_result r;
    for (size_t i = 0; i < accs.size(); ++i) {
        const spec_arbiter::decision d = a.decide();
        r.drafted.push_back(d.draft);
        if (d.draft) {
            int32_t acc = accs[i];
            if (d.max_len > 0) {
                acc = std::min(acc, d.max_len); // a probe can accept at most its length
            }
            a.on_drafted_round(acc, round_us[i], draft_us[i]);
        }
        r.gain.push_back(a.gain_ema());
    }
    r.engaged_end = a.engaged();
    return r;
}

// count how many times the decision flips drafted<->not-drafted
static int count_flips(const std::vector<bool> & v) {
    int flips = 0;
    for (size_t i = 1; i < v.size(); ++i) {
        if (v[i] != v[i-1]) ++flips;
    }
    return flips;
}

// ---- tests ---------------------------------------------------------------------

static int test_high_stays_engaged() {
    // 40 rounds of full acceptance, cheap draft: gain >> 1, must stay engaged, always draft
    spec_arbiter::config cfg; // defaults
    std::vector<int32_t>  accs(40, 3);
    std::vector<int64_t>  du(40, 15000);   // 15 ms draft
    std::vector<int64_t>  ru(40, 45000);   // 45 ms round (30 ms verify)
    auto r = run_stream(accs, du, ru, cfg);
    // gain = 4 * 30 / 45 = 2.67 -> engaged
    check(r.engaged_end, "high regime ends engaged");
    check(std::all_of(r.drafted.begin(), r.drafted.end(), [](bool b){return b;}),
          "high regime: drafts every round (no disengage)");
    check(r.gain.back() > 2.0f, "high regime: ewma gain > 2 (got " + std::to_string(r.gain.back()) + ")");
    return g_failures;
}

static int test_low_disengages() {
    // 60 rounds of low acceptance, expensive draft: gain < 1, must disengage and mostly
    // stop drafting (probe gaps allowed)
    spec_arbiter::config cfg; // defaults (warmup 4, probe_interval 16, probe_len 2)
    std::vector<int32_t>  accs;
    std::vector<int64_t>  du, ru;
    for (int i = 0; i < 60; ++i) {
        accs.push_back((i % 2) ? 1 : 0); // 0 or 1 accepted
        du.push_back(40000);             // 40 ms draft (expensive MoE drafter)
        ru.push_back(70000);             // 70 ms round (30 ms verify)
    }
    auto r = run_stream(accs, du, ru, cfg);
    // gain for acc=1: 2*30/70 = 0.857 ; acc=0: 1*30/70 = 0.429 ; ewma ~ 0.64 < 0.95
    check(!r.engaged_end, "low regime ends disengaged (got gain " + std::to_string(r.gain.back()) + ")");
    // after disengage it must stop drafting for a run of rounds (allow the warm-up drafts
    // and the occasional probe). count the longest no-draft run after the first disengage.
    int first_nodraft = -1;
    for (size_t i = 0; i < r.drafted.size(); ++i) if (!r.drafted[i]) { first_nodraft = (int)i; break; }
    check(first_nodraft > cfg.warmup, "low regime: disengaged after warm-up (first no-draft round "
          + std::to_string(first_nodraft) + ")");
    // longest consecutive no-draft run
    int longest = 0, cur = 0;
    for (bool b : r.drafted) { cur = b ? 0 : cur + 1; longest = std::max(longest, cur); }
    check(longest >= 8, "low regime: a sustained no-draft stretch exists (longest "
          + std::to_string(longest) + ")");
    return g_failures;
}

static int test_regime_switch_reengages() {
    // 30 low rounds (disengage), then 40 high rounds (must re-engage via a probe).
    // verify it re-engages within a bounded number of rounds of the switch and does not
    // oscillate (few flips).
    spec_arbiter::config cfg; // defaults
    std::vector<int32_t>  accs;
    std::vector<int64_t>  du, ru;
    for (int i = 0; i < 30; ++i) { accs.push_back((i%2)?1:0); du.push_back(40000); ru.push_back(70000); }
    for (int i = 0; i < 40; ++i) { accs.push_back(3);          du.push_back(15000); ru.push_back(45000); }
    auto r = run_stream(accs, du, ru, cfg);

    const int switch_at = 30;
    // find the round (>= switch_at) where sustained drafting resumes after the switch
    int reengage_at = -1;
    for (int i = switch_at; i < (int) r.drafted.size(); ++i) {
        // a sustained re-engage: this round drafts AND the next several do too
        bool sustained = true;
        for (int j = i; j < i + 4 && j < (int) r.drafted.size(); ++j) if (!r.drafted[j]) sustained = false;
        if (sustained) { reengage_at = i; break; }
    }
    // worst case is two probe intervals: the first probe after the switch lifts the EWMA
    // only part-way (it starts low from the previous regime), the second crosses engage.
    check(reengage_at >= switch_at && reengage_at < switch_at + 2 * cfg.probe_interval + 2,
          "switch: re-engaged within two probe windows after the switch (at round "
          + std::to_string(reengage_at) + " of " + std::to_string(r.drafted.size()) + ")");
    // no oscillation: the flip count in the whole stream should be small (disengage once,
    // re-engage once => ~2-4 flips). allow a few probe-related flips.
    const int flips = count_flips(r.drafted);
    check(flips <= 8, "switch: no oscillation (flips = " + std::to_string(flips) + ")");
    check(r.engaged_end, "switch: ends engaged");
    return g_failures;
}

static int test_warmup_no_throttle() {
    // even if the first round is terrible, the warm-up must not disengage before `warmup`
    spec_arbiter::config cfg; // warmup 4
    std::vector<int32_t>  accs = {0, 0, 0, 0};
    std::vector<int64_t>  du   = {90000, 90000, 90000, 90000};
    std::vector<int64_t>  ru   = {100000,100000,100000,100000};
    auto r = run_stream(accs, du, ru, cfg);
    // during warm-up it must keep drafting (decide() is called before the first round is
    // even measured, and the first `warmup` measured rounds cannot flip the state)
    check(r.drafted[0] && r.drafted[1] && r.drafted[2] && r.drafted[3],
          "warmup: no throttle during the first rounds");
    return g_failures;
}

static int test_probe_bounds_depth() {
    // while disengaged the probe round must cap the draft to probe_len
    spec_arbiter::config cfg; // probe_len 2, probe_interval 16
    spec_arbiter::core a(cfg);
    // force disengaged by feeding bad rounds past warm-up
    for (int i = 0; i < 10; ++i) a.on_drafted_round(0, 100000, 90000);
    check(!a.engaged(), "probe: disengaged before probe check");
    // walk rounds until a probe is issued; on that round the decision.max_len must be 2
    bool saw_probe = false;
    int max_len_seen = -1;
    for (int i = 0; i < 40; ++i) {
        const spec_arbiter::decision d = a.decide();
        if (d.draft && d.max_len > 0) { saw_probe = true; max_len_seen = d.max_len; break; }
    }
    check(saw_probe, "probe: a probe draft is issued while disengaged");
    check(max_len_seen == cfg.probe_len, "probe: probe depth capped to probe_len (got "
          + std::to_string(max_len_seen) + ")");
    return g_failures;
}

static int test_config_validation() {
    spec_arbiter::config good; good.valid(); // defaults are valid
    check(good.valid(), "config: defaults valid");
    spec_arbiter::config bad; bad.alpha = 0.0f;
    check(!bad.valid(), "config: alpha<=0 invalid");
    spec_arbiter::config bad2; bad2.engage = 0.5f; bad2.disengage = 0.9f;
    check(!bad2.valid(), "config: engage<=disengage invalid");
    return g_failures;
}

int main() {
    std::printf("test-spec-arbiter: online drafter arbiter (offline, synthetic streams)\n");

    struct { const char * name; int (*fn)(); } tests[] = {
        { "high acceptance stays engaged",      test_high_stays_engaged },
        { "low acceptance disengages",          test_low_disengages },
        { "regime switch re-engages, no osc",   test_regime_switch_reengages },
        { "warm-up does not throttle",          test_warmup_no_throttle },
        { "probe bounds draft depth",           test_probe_bounds_depth },
        { "config validation",                  test_config_validation },
    };

    int fails_before = g_failures;
    for (auto & t : tests) {
        std::printf("  %s\n", t.name);
        t.fn();
    }

    if (g_failures == fails_before && g_failures == 0) {
        std::printf("OK (%d tests, %d assertions)\n", (int)(sizeof(tests)/sizeof(tests[0])), g_assertions);
        return 0;
    }
    std::printf("FAIL: %d of %d assertions failed\n", g_failures, g_assertions);
    return 1;
}
