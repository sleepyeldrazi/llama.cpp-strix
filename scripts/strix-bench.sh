#!/usr/bin/env bash
# T1 baseline bench for Qwen3.8-Flash-Next GSQ-RCO-IQ3_XXS on Strix Halo.
#
# Drives a single resident llama-server (the real PLE-streaming deployment) so
# every cell runs in the SAME process and the SSD-streamed n-gram table is
# actually exercised. llama-bench cannot express --ngram-on-disk/--model-ple,
# so the server is the measurement vehicle here.
#
# Method per cell: one warm run (discarded, per the fork's first-run-boost note)
# then the measured run(s), all with cache_prompt off and a unique salt so no
# two requests are byte-identical.
#
# Usage:
#   scripts/strix-bench.sh                 # launch server + run all cells
#   scripts/strix-bench.sh --server-only   # just (re)launch the server
set -uo pipefail

SRC="$(cd "$(dirname "$0")/.." && pwd)"
BIN="${SRC}/build-vulkan/bin"
DIR=/tmp/baseline-bench
MODEL="${MODEL:-/var/home/sleepy/workspace/qwen-amd/models/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002.gguf}"
PORT="${PORT:-18181}"

export RADV_PERFTEST=gpl RADV_TEX_ANISO=0

SERVER_CMD=( "$BIN/llama-server"
  -m "$MODEL"
  --port "$PORT" --host 127.0.0.1
  -c 16384 --n-gpu-layers 999 --device vulkan0
  --ngram-on-disk --fit off
  --batch-size 512 --ubatch-size 512
  -t 16 --threads 16
  --no-warmup --no-webui --offline --parallel 1 )

mkdir -p "$DIR"
echo "SERVER CMD: ${SERVER_CMD[*]}" | tee "$DIR/server-cmd.txt"

if [ "${1:-}" != "--server-only" ]; then
  # run cells if the server is already healthy, else this is a fresh launch
  :
fi

if curl -s "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then
  echo "server already healthy on :$PORT"
else
  "${SERVER_CMD[@]}" > "$DIR/server.log" 2>&1 &
  echo "server pid $!"
  echo "$!" > "$DIR/pid"
  for i in $(seq 1 300); do
    grep -q "listening on" "$DIR/server.log" 2>/dev/null && break
    sleep 1
  done
fi

# cold-start time-to-ready (the "47 GB load"), measured on the fresh launch above
if [ -f "$DIR/pid" ] && [ "${1:-}" != "--server-only" ]; then
  S=$(date +%s.%N) # not exact; use the log timestamp instead
  TS=$(grep "listening on" "$DIR/server.log" 2>/dev/null | head -1 | awk '{print $1}')
  echo "time-to-ready log stamp: $TS (M.S.mmm.uuu seconds from process start)"
fi

# cells: name target n_predict n_warm n_meas
cat > "$DIR/cells.json" <<'EOF'
[
 ["pp512_d0", 512, 1, 1, 1],
 ["pp2048_d0", 2048, 1, 1, 1],
 ["tg64_d0", 8, 64, 1, 2],
 ["tg256_d0", 8, 256, 1, 1],
 ["tg64_d16k", 15800, 64, 0, 1]
]
EOF

python3 - "$PORT" "$DIR/results.jsonl" "$DIR/cells.json" <<'PY'
import json, time, urllib.request, sys, random, string
port, outpath, cellsp = sys.argv[1], sys.argv[2], sys.argv[3]
BASE = "http://127.0.0.1:" + port
UNIT = "The quick brown fox jumps over the lazy dog, and it did so again on a Tuesday, and again. "
def post(p):
    r = urllib.request.Request(BASE + "/completion", data=json.dumps(p).encode(),
                               headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(r, timeout=3600) as x:
        return json.load(x)
def toks(prompt, n):
    d = post({"prompt": prompt, "n_predict": n, "temperature": 0,
              "cache_prompt": False, "stream": False})
    return d, d["timings"]
_, t = toks(UNIT * 30, 1); per = t["prompt_n"] / 30.0
print("calibrated tokens/unit = %.3f" % per, flush=True)
toks("warmup global " * 20, 1)
out = open(outpath, "a")
for (name, target, np_, nw, nm) in json.load(open(cellsp)):
    reps = max(1, int(round(target / per)))
    text = UNIT * reps + (" [probe-" + name + " " +
           "".join(random.choices(string.ascii_lowercase, k=8)) + "]")
    rows = []
    for i in range(nw + nm):
        d, t = toks(text, np_)
        rows.append({"run": i, "discard": i < nw, "cache_n": t.get("cache_n"),
                     "prompt_n": t["prompt_n"], "prompt_ms": round(t["prompt_ms"], 1),
                     "pp_tps": round(t["prompt_per_second"], 2),
                     "pred_n": t["predicted_n"], "pred_ms": round(t["predicted_ms"], 1),
                     "tg_tps": round(t["predicted_per_second"], 3),
                     "content_head": d["content"][:40]})
    rec = {"cell": name, "target": target, "n_predict": np_, "per_unit": round(per, 4),
           "n_warm": nw, "n_meas": nm, "prompt_n_first": rows[0]["prompt_n"],
           "rows": rows, "measured": [r for r in rows if not r["discard"]], "t": time.time()}
    print(json.dumps(rec), flush=True); out.write(json.dumps(rec) + "\n"); out.flush()
out.close(); print("DONE", flush=True)
PY
