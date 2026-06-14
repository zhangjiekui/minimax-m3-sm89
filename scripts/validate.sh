#!/usr/bin/env bash
# Validate the running MiniMax-M3 endpoint: needle retrieval + throughput.
# Exits non-zero if any needle is missed.
set -euo pipefail
BASE=${BASE:-http://localhost:8000}
MODEL=${MODEL:-minimax-m3}

echo "=== 1. model + context ==="
curl -s "$BASE/v1/models" | python3 -c "import sys,json;m=json.load(sys.stdin)['data'][0];print('  model:',m['id'],'max_model_len:',m['max_model_len'])"

echo "=== 2. needle retrieval (~40K, end + middle) ==="
python3 - "$BASE" "$MODEL" <<'PY'
import json, sys, time, urllib.request
base, model = sys.argv[1], sys.argv[2]
fill = "The apparatus measures conductivity in millisiemens per meter. "
def ask(prompt, label):
    body = json.dumps({"model": model, "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 12, "temperature": 0,
        "chat_template_kwargs": {"thinking_mode": "disabled"}}).encode()
    t = time.time()
    d = json.loads(urllib.request.urlopen(urllib.request.Request(base + "/v1/chat/completions",
        data=body, headers={"Content-Type": "application/json"}), timeout=180).read())
    out = (d["choices"][0]["message"].get("content") or "").strip()
    pt = d["usage"]["prompt_tokens"]
    print(f"  [{label}] pt={pt} lat={time.time()-t:.1f}s -> {out!r}")
    return out
# warmup
try: ask("hi", "warmup")
except: pass
time.sleep(1)
r1 = ask(fill*2500 + " The vault code is ZEBRA-7291. What is the code? Just the code.", "END")
r2 = ask(fill*1250 + " The vault code is MANGO-3188. " + fill*1250 + " What is the code? Just the code.", "MID")
ok = ("ZEBRA-7291" in r1) and ("MANGO-3188" in r2)
print("  needles:", "PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
PY

echo "=== 3. throughput ==="
python3 - "$BASE" "$MODEL" <<'PY'
import json, sys, time, urllib.request
base, model = sys.argv[1], sys.argv[2]
def stream(prompt, mt, label):
    body = json.dumps({"model": model, "messages": [{"role": "user", "content": prompt}],
        "max_tokens": mt, "temperature": 0, "stream": True,
        "stream_options": {"include_usage": True},
        "chat_template_kwargs": {"thinking_mode": "disabled"}}).encode()
    t0 = time.time(); ttft = None; gen = 0; usage = None
    for line in urllib.request.urlopen(urllib.request.Request(base + "/v1/chat/completions",
        data=body, headers={"Content-Type": "application/json"}), timeout=200):
        s = line.decode(errors="ignore").strip()
        if not s.startswith("data:"): continue
        d = s[5:].strip()
        if d == "[DONE]": break
        try: ev = json.loads(d)
        except: continue
        if ev.get("usage"): usage = ev["usage"]
        for c in ev.get("choices", []):
            if (c.get("delta") or {}).get("content"):
                if ttft is None: ttft = time.time() - t0
                gen += 1
    total = time.time() - t0
    pt = usage.get("prompt_tokens") if usage else None
    ct = usage.get("completion_tokens") if usage else gen
    dt = total - ttft if ttft else 0
    print(f"  [{label}] pt={pt} gen={ct} TTFT={ttft:.2f}s "
          f"prefill={pt/ttft:.0f} tok/s decode={ct/dt:.0f} tok/s" if pt and ttft and ct else f"  [{label}] (incomplete)")
stream("List 50 even numbers comma separated starting at 2.", 150, "decode")
PY
echo "=== done ==="
