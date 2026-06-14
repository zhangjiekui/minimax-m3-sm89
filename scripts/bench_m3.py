#!/usr/bin/env python3
"""Throughput bench for MiniMax-M3 (no /tokenize dependency)."""
import json, time, sys, urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

BASE = "http://127.0.0.1:8000"
MODEL = "minimax-m3"
FILLER = ("The differential conductivity of the alloyed semiconductor substrate is "
          "characterized under standard temperature and pressure in millisiemens per "
          "unit length, recorded by the calibrated Hall-probe apparatus. ")  # ~30 tok

def build_prompt(n_tokens):
    reps = max(1, n_tokens // 30)
    body = FILLER * reps
    body += f" The unique access code for this session is ZEBRA-7291. \n\nWhat is the unique access code? Reply with only the code."
    return body

def stream_one(prompt, max_tokens, label):
    body = json.dumps({
        "model": MODEL, "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens, "temperature": 0, "stream": True,
        "stream_options": {"include_usage": True},
        "chat_template_kwargs": {"thinking_mode": "disabled"},
    }).encode()
    t0 = time.time(); ttft = None; chunks = []; usage = None
    try:
        resp = urllib.request.urlopen(urllib.request.Request(BASE + "/v1/chat/completions",
            data=body, headers={"Content-Type": "application/json"}), timeout=900)
        for line in resp:
            s = line.decode(errors="ignore").strip()
            if not s.startswith("data:"): continue
            d = s[5:].strip()
            if d == "[DONE]": break
            try: ev = json.loads(d)
            except: continue
            if ev.get("usage"): usage = ev["usage"]
            for c in ev.get("choices", []):
                p = (c.get("delta") or {}).get("content") or ""
                if p:
                    if ttft is None: ttft = time.time() - t0
                    chunks.append(p)
    except Exception as e:
        print(f"  [{label}] ERROR {e!r}"); return None
    total = time.time() - t0; text = "".join(chunks)
    pt = usage.get("prompt_tokens") if usage else None
    ct = usage.get("completion_tokens") if usage else len(chunks)
    dt = total - ttft if ttft else 0
    print(f"  [{label}] pt={pt} gen={ct} TTFT={ttft:.2f}s total={total:.2f}s")
    if pt and ttft: print(f"          PREFILL = {pt/ttft:.0f} tok/s")
    if ct and dt: print(f"          DECODE  = {ct/dt:.1f} tok/s")
    print(f"          out={text[:60]!r}")
    return {"pt": pt, "ct": ct, "ttft": ttft, "total": total, "prefill": pt/ttft if pt and ttft else None,
            "decode": ct/dt if ct and dt else None}

def bench_concurrent(n_tok, concurrency, gen=128):
    print(f"\n=== CONCURRENCY {concurrency}x @ ~{n_tok} tok, {gen} tok decode ===", flush=True)
    prompt = build_prompt(n_tok)
    t0 = time.time(); results = []
    with ThreadPoolExecutor(max_workers=concurrency) as ex:
        futs = [ex.submit(stream_one, prompt, gen, f"c{i}") for i in range(concurrency)]
        for f in as_completed(futs): results.append(f.result())
    wall = time.time() - t0
    ok = [r for r in results if r]
    if not ok: print("  ALL ERRORED"); return
    pre = [r["prefill"] for r in ok if r["prefill"]]; dec = [r["decode"] for r in ok if r["decode"]]
    ttfts = [r["ttft"] for r in ok]; tot_gen = sum(r["ct"] for r in ok)
    print(f"  ok={len(ok)}/{concurrency} wall={wall:.1f}s  pt~{ok[0]['pt']}")
    print(f"  TTFT: min={min(ttfts):.1f}s max={max(ttfts):.1f}s avg={sum(ttfts)/len(ttfts):.1f}s")
    if pre: print(f"  PREFILL/req: {sum(pre)/len(pre):.0f} tok/s avg")
    if dec: print(f"  DECODE/req: {sum(dec)/len(dec):.1f} tok/s avg | AGGREGATE decode={tot_gen/wall:.1f} tok/s")

if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "all"
    if mode == "decode":
        print("=== DECODE ==="); stream_one("List 60 even numbers comma-separated starting at 2.", 200, "decode")
        stream_one("List 60 even numbers comma-separated starting at 2.", 200, "decode-warm")
    elif mode == "prefill":
        print("=== PREFILL sweep ===")
        for n in [8000, 32000, 65536, 131072, 192000]:
            print(f"\n-- prefill ~{n} tok --"); stream_one(build_prompt(n), 12, f"p{n}")
    elif mode == "concurrent":
        bench_concurrent(int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]) if len(sys.argv) > 4 else 128)
    elif mode == "all":
        stream_one("List 60 even numbers comma-separated starting at 2.", 150, "decode-warm")
        for n in [32000, 131072]:
            print(f"\n-- prefill ~{n} tok --"); stream_one(build_prompt(n), 12, f"p{n}")
        bench_concurrent(4000, 4, 128)
