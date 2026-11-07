#!/usr/bin/env python3
"""Measured benchmark: Steane [[7,1,3]] memory vs. an unencoded qubit.

Run:  python benchmarks/steane_memory.py            (≈ 5–15 minutes)
      python benchmarks/steane_memory.py --quick    (≈ 10 seconds)

Writes benchmarks/results/steane_memory.json and prints a Markdown table.
Every number is measured from sampled, decoded Stim circuits.

© 2026 Qubitpage® | MIT License
"""
import argparse, json, os, sys, time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from qubios.memory import run_memory, per_round_error  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--quick", action="store_true")
ap.add_argument("--seed", type=int, default=2026)
args = ap.parse_args()

P_LIST = [1e-4, 3e-4, 1e-3, 3e-3] if not args.quick else [1e-3, 3e-3]
ROUNDS = [1, 3, 10] if not args.quick else [3]
TARGET_FAILS, MAX_SHOTS = (200, 2_000_000) if not args.quick else (50, 200_000)

rows = []
t0 = time.time()
MID = ROUNDS[len(ROUNDS) // 2]
CONFIGS = ([("Z", f, r, 1.0) for f in (False, True) for r in ROUNDS] +
           [("X", f, MID, 1.0) for f in (False, True)] +           # basis symmetry check
           [("Z", f, MID, 0.1) for f in (False, True)])            # quieter idling (closer to hardware)
for basis, flags, r, idle in CONFIGS:
    for p in P_LIST:
        if True:
            shots = 20_000
            while True:
                res = run_memory(p=p, rounds=r, shots=shots, basis=basis, seed=args.seed, flags=flags, idle_factor=idle)
                if res.logical_errors >= TARGET_FAILS or shots >= MAX_SHOTS:
                    break
                need = TARGET_FAILS / max(res.logical_error_rate, TARGET_FAILS / (shots * 50))
                shots = int(min(MAX_SHOTS, max(shots * 2, need * 1.2)))
            d = res.to_dict()
            d["idle_factor"] = idle
            d["logical_per_round"] = per_round_error(res.logical_error_rate, r)
            d["physical_per_round"] = per_round_error(res.physical_error_rate, r)
            d["ratio_logical_over_physical"] = (
                res.logical_error_rate / res.physical_error_rate if res.physical_error_rate else None)
            rows.append(d)
            print(f"flags={flags!s:5} idle={idle} basis={basis} p={p:.0e} rounds={r:>2} shots={shots:>8} "
                  f"P_L={res.logical_error_rate:.2e} P_phys={res.physical_error_rate:.2e} "
                  f"ratio={d['ratio_logical_over_physical']:.2f}", flush=True)

os.makedirs(os.path.join(os.path.dirname(__file__), "results"), exist_ok=True)
out = os.path.join(os.path.dirname(__file__), "results", "steane_memory.json")
with open(out, "w") as f:
    json.dump({"generated_s": round(time.time() - t0, 1), "seed": args.seed,
               "noise_model": "uniform circuit-level depolarizing p (gates, idle, reset, measure)",
               "decoder": "BP+OSD (ldpc) on Stim detector error model",
               "rows": rows}, f, indent=2)

print("\n| Syndrome circuit | Idle noise | Basis | p | Rounds | Shots | Logical error (Steane, 95% CI) | Physical error (no QEC) | Logical / physical |")
print("|---|---|---|---|---|---|---|---|---|")
for d in rows:
    lo, hi = d["logical_error_rate_ci95"]
    print(f"| {'flagged' if d['extra']['flags'] else 'basic'} | {d['idle_factor']:g}·p | {d['basis']} | {d['p']:.0e} | {d['rounds']} | {d['shots']:,} | "
          f"{d['logical_error_rate']:.2e} ({lo:.1e}–{hi:.1e}) | {d['physical_error_rate']:.2e} | "
          f"{d['ratio_logical_over_physical']:.2f}× |")
print(f"\nSaved {out}  ({time.time() - t0:.0f}s)")
