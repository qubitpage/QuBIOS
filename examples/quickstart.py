#!/usr/bin/env python3
"""QuBIOS v1.1.0 quick start — Qubitpage®.

Everything printed here is MEASURED from sampled Stim circuits.
https://qubitpage.com | https://github.com/qubitpage/QuBIOS
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import qubios  # noqa: E402
from qubios import (run_memory, SteaneQEC, TeleportEngine, EscortQubVirt,  # noqa: E402
                    SuperdenseEngine, EntanglementDistiller)

print(f"QuBIOS v{qubios.__version__}\n")

# ─── 1. Steane memory vs. an unencoded qubit (same noise, same duration) ───
print("=== Steane [[7,1,3]] memory, p = 0.1 %, 3 rounds ===")
for flags in (False, True):
    r = run_memory(p=1e-3, rounds=3, shots=100_000, flags=flags, seed=1)
    lo, hi = r.logical_error_rate_ci95
    label = "flagged" if flags else "basic  "
    print(f"{label}: logical error {r.logical_error_rate:.2e} (95% CI {lo:.1e}–{hi:.1e})"
          f"  vs unencoded {r.physical_error_rate:.2e}"
          f"  → {r.logical_error_rate / r.physical_error_rate:.2f}×")

# ─── 2. The same through the classic SteaneQEC API ─────────────────────────
stats = SteaneQEC(physical_error_rate=1e-3).run_qec_cycle(shots=50_000, rounds=1)
print(f"\nSteaneQEC.run_qec_cycle: logical error {stats.logical_error_rate:.2e}, "
      f"shots with detections {stats.errors_detected}")

# ─── 3. Teleportation and Bell-pair fidelity (measured in 3 bases) ─────────
print("\n=== Teleportation / Bell pair, p = 0.1 % ===")
print(f"Teleport fidelity:  {TeleportEngine(1e-3).run_teleport(60_000).fidelity:.4f}")
print(f"Bell-pair fidelity: {EscortQubVirt(0, error_rate=1e-3).establish_bell_pair(60_000)['bell_fidelity']:.4f}")

# ─── 4. Superdense coding ──────────────────────────────────────────────────
sd = SuperdenseEngine(error_rate=1e-3).run_all_messages(shots=10_000)
print(f"\nSuperdense coding accuracy (all 4 messages): {sd['overall_accuracy']:.4f}")

# ─── 5. BBPSSW distillation (Z-basis correlation) ──────────────────────────
d = EntanglementDistiller().run_distillation(shots=50_000, noise_rate=0.01)
print(f"BBPSSW: ZZ-correlation {d['pre_fidelity']:.4f} → {d['post_fidelity']:.4f} after post-selection")
