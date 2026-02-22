#!/usr/bin/env python3
"""QuBIOS Quick Start Example — Qubitpage®.

https://qubitpage.com | https://github.com/qubitpage/QuBIOS
"""
from qubios import QubiLogicEngine, SteaneQEC, TransitRing, EntanglementDistiller

# ─── 1. Bell State Benchmark ─────────────────────────────────────────────────
print("=== Bell State: No Buffer vs QuBIOS ===")
engine = QubiLogicEngine(n_escorts=2, n_transit=3, error_rate=0.001)

no_buf = engine.benchmark_no_buffer(
    circuit_type="bell", shots=4096, noise_rate=0.001, idle_rounds=10)
with_buf = engine.benchmark_with_buffer(
    circuit_type="bell", shots=4096, noise_rate=0.001, idle_rounds=10)

print(f"Without QuBIOS: {no_buf['fidelity']:.6f}")
print(f"With QuBIOS:    {with_buf['effective_fidelity']:.6f}")
print(f"Improvement:    +{with_buf['effective_fidelity'] - no_buf['fidelity']:.6f}")

# ─── 2. Steane QEC Standalone ────────────────────────────────────────────────
print("\n=== Steane [[7,1,3]] QEC ===")
qec = SteaneQEC(physical_error_rate=0.001)
stats = qec.run_qec_cycle(shots=10000, rounds=1)
print(f"Errors detected:      {stats.errors_detected}")
print(f"Errors corrected:     {stats.errors_corrected}")
print(f"Physical error rate:  {stats.physical_error_rate:.4f}")
print(f"Logical error rate:   {stats.logical_error_rate:.2e}")
print(f"Fidelity estimate:    {stats.fidelity_estimate:.6f}")

# ─── 3. Transit Ring ─────────────────────────────────────────────────────────
print("\n=== Transit Ring (3 hops) ===")
ring = TransitRing(n_blocks=3, error_rate=0.001)
inject = ring.inject_state(shots=5000)
print(f"Injected — teleport fidelity: {inject['teleport']['fidelity']:.4f}")
for i in range(3):
    hop = ring.relay_hop(shots=5000)
    print(f"Hop {i+1}: fidelity={hop['teleport']['fidelity']:.4f}, "
          f"block={hop['current_block']}, time={hop['time_us']:.0f} μs")

# ─── 4. Entanglement Distillation ────────────────────────────────────────────
print("\n=== BBPSSW Entanglement Distillation ===")
distiller = EntanglementDistiller(error_rate=0.001)
result = distiller.multi_round_distill(shots=50000, noise_rate=0.01, rounds=3)
print(f"Initial fidelity: {result['initial_fidelity']:.6f}")
print(f"Final fidelity:   {result['final_fidelity']:.6f}")
print(f"Total improvement: +{result['total_improvement_pct']:.2f}%")
print(f"Rounds: {result['rounds']}")
