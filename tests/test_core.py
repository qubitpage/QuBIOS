#!/usr/bin/env python3
"""QuBIOS Test Suite — Qubitpage® Quantum BIOS Framework.

Rigorous comparison tests for QubiLogic dual-layer protection.
Tests: Bell, GHZ-3, GHZ-5, Superposition circuits.
Methods: No buffer, Escort-only, Escort+Transit.

Real Stim circuits — no mocks, no pre-computed results.

https://qubitpage.com | https://github.com/qubitpage/QuBIOS
"""
import json
import time
from qubios import QubiLogicEngine, SteaneQEC


def run_qec_standalone():
    """Test Steane QEC independently at various error rates."""
    print("=" * 70)
    print("  STEANE [[7,1,3]] QEC — Standalone Test")
    print("  5000 shots | 1 QEC round | Real Stim circuits")
    print("=" * 70)
    print(f"{'p_phys':>10} {'detected':>10} {'corrected':>10} "
          f"{'corr_rate':>10} {'p_logical':>12} {'fidelity':>10}")
    print("-" * 70)

    for p in [0.0001, 0.0005, 0.001, 0.003, 0.005, 0.01, 0.05]:
        qec = SteaneQEC(physical_error_rate=p)
        stats = qec.run_qec_cycle(shots=5000, rounds=1, noise=True)
        print(f"{p:>10.4f} {stats.errors_detected:>10d} "
              f"{stats.errors_corrected:>10d} "
              f"{stats.correction_success_rate:>10.4f} "
              f"{stats.logical_error_rate:>12.8f} "
              f"{stats.fidelity_estimate:>10.6f}")
    print()


def run_bell_comparison():
    """Test Bell state across noise/idle parameter space."""
    print("=" * 70)
    print("  BELL STATE: No Buffer vs QuBIOS Buffer")
    print("=" * 70)
    print(f"{'noise':>8} {'idle':>6} {'no_buf':>10} {'escort':>10} "
          f"{'improv':>10} {'p_logical':>12}")
    print("-" * 70)

    for noise in [0.0, 0.0005, 0.001, 0.003, 0.005, 0.01]:
        for idle in [0, 5, 10, 20, 50]:
            engine = QubiLogicEngine(n_escorts=2, n_transit=3,
                                     error_rate=max(noise, 0.0001))
            nb = engine.benchmark_no_buffer(
                circuit_type="bell", shots=5000,
                noise_rate=noise, idle_rounds=idle)
            wb = engine.benchmark_with_buffer(
                circuit_type="bell", shots=5000,
                noise_rate=noise, idle_rounds=idle)

            nb_f = nb.get("fidelity", 0)
            wb_f = wb.get("effective_fidelity", 0)
            imp = round(wb_f - nb_f, 6)
            p_log = 21 * noise * noise

            print(f"{noise:>8.4f} {idle:>6d} {nb_f:>10.6f} {wb_f:>10.6f} "
                  f"{imp:>+10.6f} {p_log:>12.8f}")
    print()


def run_circuit_comparison():
    """Compare all 4 circuit types at fixed noise/idle."""
    print("=" * 70)
    print("  CIRCUIT TYPE COMPARISON (noise=0.005, idle=10)")
    print("=" * 70)
    print(f"{'circuit':>15} {'no_buf':>10} {'escort':>10} {'improv':>10}")
    print("-" * 55)

    for circ in ["bell", "ghz3", "ghz5", "superposition"]:
        engine = QubiLogicEngine(n_escorts=2, n_transit=3, error_rate=0.005)
        nb = engine.benchmark_no_buffer(
            circuit_type=circ, shots=5000,
            noise_rate=0.005, idle_rounds=10)
        wb = engine.benchmark_with_buffer(
            circuit_type=circ, shots=5000,
            noise_rate=0.005, idle_rounds=10)

        nb_f = nb.get("fidelity", 0)
        wb_f = wb.get("effective_fidelity", 0)
        imp = round(wb_f - nb_f, 6)
        print(f"{circ:>15} {nb_f:>10.6f} {wb_f:>10.6f} {imp:>+10.6f}")
    print()


def run_transit_comparison():
    """Compare escort-only vs escort+transit for long idle periods."""
    print("=" * 70)
    print("  TRANSIT RING TEST (noise=0.005)")
    print("=" * 70)
    print(f"{'idle':>6} {'no_buf':>10} {'escort':>10} {'transit':>10} "
          f"{'esc_imp':>10} {'tran_imp':>10}")
    print("-" * 70)

    for idle in [0, 3, 5, 10, 20, 50]:
        engine = QubiLogicEngine(n_escorts=2, n_transit=3, error_rate=0.005)
        nb = engine.benchmark_no_buffer(
            circuit_type="bell", shots=5000,
            noise_rate=0.005, idle_rounds=idle)

        engine2 = QubiLogicEngine(n_escorts=2, n_transit=3, error_rate=0.005)
        wb_esc = engine2.benchmark_with_buffer(
            circuit_type="bell", shots=5000,
            noise_rate=0.005, idle_rounds=idle,
            use_transit=False)

        wb_tra = None
        if idle >= 3:
            engine3 = QubiLogicEngine(n_escorts=2, n_transit=3, error_rate=0.005)
            wb_tra = engine3.benchmark_with_buffer(
                circuit_type="bell", shots=5000,
                noise_rate=0.005, idle_rounds=idle,
                use_transit=True)

        nb_f = nb.get("fidelity", 0)
        esc_f = wb_esc.get("effective_fidelity", 0)
        tra_f = wb_tra.get("effective_fidelity", 0) if wb_tra else 0
        esc_imp = round(esc_f - nb_f, 6)
        tra_imp = round(tra_f - nb_f, 6) if wb_tra else 0

        tra_str = f"{tra_f:>10.6f}" if wb_tra else "       N/A"
        imp_str = f"{tra_imp:>+10.6f}" if wb_tra else "       N/A"
        print(f"{idle:>6d} {nb_f:>10.6f} {esc_f:>10.6f} {tra_str} "
              f"{esc_imp:>+10.6f} {imp_str}")
    print()


def run_scaling_test():
    """Test how improvement scales with idle rounds at fixed noise."""
    print("=" * 70)
    print("  SCALING: Improvement vs Idle Rounds (noise=0.005, bell)")
    print("=" * 70)
    print(f"{'idle':>6} {'no_buf':>10} {'escort':>10} {'ratio':>10} {'abs_imp':>10}")
    print("-" * 50)

    for idle in [0, 1, 2, 5, 10, 20, 30, 50]:
        engine = QubiLogicEngine(n_escorts=2, n_transit=3, error_rate=0.005)
        nb = engine.benchmark_no_buffer(
            circuit_type="bell", shots=5000,
            noise_rate=0.005, idle_rounds=idle)
        wb = engine.benchmark_with_buffer(
            circuit_type="bell", shots=5000,
            noise_rate=0.005, idle_rounds=idle)

        nb_f = nb.get("fidelity", 0)
        wb_f = wb.get("effective_fidelity", 0)
        ratio = wb_f / nb_f if nb_f > 0 else 0
        imp = round(wb_f - nb_f, 6)
        print(f"{idle:>6d} {nb_f:>10.6f} {wb_f:>10.6f} {ratio:>10.4f}x {imp:>+10.6f}")
    print()


if __name__ == "__main__":
    t0 = time.time()
    print("\n" + "=" * 70)
    print("  QuBIOS — Qubitpage® Quantum BIOS Framework")
    print("  Full Test Suite | Real Stim circuits | No mocks")
    print("  https://qubitpage.com | https://github.com/qubitpage/QuBIOS")
    print("=" * 70 + "\n")

    run_qec_standalone()
    run_bell_comparison()
    run_circuit_comparison()
    run_transit_comparison()
    run_scaling_test()

    elapsed = time.time() - t0
    print(f"\nTotal test time: {elapsed:.1f} seconds")
    print("=" * 70)
    print("© 2026 Qubitpage® | https://qubitpage.com")
    print("=" * 70)
