#!/usr/bin/env python3
"""Smoke tests: every public QuBIOS class still runs after the v1.1.0 fixes.

Physics/correctness tests live in tests/test_memory.py.
Run with `pytest` or `python tests/test_core.py`.

© 2026 Qubitpage® | MIT License
"""
import os
import sys
import warnings

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from qubios import (QubiLogicEngine, SteaneQEC, TransitRing, VirtualQubitManager,  # noqa: E402
                    CircuitCutter, SuperdenseEngine, EntanglementDistiller, EscortQubVirt)


def test_superdense():
    r = SuperdenseEngine(error_rate=0.001).run_all_messages(shots=4000)
    assert r["overall_accuracy"] > 0.98


def test_distillation_improves_correlation():
    d = EntanglementDistiller().run_distillation(shots=40_000, noise_rate=0.02)
    assert d["post_fidelity"] > d["pre_fidelity"]


def test_escort_refresh_runs():
    e = EscortQubVirt(0, error_rate=0.002)
    e.establish_bell_pair(3000)
    out = e.check_and_refresh(shots=2000)
    assert "qec" in out and 0.0 <= out["fidelity_after"] <= 1.0


def test_transit_ring_and_virtual_qubits_run():
    ring = TransitRing(n_blocks=3, error_rate=0.001)
    ring.inject_state(shots=2000)
    cyc = ring.relay_full_cycle(shots=2000)
    assert len(cyc["hops"]) == 3
    vqm = VirtualQubitManager(ring, error_rate=0.001)
    b = vqm.park_unpark_benchmark(n_qubits=2, shots=2000)
    assert 0.0 <= b["round_trip_fidelity"] <= 1.0


def test_circuit_cutter_runs():
    r = CircuitCutter(error_rate=0.001).run_cut_circuit(n_qubits=4, n_cuts=1, shots=2000)
    assert r.get("success", True)


def test_modelled_benchmark_is_labelled():
    eng = QubiLogicEngine(n_escorts=1, n_transit=3, error_rate=0.001)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        r = eng.benchmark_with_buffer("bell", shots=1000, noise_rate=0.001, idle_rounds=2)
    assert r["method"].startswith("MODELLED")
    assert any("MODEL" in str(x.message) for x in w)


def test_qec_standalone_table():
    print(f"\n{'p_phys':>8} {'p_logical (measured)':>22} {'shots w/ detection':>20}")
    for p in [0.0005, 0.001, 0.003]:
        s = SteaneQEC(physical_error_rate=p).run_qec_cycle(shots=20_000, rounds=1)
        print(f"{p:>8.4f} {s.logical_error_rate:>22.2e} {s.errors_detected:>20d}")
        assert 0.0 <= s.logical_error_rate <= 1.0


if __name__ == "__main__":
    tests = [v for k, v in dict(globals()).items() if k.startswith("test_")]
    for t in tests:
        t()
        print(f"PASS  {t.__name__}")
    print(f"\nAll {len(tests)} smoke tests passed.")
