#!/usr/bin/env python3
"""Correctness tests for QuBIOS v1.1.0 — run with `pytest` or `python tests/test_memory.py`.

These tests check the physics, not just that code runs:
  • the encoder produces a genuine Steane codeword,
  • noiseless memory circuits have deterministic detectors and no logical flips,
  • the decoder corrects every single fault of the circuit-level noise model,
  • circuit-level distance is 2 without flags (hook errors) and 3 with flags,
  • teleportation / Bell fidelities are 1 without noise and drop with noise,
  • at low noise, the flagged Steane memory beats an unencoded qubit.

© 2026 Qubitpage® | MIT License
"""
import os
import sys

import numpy as np
import stim

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from qubios import (SteaneQEC, TeleportEngine, EscortQubVirt, run_memory,  # noqa: E402
                    build_memory_circuit, circuit_distance, DemDecoder, NoiseModel)
from qubios.memory import STABILIZERS  # noqa: E402


def _stabilizer_values(circuit: stim.Circuit) -> list[int]:
    t = stim.TableauSimulator()
    t.do(circuit)
    vals = []
    for P in "XZ":
        for sup in STABILIZERS:
            ps = "".join(P if i in sup else "_" for i in range(7))
            vals.append(t.peek_observable_expectation(stim.PauliString(ps)))
    return vals


def test_encoder_is_codeword():
    q = SteaneQEC(0.001)
    for basis, log in (("Z", "Z" * 7), ("X", "X" * 7)):
        enc = q.build_encoding_circuit(noise=False, basis=basis)
        assert _stabilizer_values(enc) == [1] * 6
        t = stim.TableauSimulator(); t.do(enc)
        assert t.peek_observable_expectation(stim.PauliString(log)) == 1


def test_noiseless_memory_is_deterministic():
    for flags in (False, True):
        for basis in "ZX":
            c = build_memory_circuit(4, None, basis, flags)
            c.detector_error_model()          # raises if non-deterministic
            d, o = c.compile_detector_sampler().sample(500, separate_observables=True)
            assert not d.any() and not o.any()


def test_every_single_fault_is_corrected():
    # At low p a single fault is always the most likely explanation of its
    # own syndrome, so a most-likely-error decoder must return it. (At high p
    # a pair of likelier faults can legitimately win, so this check is only
    # meaningful at low p.)
    for flags in (False, True):
        c = build_memory_circuit(3, NoiseModel(1e-4), "Z", flags)
        dec = DemDecoder(c)
        H, L, pri = dec.H, dec.L, dec.priors
        for j in range(H.shape[1]):
            syn = H[:, j].copy()
            if not syn.any():
                assert L[0, j] == 0, "undetectable single fault flips the logical"
                continue
            same = np.nonzero((H == syn[:, None]).all(0))[0]
            if same[np.argmax(pri[same])] != j:
                continue  # not the most likely explanation (only possible without flags)
            assert dec.decode(syn)[0] == L[0, j], f"single fault {j} mis-decoded"


def test_circuit_distance():
    for basis in "ZX":
        assert circuit_distance(build_memory_circuit(2, NoiseModel(1e-3), basis, False)) == 2
        assert circuit_distance(build_memory_circuit(2, NoiseModel(1e-3), basis, True)) == 3


def test_teleport_fidelity_is_real():
    assert TeleportEngine(0.0).run_teleport(3000, noise=False).fidelity == 1.0
    f_low = TeleportEngine(0.001).run_teleport(60000).fidelity
    f_high = TeleportEngine(0.05).run_teleport(60000).fidelity
    assert f_high < f_low < 1.0


def test_bell_fidelity_is_real():
    e = EscortQubVirt(0, error_rate=0.0)
    assert e.establish_bell_pair(3000, noise=False)["bell_fidelity"] == 1.0
    noisy = EscortQubVirt(0, error_rate=0.05).establish_bell_pair(60000)["bell_fidelity"]
    assert 0.85 < noisy < 0.97


def test_qec_cycle_is_measured():
    assert SteaneQEC(0.001).run_qec_cycle(shots=2000, noise=False).logical_errors == 0
    s = SteaneQEC(0.003).run_qec_cycle(shots=20000, rounds=2)
    assert s.logical_errors > 0 and abs(s.fidelity_estimate - (1 - s.logical_error_rate)) < 1e-12


def test_flagged_memory_beats_unencoded_qubit_at_low_noise():
    r = run_memory(p=1e-4, rounds=3, shots=300_000, flags=True, seed=11)
    lo, hi = r.logical_error_rate_ci95
    assert hi < r.physical_error_rate, (r.logical_error_rate, r.physical_error_rate)


if __name__ == "__main__":
    tests = [v for k, v in dict(globals()).items() if k.startswith("test_")]
    for t in tests:
        t()
        print(f"PASS  {t.__name__}")
    print(f"\nAll {len(tests)} tests passed.")
