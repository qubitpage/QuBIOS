#!/usr/bin/env python3
"""QuBIOS real-hardware test: Steane [[7,1,3]] memory vs. an unencoded qubit
on IBM Quantum — Qubitpage®.

What it runs (all in ONE Sampler job, ~20–60 s of QPU time):
  • Steane memory, basic and flagged syndrome extraction, 1 and 2 rounds,
    logical |0_L> and |1_L>                                  → 8 circuits
  • Unencoded qubit prepared in |0> and |1>, idling exactly as long as each
    transpiled Steane circuit (same wall-clock time)          → 8 circuits

The circuits are converted 1:1 from the Stim circuits validated in
tests/test_memory.py, so the hardware bits are decoded with the same tested
decoder (Tesseract on the Stim detector error model).

USAGE (Windows PowerShell / Linux):
  pip install -r hardware/requirements-hardware.txt

  # 1) Free local rehearsal on IBM's published Fez noise model (no account used)
  python hardware/ibm_steane_test.py --simulate

  # 2) Real hardware (uses your IBM Quantum minutes)
  python hardware/ibm_steane_test.py --key-file C:\\path\\to\\apikey.json --backend ibm_fez

  # 3) Re-analyse a finished job later without spending more time
  python hardware/ibm_steane_test.py --key-file apikey.json --job-id <JOB_ID>

The API key is read from --key-file (IBM's downloaded JSON, field "apikey")
or the QISKIT_IBM_TOKEN environment variable. It is never printed or saved.

© 2026 Qubitpage® | MIT License
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sys

import numpy as np
import stim

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from qubios import __version__  # noqa: E402
from qubios.memory import (DemDecoder, NoiseModel, build_memory_circuit,  # noqa: E402
                           wilson_ci)

from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister  # noqa: E402
from qiskit.transpiler import generate_preset_pass_manager  # noqa: E402

DECODER_P = 3e-3       # noise strength assumed when building decoder priors
SHOTS = 4000


# ── Stim → Qiskit (exact, gate by gate) ─────────────────────────────────────
def stim_to_qiskit(c: stim.Circuit, name: str) -> QuantumCircuit:
    c = c.without_noise().flattened()
    nq, nm = c.num_qubits, c.num_measurements
    qc = QuantumCircuit(QuantumRegister(nq, "q"), ClassicalRegister(nm, "c"), name=name)
    k = 0
    for inst in c:
        g = inst.name
        ts = [t.value for t in inst.targets_copy()]
        if g in ("TICK", "DETECTOR", "OBSERVABLE_INCLUDE", "QUBIT_COORDS", "SHIFT_COORDS"):
            continue
        if g == "R":
            for q in ts: qc.reset(q)
        elif g == "RX":
            for q in ts: qc.reset(q); qc.h(q)
        elif g in ("H", "X", "Z", "S"):
            for q in ts: getattr(qc, g.lower())(q)
        elif g == "CX":
            for a, b in zip(ts[::2], ts[1::2]): qc.cx(a, b)
        elif g == "M":
            for q in ts: qc.measure(q, k); k += 1
        elif g == "MX":
            for q in ts: qc.h(q); qc.measure(q, k); k += 1
        else:
            raise ValueError(f"unsupported Stim gate {g}")
    assert k == nm
    return qc


def baseline_circuit(logical_one: bool, duration_s: float, name: str) -> QuantumCircuit:
    qc = QuantumCircuit(1, 1, name=name)
    qc.reset(0)
    if logical_one:
        qc.x(0)
    qc.delay(duration_s, 0, unit="s")
    qc.measure(0, 0)
    return qc


def build_experiments(backend):
    pm = generate_preset_pass_manager(optimization_level=3, backend=backend, seed_transpiler=7)
    exps, pubs = [], []
    for flags in (False, True):
        for rounds in (1, 2):
            for one in (False, True):
                sc = build_memory_circuit(rounds, None, "Z", flags, logical_one=one)
                tag = f"steane_{'flag' if flags else 'basic'}_R{rounds}_L{int(one)}"
                qc_t = pm.run(stim_to_qiskit(sc, tag))
                try:
                    dur = qc_t.estimate_duration(backend.target, unit="s")
                except Exception:
                    dur = None
                ops = qc_t.count_ops()
                twoq = sum(v for g, v in ops.items() if g in ("cz", "ecr", "cx"))
                exps.append({"name": tag, "kind": "steane", "flags": flags, "rounds": rounds,
                             "logical_one": one, "duration_s": dur, "two_qubit_gates": twoq,
                             "depth": qc_t.depth()})
                pubs.append(qc_t)
                bdur = dur if dur else 5e-6 * rounds
                bq = pm.run(baseline_circuit(one, bdur, tag.replace("steane", "bare")))
                exps.append({"name": tag.replace("steane", "bare"), "kind": "bare", "flags": flags,
                             "rounds": rounds, "logical_one": one, "duration_s": bdur})
                pubs.append(bq)
    return exps, pubs


# ── Analysis ────────────────────────────────────────────────────────────────
def analyse(exps, bit_arrays):
    rows = []
    for e, bits in zip(exps, bit_arrays):
        meas = bits.to_bool_array(order="little")          # [shots, clbits], clbit 0 first
        n = meas.shape[0]
        if e["kind"] == "bare":
            wrong = int((meas[:, 0] != e["logical_one"]).sum())
            e.update(shots=n, errors=wrong, error_rate=wrong / n, ci95=wilson_ci(wrong, n))
        else:
            sc = build_memory_circuit(e["rounds"], NoiseModel(DECODER_P), "Z", e["flags"],
                                      logical_one=e["logical_one"])
            det, obs = sc.compile_m2d_converter().convert(measurements=meas, separate_observables=True)
            dec = DemDecoder(sc)
            pred = np.array([dec.decode(d)[0] for d in det], dtype=bool)
            wrong = int((pred ^ obs[:, 0]).sum())
            clean = ~det.any(axis=1)
            ps_wrong = int(obs[clean, 0].sum())
            e.update(shots=n, errors=wrong, error_rate=wrong / n, ci95=wilson_ci(wrong, n),
                     raw_logical_error=float(obs[:, 0].mean()),
                     postselected_keep=float(clean.mean()),
                     postselected_error=(ps_wrong / int(clean.sum())) if clean.any() else None,
                     detection_rate_per_detector=float(det.mean()))
        rows.append(e)
    return rows


def summarise(rows):
    print(f"\n{'experiment':<24}{'shots':>7}{'error':>10}{'95% CI':>20}{'2q gates':>10}{'dur (µs)':>10}")
    for r in rows:
        lo, hi = r["ci95"]
        d = f"{r['duration_s'] * 1e6:.1f}" if r.get("duration_s") else "-"
        print(f"{r['name']:<24}{r['shots']:>7}{r['error_rate']:>10.4f}{f'{lo:.3f}–{hi:.3f}':>20}"
              f"{r.get('two_qubit_gates', ''):>10}{d:>10}")
    print("\nSteane vs bare (average of |0> and |1>):")
    for flags in (False, True):
        for rounds in (1, 2):
            st = [r for r in rows if r["kind"] == "steane" and r["flags"] == flags and r["rounds"] == rounds]
            ba = [r for r in rows if r["kind"] == "bare" and r["flags"] == flags and r["rounds"] == rounds]
            s = np.mean([r["error_rate"] for r in st]); b = np.mean([r["error_rate"] for r in ba])
            ps = [r["postselected_error"] for r in st if r["postselected_error"] is not None]
            print(f"  {'flagged' if flags else 'basic  '} R={rounds}: Steane {s:.4f}  bare {b:.4f}  "
                  f"ratio {s / b if b else float('inf'):.2f}×   | error-DETECTION (post-select): "
                  f"{np.mean(ps):.4f} keeping {np.mean([r['postselected_keep'] for r in st]) * 100:.1f}%")


# ── Main ────────────────────────────────────────────────────────────────────
def load_token(path):
    if path:
        with open(path, encoding="utf-8") as f:
            return json.load(f)["apikey"]
    tok = os.environ.get("QISKIT_IBM_TOKEN")
    if not tok:
        sys.exit("Give --key-file apikey.json or set QISKIT_IBM_TOKEN")
    return tok


def main():
    ap = argparse.ArgumentParser(description=f"QuBIOS v{__version__} IBM hardware test")
    ap.add_argument("--simulate", action="store_true", help="local run on FakeFez noise model")
    ap.add_argument("--key-file"); ap.add_argument("--backend", default=None)
    ap.add_argument("--instance", default=None); ap.add_argument("--job-id", default=None)
    ap.add_argument("--shots", type=int, default=SHOTS)
    ap.add_argument("--yes", action="store_true", help="skip the confirmation prompt")
    args = ap.parse_args()

    from qiskit_ibm_runtime import SamplerV2
    if args.simulate:
        from qiskit_ibm_runtime.fake_provider import FakeFez
        backend, mode = FakeFez(), "LOCAL SIMULATION (FakeFez noise model — NOT hardware)"
    else:
        from qiskit_ibm_runtime import QiskitRuntimeService
        svc = QiskitRuntimeService(channel="ibm_quantum_platform", token=load_token(args.key_file),
                                   instance=args.instance)
        if args.job_id:
            job = svc.job(args.job_id)
            backend = job.backend() if callable(getattr(job, "backend", None)) else svc.backend(job.backend().name)
        else:
            backend = svc.backend(args.backend) if args.backend else svc.least_busy(operational=True, simulator=False, min_num_qubits=19)
        mode = f"REAL HARDWARE: {backend.name}"
    print(f"QuBIOS v{__version__} — {mode}")

    exps, pubs = build_experiments(backend)
    for e in exps:
        if e["kind"] == "steane":
            print(f"  {e['name']:<22} 2q gates={e['two_qubit_gates']:>4} depth={e['depth']:>4} "
                  f"duration={e['duration_s'] * 1e6 if e['duration_s'] else float('nan'):.1f} µs")

    if args.job_id:
        job = svc.job(args.job_id)
    else:
        est = len(pubs) * args.shots * 300e-6 + 15
        print(f"\n{len(pubs)} circuits × {args.shots} shots — rough QPU estimate ≈ {est:.0f} s")
        if not args.simulate and not args.yes and input("Submit to IBM? [y/N] ").strip().lower() != "y":
            sys.exit("Cancelled.")
        sampler = SamplerV2(mode=backend)
        job = sampler.run(pubs, shots=args.shots)
        print(f"Job ID: {job.job_id()}  (save this; re-analyse with --job-id)")
    result = job.result()
    bit_arrays = [res.data.c if hasattr(res.data, "c") else next(iter(res.data.values())) for res in result]
    rows = analyse(exps, bit_arrays)
    summarise(rows)

    out_dir = os.path.join(os.path.dirname(__file__), "results")
    os.makedirs(out_dir, exist_ok=True)
    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    out = os.path.join(out_dir, f"{'sim_fakefez' if args.simulate else backend.name}_{stamp}.json")
    meta = {"qubios_version": __version__, "mode": mode, "backend": backend.name,
            "job_id": None if args.simulate else job.job_id(), "shots": args.shots,
            "decoder": "tesseract on Stim DEM", "decoder_prior_p": DECODER_P,
            "timestamp": stamp, "rows": rows}
    if not args.simulate:
        try:
            meta["usage"] = job.usage() if callable(getattr(job, "usage", None)) else None
            meta["metrics"] = job.metrics()
        except Exception:
            pass
    with open(out, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, default=str)
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
