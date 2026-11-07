<div align="center">

<img src="docs/logo.png" alt="Qubitpage®" width="220" />

[![lablab.ai 1st Place](https://img.shields.io/badge/🥇_1ST_PLACE-lablab.ai_AI_Meets_Robotics_2026-FFD700?style=for-the-badge&labelColor=1a1a2e)](https://lablab.ai/ai-hackathons/launch-fund-ai-meets-robotics)

# QuBIOS v1.2.2
### Measured quantum error-correction & entanglement toolkit, by **[Qubitpage®](https://qubitpage.com)**

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Stim](https://img.shields.io/badge/Stim-1.13+-green.svg)](https://github.com/quantumlib/Stim)
[![Tests](https://img.shields.io/badge/tests-15%2F15_passing-brightgreen.svg)](tests/)
[![Hardware](https://img.shields.io/badge/IBM_hardware-3_QPUs,_~40_jobs-blue.svg)](docs/HARDWARE_REPORT_2026-11-07.md)
[![Status](https://img.shields.io/badge/Status-Research%20%2F%20Educational-orange.svg)](#project-status)

</div>

---

## What is QuBIOS?

QuBIOS is a Python library for **measuring** how well quantum error correction protects a qubit. It builds circuit-level noisy [Stim](https://github.com/quantumlib/Stim) circuits, decodes every shot with a real decoder, and compares the result with an unprotected qubit under the same noise for the same time.

- **Steane [[7,1,3]] memory**: correct encoder, scheduled syndrome extraction, detectors and a logical observable
- **Flag-qubit syndrome extraction** ([Chao & Reichardt 2018](https://doi.org/10.1103/PhysRevLett.121.050502)): restores circuit-level distance 3 (it's 2 without flags, because of hook errors)
- **Real decoding**: most-likely-error decoding of Stim's detector error model with [Tesseract](https://github.com/quantumlib/tesseract-decoder)
- **Fair baseline**: an unencoded qubit idling for exactly as many time steps, with the same noise
- **Real IBM hardware test** ([hardware/ibm_steane_test.py](hardware/ibm_steane_test.py)): the same circuits, converted gate by gate to Qiskit, with the hardware bits decoded by the same decoder
- **IBM hardware experiment suite** ([hardware/colab/](hardware/colab/)): repetition and phase codes with software matching decoding, coherence-aware qubit placement, GHZ noise-scaling tests and the [[4,2,2]] logical Bell pair. Run on 3 IBM Heron QPUs; see the [hardware report](docs/HARDWARE_REPORT_2026-11-07.md)
- **Protocols**: teleportation with feed-forward, Bell-pair fidelity (XX/YY/ZZ), superdense coding, BBPSSW distillation, a circuit-cutting demo

---

## 🌐 Ecosystem

| Repository | Description |
|------------|-------------|
| **[QLang](https://github.com/qubitpage/QLang)** | Beginner-friendly quantum command language + browser SDK |
| **[QuBIOS](https://github.com/qubitpage/QuBIOS)** | ← This repo: measured QEC & entanglement toolkit |
| **[QubitPage-OS](https://github.com/qubitpage/QubitPage-OS)** | Browser-based quantum & AI learning platform |

---

## Quick start

```bash
pip install -r requirements.txt
python tests/test_memory.py          # 8 physics tests (≈ 10 s)
python examples/quickstart.py        # measured numbers (≈ 40 s)
python benchmarks/steane_memory.py   # full simulation sweep (≈ 15 min)
```

```python
from qubios import run_memory

r = run_memory(p=1e-3, rounds=3, shots=100_000, flags=True)
print(r.logical_error_rate, "vs unprotected", r.physical_error_rate)
```

---

## Results: simulation (circuit-level noise)

Every number below comes from sampled, decoded circuits. Noise model: depolarizing probability **p** after every gate, on every idle qubit in every time step, and on every reset and measurement. Logical |0_L⟩ memory; the X basis gives the same numbers within error bars. **Ratio < 1 means error correction helps.**

| Circuit | p | Rounds | Steane logical error | Unprotected qubit | Ratio |
|---|---|---|---|---|---|
| basic | 0.01% | 3 | 4.00e-04 | 3.44e-03 | **0.12×** |
| basic | 0.1% | 3 | 1.40e-02 | 3.14e-02 | **0.45×** |
| basic | 0.3% | 3 | 8.27e-02 | 9.15e-02 | **0.90×** |
| basic | 0.3% | 10 | 2.35e-01 | 2.35e-01 | **1.00×** |
| flagged | 0.01% | 1 | 4.95e-05 | 1.61e-03 | **0.03×** |
| flagged | 0.01% | 3 | 2.07e-04 | 4.47e-03 | **0.05×** |
| flagged | 0.01% | 10 | 6.00e-04 | 1.38e-02 | **0.04×** |
| flagged | 0.1% | 3 | 1.93e-02 | 4.11e-02 | **0.47×** |
| flagged | 0.3% | 3 | 1.24e-01 | 1.15e-01 | **1.08×** |

Full table (28 points): [benchmarks/results/steane_memory.json](benchmarks/results/steane_memory.json)

**What this shows**
- At low noise, the flagged Steane memory reduces errors by **20–30×**, and its logical error scales as **p²**, as a distance-3 code should.
- Around **p ≈ 0.3%** (similar to today's two-qubit gate errors) the gain disappears. That's the break-even region.
- Flags help most at low p. At high p, the extra gates they add cost more than they save.
- **Caveat:** in this model idle qubits are as noisy as gates. On real hardware idling is usually quieter, which makes the unprotected qubit look better. The hardware test below settles this on a real chip.

---

## Results: real IBM Quantum hardware (7 Nov 2026)

About 4½ minutes of QPU time (265 s) on **ibm_fez** and **ibm_marrakesh** (IBM Heron r2, 156 qubits), ~30 jobs. Every job ID is listed in [hardware/results/ibm_hardware_20261107_all_runs.json](hardware/results/ibm_hardware_20261107_all_runs.json) and can be checked in IBM Quantum Platform → Workloads.

### Verdict
| Question | Answer on real hardware |
|---|---|
| Does Steane error **correction** (decoding) make a qubit last longer than a bare qubit? | **No.** Encoded qubits had higher error at every wait time (0–200 µs), on both devices, with and without dynamical decoupling. |
| Does Steane error **detection** (discard shots with a nonzero syndrome) help? | **Partly.** For \|+⟩ states with DD, the error was **2–5× lower than a bare qubit up to ~20 µs**, but 50–70% of shots were discarded. For \|1⟩ it roughly ties. This effect is known in the literature, so it isn't a novelty. |
| Does relaying a state through "escort" qubits (QuBIOS transit-ring idea) keep it alive longer? | **No: it makes it worse.** Same 100 µs total: resting \|1⟩ 36.8% error, relayed through 2/5/10 escorts 60.7/60.9/64.5% (no DD); with DD 19.9% vs 35.7–41.5%. Each SWAP adds errors and moves them along; it can't clean the state. |
| Do "opposed" qubit pairs (\|01⟩+\|10⟩, decoherence-free subspace) shield a superposition? | **No, not on IBM.** Opposed pairs behaved exactly like aligned pairs (\|00⟩+\|11⟩) and worse than a single qubit, e.g. at 10 µs with XY4: 17.3% vs 17.6% vs 8.5%. That means IBM's noise is mostly independent per qubit, not shared, so there's nothing to cancel. |
| Does the result depend on the device? | **Yes.** On ibm_fez, error-detected \|1_L⟩ beat bare \|1⟩ 2–6× (5–40 µs, 19–40% of shots kept); on ibm_marrakesh it only tied. |
| Does dynamical decoupling help? | **Yes**, a lot for bare qubits (e.g. \|1⟩ at 100 µs: 38% → 20%). |

### Key data: fixed physical layout, ibm_marrakesh, 8,000 shots, means over 4 copies
| Wait | Bare \|1⟩ | Encoded \|1_L⟩ decoded | Encoded \|1_L⟩ error-detected (kept) | Bare \|+⟩ (XY4) | Encoded \|+_L⟩ error-detected, XY4 (kept) |
|---|---|---|---|---|---|
| 0 µs | 1.1% | 14.8% | 2.1% (62%) | 2.6% | **0.6%** (59%) |
| 5 µs | 3.8% | 19.6% | 3.9% (45%) | 4.7% | **1.1%** (47%) |
| 10 µs | 6.4% | 23.4% | 6.1% (32%) | 7.0% | **2.4%** (40%) |
| 20 µs | 10.8% | 29.6% | 11.7% (21%) | 11.4% | **6.5%** (29%) |
| 40 µs | 19.1% | 39.6% | 27.8% (17%) | 20.5% | 27.3% (18%) |

A repeat run agreed within ~1–2% (jobs `dbn6enoou4as73b3utc0` and `dbn6eo5t7fqc73aj0te0`).

### Lessons and pitfalls (self-critique)
- **IBM Composer silently removes `delay` and `barrier` instructions.** It also cancels H·H pairs and can delete SWAP chains. Timing experiments submitted through Composer are therefore invalid (we caught this and discarded/cancelled those jobs). Use Qiskit with explicit barriers and check the compiled circuit, as in `hardware/colab/qubios_hw_experiments.py`.
- **The layout confound.** When each wait time was compiled separately, the encoded qubit *appeared* to beat bare qubits (round 2). With a fixed physical layout (round 3) the advantage disappeared. Always fix the layout when comparing.
- **Routing overhead.** The 7-qubit encoder needs ~30 two-qubit gates after mapping to the heavy-hex lattice (ideal 9). That routing cost is the main reason encoding loses.

### Reproduce
`hardware/colab/qubios_hw_experiments.py` (works in Google Colab; the API key is read with `getpass`).

---

## v1.2.2: night of 7 Nov 2026, rounds 5–19 on ibm_marrakesh, ibm_kingston and ibm_fez
All job IDs and raw numbers are in `hardware/results/ibm_hardware_20261107_all_runs.json`; code is in `hardware/colab/exp2.py … exp10.py`. These are short repetition (bit-flip or phase-flip) codes on 1-D qubit chains. Each protects **one** kind of information (0/1 *or* +/−), not a full logical qubit.

### What worked (real hardware, reproduced or controlled)
| Experiment | Code | Bare qubit (same circuit) |
|---|---|---|
| \|1⟩ held 4 rounds (~50 µs), real-time feedback, n=5 | 15.7% | 26.1% |
| \|+⟩ held ~80 µs, phase code n=5 + hand-written DD, qubits picked by T1/T2 | **3.2%** | 51% |
| \|1⟩ held ~150 µs, n=9 bit code, space-time matching decoder (PyMatching) | **3.0%** | 41–75% |

- **The decoder matters:** on the same 7-qubit data, majority vote failed 55% of the time, matching 7.6%.
- **Placement matters:** coherence-scored qubits (T1, T2, readout) halved the error vs readout-only placement (3.2% vs 7.1%).
- **Software correction beat real-time feedback** for the phase code (5.7% vs 11.4%).
- **Noise scaling of big superpositions:** a GHZ state of N qubits (prepared, held, then undone so coherence is read on one qubit) loses coherence at a rate ∝ N^0.7–1.0: N^1.03 on ibm_marrakesh, N^0.71 on ibm_fez, N^0.91 on coherence-scored qubits. The noise acts independently per qubit, with no "superdecoherence". A 16× outlier at N=12 disappeared on better qubits (9.3×). Without DD, the GHZ phase rotates N times faster: a built-in amplifier of frequency drift, which is the principle behind quantum sensing.
- **No-cloning confirmed:** CNOT fan-out of \|+⟩ onto 3 qubits gives each qubit a 50.6% "+" outcome (pure chance) but 93.5% XXX parity. The result is one entangled (GHZ) state, not three copies.

### Both bases at once (second allocation, rounds 17–19)
- **[[4,2,2]] logical Bell pair** (error **detection** in both bases): it beat the **best** physical Bell pair on **all three** machines. For example, ibm_kingston at 10 µs: **97.7%** vs 89.7% fidelity bound, keeping 85% of shots.
- **Majority vote over 1 → 3 → 5 Bell pairs** (error **correction**, all shots kept): on ibm_kingston at 10 µs the bound rose from 79.7% to 94.1% to **98.1%**.

Full tables, methods and every job ID: [docs/HARDWARE_REPORT_2026-11-07.md](docs/HARDWARE_REPORT_2026-11-07.md).

### What did not work (kept for honesty)
- **"Conveyor" (hand the state to a fresh qubit every 10 µs).** It won on ibm_marrakesh for \|+⟩, but not for \|+i⟩, and not on ibm_kingston. The effect averages away bad qubits; it doesn't beat good ones. Related prior work: Averin et al., PRL 116, 010501 (2016); Chang, Su & Puri, arXiv:2508.04947.
- **Echo pulses during watcher readout:** no gain.
- **Closed-loop re-placement from the learned noise map:** the noise map correctly flagged noisy qubits, but the available replacement qubits were worse.
- **Correlation-learned decoder (p_ij method):** inconsistent with 2,000–8,000 shots.
- **Raw GHZ fidelity** (no readout mitigation): 42% at N=10, 4% at N=80. No size was certified above 50%.

### Platform pitfalls found
- IBM refuses automatic DD in circuits with mid-circuit measurements; write DD pulses yourself.
- `switch_case` is rejected by the real backend; real-time lookup tables above ~15 branches per round hit controller limits (error 6057).
- Sabre layout with many `if_else` blocks can hang for 8+ minutes; hand-pick chains and use `routing_method="none"`.
- Without a barrier, the compiler cancels prepare/unprepare pairs (our "τ = 0" points in round 13 were invalid).

---

## Project status

**v1.2.2:** hardware rounds 5–19: repetition and phase codes, software matching decoder, coherence-aware placement, GHZ noise scaling, the [[4,2,2]] logical Bell pair, Bell-pair majority voting, and the conveyor test and its refutation. Full report in [docs/HARDWARE_REPORT_2026-11-07.md](docs/HARDWARE_REPORT_2026-11-07.md).

**v1.2.1:** added the opposed-pair (decoherence-free subspace) test and the ibm_fez fixed-layout replication.

**v1.2.0:** ~25 real-hardware jobs on ibm_fez and ibm_marrakesh: memory vs time, error detection, dynamical decoupling and the escort-relay test (see above).

**v1.1.1:** first real ibm_fez results.

**Fixed in v1.1.0** (all covered by [tests/test_memory.py](tests/test_memory.py)):
1. The Steane encoder now produces a real codeword (all 6 stabilizers +1; v1.0.0 gave 0).
2. `SteaneQEC.run_qec_cycle` now **measures** the logical error rate (v1.0.0 returned the formula 21·p²).
3. Teleportation fidelity is now measured in the Z/X/Y bases after feed-forward correction (v1.0.0 measured outcome uniformity, which can't detect errors).
4. Bell-pair fidelity is now F = (1 + ⟨XX⟩ − ⟨YY⟩ + ⟨ZZ⟩)/4 (v1.0.0 used ZZ only).
5. v1.0.0's hardware claims were withdrawn: those numbers were local simulations.

**Still experimental:** `QubiLogicEngine.benchmark_with_buffer` is a **model** (it assumes p_L = 21·p²). It now says so in its output and raises a warning. `EscortQubVirt`, `TransitRing` and `VirtualQubitManager` are design ideas that haven't been shown to give any benefit.

**Roadmap**
- [x] First real-hardware results with job IDs (ibm_fez, above)
- [x] Repeat the hardware runs to measure job-to-job and layout variation (v1.2.2)
- [x] Hardware-aware layout: coherence-scored qubit chains (v1.2.2)
- [ ] Noise model calibrated from backend data (T1/T2, gate and readout errors) instead of a uniform p
- [x] X-basis (phase-code) hardware runs (v1.2.2)
- [x] Both-basis protection on hardware by detection ([[4,2,2]]) (v1.2.2)
- [ ] Distance-3 code that **corrects** in both bases (surface or Bacon–Shor patch)

---

## Background reading
- A. Steane, *Multiple particle interference and quantum error correction*, Proc. R. Soc. A 452 (1996)
- R. Chao & B. Reichardt, *Quantum error correction with only two extra qubits*, PRL 121, 050502 (2018)
- C. Gidney, *Stim: a fast stabilizer circuit simulator*, Quantum 5, 497 (2021)
- C. Bennett et al., *Purification of noisy entanglement and faithful teleportation via noisy channels*, PRL 76, 722 (1996)

---

## License
MIT. See [LICENSE](LICENSE).

<div align="center">
<strong><a href="https://qubitpage.com">🌐 qubitpage.com</a></strong> · © 2026 Qubitpage® · contact@qubitpage.com<br/>
<sub>QuBIOS v1.2.2</sub>
</div>
