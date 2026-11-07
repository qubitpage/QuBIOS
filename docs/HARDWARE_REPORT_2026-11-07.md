# QuBIOS hardware report: 7 November 2026

**Devices:** IBM Heron r2 `ibm_marrakesh`, `ibm_fez`, `ibm_kingston` (156 qubits, heavy-hex lattice).
**Access:** IBM Quantum Open Plan, instance "Qubit OS". About 584 s of the first 600 s allocation, plus about 100 s of a second allocation.
**Code:** `hardware/colab/exp2.py … exp13.py`, run in Google Colab with Qiskit 2.x, qiskit-ibm-runtime and PyMatching. The API key was entered with `getpass` and never stored.
**Raw numbers and every job ID:** `hardware/results/ibm_hardware_20261107_all_runs.json`.

Each job ID below can be opened by the account owner on the IBM Quantum Platform → Workloads page. IBM keeps the raw measurement data of every job, so anyone with access can re-download the bits and re-run the analysis functions in `hardware/colab/`.

---

## 1. Summary

| # | Question | Answer from hardware | Status |
|---|---|---|---|
| 1 | Does CNOT "cloning" copy a superposition? | No. Each "clone" reads "+" 50.6% of the time (chance), a real \|+⟩ 98.7%. The 3 qubits form one GHZ state (XXX parity 93.5%). | Confirmed (expected physics) |
| 2 | Can a repetition code with escort (ancilla) qubits beat a bare qubit? | Yes, for one basis. \|1⟩ after ~150 µs: 9-qubit code 3.0% error vs 41–75% bare. \|+⟩ after ~80 µs: 5-qubit phase code + DD 3.2% vs 51% bare. | Confirmed, several rounds |
| 3 | Real-time feedback vs correcting afterwards in software? | Software decoding (space-time matching) won; real-time branches added latency and hit controller limits. | Confirmed |
| 4 | Does choosing qubits by T1/T2 help? | Yes, it roughly halved errors (3.2% vs 7.1%). | Confirmed |
| 5 | Does handing the state to fresh qubits ("conveyor") beat staying put? | No. It won for one state on one device, but not for \|+i⟩ and not on ibm_kingston. | Refuted |
| 6 | Do big superpositions die faster than proportionally (correlated noise)? | No. GHZ decay rate ∝ N^0.7–1.0 on two devices, so the noise is independent per qubit. | Confirmed, 2 devices |
| 7 | Can we protect a full quantum state (both bases)? | Yes, by detection. A [[4,2,2]] logical Bell pair beats the best physical pair on all 3 devices up to 20–40 µs. | Confirmed, 3 devices |
| 8 | Does more redundancy keep helping, without discarding shots? | Yes. 1 → 3 → 5 Bell pairs with majority vote: 79.7% → 94.1% → 98.1% at 10 µs (kingston). | Confirmed, 2 devices |

**Novelty assessment (self-critical).** All positive results use known techniques: repetition codes, matching decoders, DD, [[4,2,2]] detection, and repetition of Bell pairs. They are measured and reproducible engineering results, not new physics. The two ideas that were genuinely new to this project were tested and failed or stayed inconclusive:
- **"Conveyor" refresh:** refuted. Closest prior work: Averin et al., PRL 116, 010501 (2016).
- **Noise-map-driven re-placement:** no gain when the flagged qubits had no better spares.

---

## 2. Methods common to all rounds
- **Layout:** chains of physical qubits chosen by our own picker (readout error, 2-qubit error, and from round 10 also T1/T2). Transpiled with `routing_method="none"`, so the physical qubits are exactly the ones reported.
- **Baselines:** bare qubits sit **in the same circuit** as the code, on qubits of comparable quality, for the same wall-clock time.
- **Dynamical decoupling:**
  - Static circuits use IBM's XY4 option.
  - Dynamic circuits (with mid-circuit measurement) can't use IBM's automatic DD, so DD pulses are written into the circuit: X–X echo or XY4 inside every wait.
- **Decoding:** space-time minimum-weight matching (PyMatching) on detection events, with uniform weights. A correlation-learned (p_ij) variant was also tested.
- **Fidelity bound for Bell pairs:** F(Φ⁺) ≥ (⟨XX⟩ + ⟨ZZ⟩)/2. This holds for any state because (I−XX)(I−ZZ) ≥ 0.

---

## 3. Results by round
Errors are the fraction of shots with the wrong logical value. Waits are 10 µs per round unless stated.

### Round 5: real-time feedback repetition code (marrakesh, jobs `dbn78ms4148s7390eeh0`, `dbn78n5t7fqc73aj23eg`)
| Rounds | n=3 off | n=3 feedback | n=5 off | n=5 feedback | bare |
|---|---|---|---|---|---|
| 1 | 5.8% | 4.7% | 1.8% | 3.4% | 9.8% |
| 2 | 15.1% | 9.3% | 6.5% | 8.0% | 16.1% |
| 4 | 31.9% | 18.8% | 22.6% | 15.7% | 26.1% |

### Round 6: phase code (\|+⟩) and conveyor (marrakesh, `dbn7ed8ou4as73b409kg`, `dbn7eg8qtr0c73acg2ug`)
- **Phase code, n=5 with DD, after 4 rounds:** software decoding 5.7%, real-time feedback 11.4%, bare 52%.
- **Bare \|+⟩ without DD:** error above 50% (65%), which means a coherent phase rotation rather than random dephasing.

### Round 7: software matching decoder (`dbn7g1s4148s7390en00` and the others in the JSON)
| Code | R=2 | R=4 | R=8 |
|---|---|---|---|
| phase n=5 + DD | 2.4% | 5.2% | 9.0% |
| bare \|+⟩ + DD | 31.7% | 53% | 57% |
| bit n=7, matching | 1.1% | 4.2% | 7.6% |
| bit n=7, majority vote | 2.3% | 15.3% | 54.9% |
| bare \|1⟩ | 14.9% | 26.8% | 45.6% |

### Rounds 8–9: conveyor confirmation (marrakesh `dbn7hc8ou4as73b40d6g`, `dbn7ipgqtr0c73acg89g`; kingston `dbn7it44148s7390eqo0`)
| Device | State | Hops | Conveyor | Stay-put (same qubits, XY4) |
|---|---|---|---|---|
| marrakesh | \|+⟩ | 4 | 24.5% | 29.3% |
| marrakesh | \|+i⟩ | 8 | 44.8% | 43.1% |
| kingston | \|+⟩ | 8 | 32.5% | 32.0% |
| kingston | \|+i⟩ | 8 | 37.1% | 31.1% |

**Verdict:** not a general improvement. The best single stay-put qubit (13%) beat every conveyor chain.

### Round 8: large bit codes, per copy (`dbn7hfgou4as73b40d9g`)
| Wait | n=9 | n=7 | n=3 | bare |
|---|---|---|---|---|
| R=8 | 7.7%, 2.9% | 20.4%, 13.3% | 24.3%, 20.6% | 37.6%, 46.2% |
| R=12 | 12.2%, 4.2% | 18.5%, 15.8% | 28.3%, 25.4% | 51.9%, 56.7% |

### Round 10: placement, park diagnosis, learned decoder (`dbn7nndt7fqc73aj2obg`, `dbn7nq8ou4as73b40ou0`, `dbn7nv0ou4as73b40pc0`)
- **Phase code n=5 + DD at R=8:** 3.2% with coherence-scored qubits vs 7.1% with readout-only qubits.
- **Parked \|+⟩ after 2 steps:** 31.7% without DD, 10.3% with XY4, 10.1% with XY8, 13.0% with XY4 plus a neighbour measured each step. Neighbour measurement crosstalk costs 3–6 points.

### Round 11: readout echo and closed loop (`dbn7rldt7fqc73aj30h0`, `dbn7rodt7fqc73aj30og`, `dbn7rr0qtr0c73acgre0`)
- **Readout echo:** no gain.
- **Closed loop:** the learned noise map flagged qubits 65, 132, 107 and 62. Re-placing all chains away from them made things worse (4.0% → 9.4%), because the replacement qubits were worse.

### Round 12: corrected loop and GHZ up to 80 qubits (`dbn7udlt7fqc73aj36ag`, `dbn7uhk4148s7390fjq0`, `dbn7uult7fqc73aj37fg`)
Raw GHZ fidelity estimate (no readout mitigation): N=10 42%, N=20 26%, N=40 11%, N=80 4%. No size was certified above 50%.

### Rounds 13–16: how fast does an N-qubit superposition fade? (marrakesh `dbn7vcgou4as73b419dg`, `dbn80a44148s7390fnm0`, `dbn82pgou4as73b41h6g`, `dbn83elt7fqc73aj3hl0`; fez `dbn80lk4148s7390foig`)
**Method:** GHZ_N is prepared, held for τ, then undone, so the coherence is read on a single qubit (immune to N-qubit readout error).

| N | marrakesh decay /µs | ×N=1 | fez decay /µs | ×N=1 |
|---|---|---|---|---|
| 1 | 0.0111 | 1 | 0.0192 | 1 |
| 2 | 0.0223 | 2.0 | 0.048 | 2.5 |
| 4 | 0.0328 | 3.0 | 0.0727 | 3.8 |
| 8 | 0.0678 | 6.1 | 0.1047 | 5.5 |
| 12 | 0.0917 (scored path) | 9.3 | 0.1218 | 6.3 |

- **Fitted exponent:** 1.03 (marrakesh), 0.71 (fez), 0.91 (marrakesh on coherence-scored qubits). The noise is independent; there is no superdecoherence.
- **Without DD,** the GHZ coherence turned negative: a coherent phase rotation N times faster than for one qubit (the quantum-sensing effect).
- **Round 13 note:** its "τ = 0" points were invalid. Without a barrier, the compiler cancelled the prepare/unprepare pair; this was fixed in round 14.

### Round 17: [[4,2,2]] logical Bell pair, protected in both bases (marrakesh `dbn859k4148s7390g18g`, fez `dbn85glt7fqc73aj3ktg`, kingston `dbn8780qtr0c73achiug`)
Fidelity lower bound. Format: bare pair (mean) / best physical pair / **logical pair** (fraction of shots kept).

| Wait | marrakesh | fez | kingston |
|---|---|---|---|
| 0.1 µs | 93.5 / 96.7 / **99.6** (93%) | 91.1 / 96.3 / **99.6** (92%) | 93.1 / 96.0 / **99.5** (93%) |
| 10 µs | 72.5 / 86.2 / **93.6** (77%) | 68.4 / 81.6 / **92.0** (73%) | 85.1 / 89.7 / **97.7** (85%) |
| 20 µs | 61.1 / 77.4 / **86.0** (70%) | 47.2 / 62.7 / **75.9** (61%) | 76.0 / 82.5 / **93.4** (79%) |
| 40 µs | 41.8 / 53.6 / **65.9** (60%) | 17.3 / 36.4 / 36.1 (52%) | 55.0 / 65.6 / **76.5** (66%) |

This is error **detection** (shots with a flagged error are discarded). The [[4,2,2]] code on IBM hardware has been published before (Lidar group).

### Rounds 18–19: more Bell pairs, majority vote, all shots kept (marrakesh `dbn877gqtr0c73achit0`, `dbn8900ou4as73b41r8g`; kingston `dbn88d8qtr0c73achkhg`)
| Device, wait | 1 pair | 3 pairs | 5 pairs |
|---|---|---|---|
| kingston 10 µs | 79.7% | 94.1% | **98.1%** |
| kingston 20 µs | 67.6% | 86.3% | **93.0%** |
| marrakesh 10 µs | 77.3% | 89.5% | **94.0%** |
| marrakesh 20 µs | 65.2% | 79.4% | **85.7%** |

The bound is measured basis by basis, on the decoded (corrected) logical pair. The scheme is a repetition code built from Bell pairs, related to entanglement distillation.

---

## 4. Pitfalls we hit (useful to anyone repeating this)
1. **IBM Composer** removes `delay` and `barrier` instructions and cancels H·H pairs, which invalidates timing experiments.
2. **Automatic DD** isn't allowed in dynamic circuits; write the pulses yourself.
3. **`switch_case`** is rejected by the real backend. Real-time lookup tables above ~15 branches per round fail with error 6057.
4. **Sabre layout with many `if_else`** blocks can hang for 8+ minutes; hand-pick qubit chains instead.
5. **Without barriers,** the compiler cancels prepare/unprepare pairs.
6. **Compile each comparison on a fixed physical layout**, or layout differences masquerade as physics.

## 5. Open next steps
- A distance-3 code that **corrects** in both bases (surface or Bacon–Shor patch) with repeated mid-circuit syndrome rounds.
- Readout-error mitigation for GHZ fidelity certification.
- A larger-sample test of the correlation-learned decoder.
