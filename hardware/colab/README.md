# Colab hardware scripts (v1.2.2)
Run in Google Colab with `qiskit`, `qiskit-ibm-runtime`, `pymatching`; paste your IBM key into the `getpass` prompt (never into a file).
Each script defines functions; later rounds reuse earlier ones, so `exec` them in order: exp2 → exp10.

| File | Round | What it tests |
|---|---|---|
| exp2.py | 5 | chain picker, cloning test, repetition code with real-time feedback |
| exp3.py | 6 | conveyor (one-bit teleport hops), phase code + manual DD |
| exp4.py | 7 | syndromes recorded, space-time matching decoder |
| exp5.py | 8 | same-qubit conveyor control, per-copy decoding |
| exp6.py | 9 | conveyor confirmation: XY4, two states, two devices |
| exp7.py | 10 | coherence-scored placement, park diagnosis, learned decoder |
| exp8.py | 11 | readout echo, closed loop noise map → re-placement |
| exp9.py | 12 | GHZ up to 80 qubits |
| exp10.py | 13–14 | GHZ coherence decay vs N (noise correlation test) |
| exp11.py | 17 | [[4,2,2]] logical Bell pair (detection in both bases) |
| exp12.py | 18 | three Bell pairs, majority vote |
| exp13.py | 19 | 1/3/5 Bell pairs on the same shots |
