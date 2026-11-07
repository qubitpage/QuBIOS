# QuBIOS hardware round 12 — Qubitpage.  Big superpositions: N-qubit GHZ (|00..0> + |11..1>)/sqrt2 grown from the middle
# of a hand-picked qubit path, measured in Z (populations) and X (N-body parity = coherence).  Fidelity estimate
# F ~ (P_all0 + P_all1)/2 + |<X^N>|/2 ; F > 0.5 certifies genuine N-qubit entanglement (lower-bound style estimate).
import numpy as np
from qiskit import QuantumCircuit

def ghz_circuit(N, basis):
    qc = QuantumCircuit(N, N, name=f"ghz{N}_{basis}")
    m = N // 2; qc.h(m)
    for k in range(1, max(m, N - 1 - m) + 1):
        if m - k >= 0: qc.cx(m - k + 1, m - k)
        if m + k <= N - 1: qc.cx(m + k - 1, m + k)
    qc.barrier()
    if basis == "X": qc.h(range(N))
    qc.measure(range(N), range(N))
    return qc

def ghz_stats(bz, bx):
    N = bz.shape[1]; w = bz.sum(1)
    p0, p1 = float((w == 0).mean()), float((w == N).mean())
    par = float(np.mean(1 - 2 * (bx.sum(1) % 2)))
    return {"P_all0": round(p0, 4), "P_all1": round(p1, 4), "parity_X": round(par, 4), "F_est": round((p0 + p1) / 2 + abs(par) / 2, 4)}

def long_path(bm, N, tries=((0.03, 0.01), (0.05, 0.015), (0.08, 0.025), (0.15, 0.05))):
    for ro, e2 in tries:
        ch = pick_chains(bm, [N], ro, e2, False)
        if ch: return ch[0], (ro, e2)
    return None, None
