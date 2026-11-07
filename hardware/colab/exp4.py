# QuBIOS hardware round 7 — Qubitpage. Needs exp2 (pick_chains_auto) + exp3 (_wait, conveyor_circuit, analyse_conveyor).
# Fixes from the round-6 self-critique:
#  * no real-time branches (they cost latency and hit the controller limit) -> record every syndrome, decode in software
#  * space-time minimum-weight matching decoder (PyMatching) over ALL rounds instead of a final majority vote
#  * fair baselines: bare qubits get the SAME manual dynamical decoupling as the code
import numpy as np
import pymatching
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister


def mem_circuit(R, copies, basis="Z", dd=False, wait=10.0):
    """basis Z: bit-flip code holding |1..1>; basis X: phase-flip code holding |+..+>. copies: list of n (1 = bare)."""
    sizes = [2 * n - 1 for n in copies]
    nsyn = sum(n - 1 for n in copies)
    q = QuantumRegister(sum(sizes), "q")
    syn = ClassicalRegister(max(1, R * nsyn), "syn")
    fin = ClassicalRegister(sum(copies), "fin")
    qc = QuantumCircuit(q, syn, fin, name=f"mem{basis}{'_dd' if dd else ''}_R{R}")
    off = np.cumsum([0] + sizes)
    data = [[int(off[k]) + 2 * i for i in range(n)] for k, n in enumerate(copies)]
    anc = [[int(off[k]) + 2 * i + 1 for i in range(n - 1)] for k, n in enumerate(copies)]
    alld = [d for D in data for d in D]
    for d in alld:
        (qc.x if basis == "Z" else qc.h)(d)
    for r in range(R):
        qc.barrier(); _wait(qc, alld, wait, dd); qc.barrier()
        b = r * nsyn
        for k, n in enumerate(copies):
            if n == 1: continue
            if basis == "X":
                for d in data[k]: qc.h(d)
            for j, a in enumerate(anc[k]):
                qc.cx(data[k][j], a); qc.cx(data[k][j + 1], a)
            if basis == "X":
                for d in data[k]: qc.h(d)
            for j, a in enumerate(anc[k]):
                qc.measure(a, syn[b]); b += 1
            for a in anc[k]: qc.reset(a)
    qc.barrier()
    f = 0
    for k, n in enumerate(copies):
        for d in data[k]:
            if basis == "X": qc.h(d)
            qc.measure(d, fin[f]); f += 1
    return qc


_M = {}
def _matcher(n, R):
    """Space-time matching graph of an n-qubit repetition code with R noisy syndrome rounds + perfect final readout."""
    if (n, R) in _M: return _M[(n, R)]
    m = pymatching.Matching(); J = n - 1
    node = lambda r, j: r * J + j
    for r in range(R + 1):
        for i in range(n):                         # data error before layer r
            fid = {0} if i == 0 else set()
            if i == 0:   m.add_boundary_edge(node(r, 0), fault_ids=fid, weight=1.0)
            elif i == n - 1: m.add_boundary_edge(node(r, J - 1), weight=1.0)
            else:        m.add_edge(node(r, i - 1), node(r, i), fault_ids=fid, weight=1.0)
        if r < R:
            for j in range(J):                     # measurement error
                m.add_edge(node(r, j), node(r + 1, j), weight=1.0)
    _M[(n, R)] = m
    return m


def decode_mem(syn, fin, copies, R, basis):
    """Returns {n: (majority_error, matching_error)} averaged over copies of that size."""
    exp = 1 if basis == "Z" else 0
    nsyn = sum(n - 1 for n in copies)
    out, f, s0 = {}, 0, 0
    for n in copies:
        fb = fin[:, f:f + n].astype(np.uint8); f += n
        maj = ((fb.sum(1) * 2 > n).astype(np.uint8) != exp).mean()
        if n == 1:
            out.setdefault(n, []).append((maj, maj)); continue
        J = n - 1
        S = np.stack([syn[:, r * nsyn + s0: r * nsyn + s0 + J] for r in range(R)], 1).astype(np.uint8)  # [shots,R,J]
        s0 += J
        final = (fb[:, :-1] ^ fb[:, 1:])                           # data parities at the end
        layers = np.concatenate([S, final[:, None, :]], 1)         # [shots,R+1,J]
        prev = np.concatenate([np.zeros_like(layers[:, :1]), layers[:, :-1]], 1)
        det = (layers ^ prev).reshape(len(fb), -1)
        pred = _matcher(n, R).decode_batch(det)[:, 0]
        mw = ((fb[:, 0] ^ pred) != exp).mean()
        out.setdefault(n, []).append((maj, mw))
    return {n: (round(float(np.mean([a for a, _ in v])), 4), round(float(np.mean([b for _, b in v])), 4)) for n, v in out.items()}
