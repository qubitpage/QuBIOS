# QuBIOS hardware round 11 — Qubitpage.  Needs exp2..exp7 functions in the kernel.
# (A) READOUT ECHO: data qubits get an X-X echo WHILE the watcher qubits are being measured/reset
#     (IBM forbids automatic DD in dynamic circuits, and our DD so far covered only the waits)
# (B) CLOSED LOOP "the decoder reads the noise": run a code, learn every physical error rate from the
#     syndrome correlations, map them back onto physical qubits, then RE-PLACE the code away from the
#     noisiest qubits and run again.  Same code, same circuit, only placement driven by the learned noise.
import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister


def mem_circuit2(R, copies, basis="X", dd=True, readout_echo=False, wait=10.0):
    sizes = [2 * n - 1 for n in copies]
    nsyn = sum(n - 1 for n in copies)
    q = QuantumRegister(sum(sizes), "q")
    syn = ClassicalRegister(max(1, R * nsyn), "syn"); fin = ClassicalRegister(sum(copies), "fin")
    qc = QuantumCircuit(q, syn, fin, name=f"mem{basis}{'_dd' if dd else ''}{'_rde' if readout_echo else ''}_R{R}")
    off = np.cumsum([0] + sizes)
    data = [[int(off[k]) + 2 * i for i in range(n)] for k, n in enumerate(copies)]
    anc = [[int(off[k]) + 2 * i + 1 for i in range(n - 1)] for k, n in enumerate(copies)]
    alld = [d for D in data for d in D]
    for d in alld: (qc.x if basis == "Z" else qc.h)(d)
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
        for k, n in enumerate(copies):
            for j, a in enumerate(anc[k]):
                qc.measure(a, syn[b]); b += 1
            for a in anc[k]: qc.reset(a)
        if readout_echo:                      # echo on every data qubit, scheduled in parallel with readout
            for d in alld:
                qc.delay(0.3, d, unit="us"); qc.x(d); qc.delay(0.6, d, unit="us"); qc.x(d); qc.delay(0.3, d, unit="us")
    qc.barrier()
    f = 0
    for k, n in enumerate(copies):
        for d in data[k]:
            if basis == "X": qc.h(d)
            qc.measure(d, fin[f]); f += 1
    return qc


def noise_map(det, n, R):
    """Learned per-data-qubit error probability (avg over time layers) and per-watcher measurement error."""
    X = det.astype(float); m = X.mean(0); XX = (X.T @ X) / len(X); J = n - 1
    node = lambda r, j: r * J + j
    def pij(a, b):
        num = 4 * (XX[a, b] - m[a] * m[b]); den = 1 - 2 * m[a] - 2 * m[b] + 4 * XX[a, b]
        v = 1 - num / den if den > 1e-9 else 1.0
        return float(np.clip(0.5 - 0.5 * np.sqrt(max(v, 0.0)), 0, 0.5))
    data = np.zeros(n); meas = np.zeros(J); cnt = np.zeros(n)
    for r in range(R + 1):
        for i in range(1, n - 1):
            data[i] += pij(node(r, i - 1), node(r, i)); cnt[i] += 1
        if r < R:
            for j in range(J): meas[j] += pij(node(r, j), node(r + 1, j)) / R
    # end qubits: boundary edges = detection rate not explained by the bulk
    for r in range(R + 1):
        for i, j in ((0, 0), (n - 1, J - 1)):
            a = node(r, j)
            nb = [node(r, j - 1)] if j > 0 else []
            nb += [node(r, j + 1)] if j < J - 1 else []
            nb += [node(r - 1, j)] if r > 0 else []
            nb += [node(r + 1, j)] if r < R else []
            s = sum(pij(a, b) for b in nb)
            data[i] += float(np.clip((m[a] - s) / max(1 - 2 * s, 1e-3), 0, 0.5)); cnt[i] += 1
    return data / np.maximum(cnt, 1), meas


def pick_chains_scored_ex(bm, lengths, exclude=(), **kw):
    """pick_chains_scored with some physical qubits forbidden (sets their score to +inf)."""
    global qubit_scores
    orig = qubit_scores
    def patched(bm_, wait_us=10.0):
        adj, e2q, sc = orig(bm_, wait_us)
        for q in exclude:
            if q in sc: sc[q] = 1e9
        adj = {q: (v - set(exclude)) for q, v in adj.items() if q not in exclude}
        sc = {q: s for q, s in sc.items() if q not in exclude}
        return adj, e2q, sc
    qubit_scores = patched
    try:
        return pick_chains_scored(bm, lengths, **kw)
    finally:
        qubit_scores = orig
