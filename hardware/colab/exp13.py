# QuBIOS round 19 — Qubitpage.  Does more redundancy keep helping?  K=5 Bell pairs on a 10-qubit path;
# analyse the SAME shots as 1, 3 and 5 pairs (majority vote in each basis, keeps 100% of shots).
import numpy as np
from qiskit import QuantumCircuit

def bellK_circuit(K, ncopy, tau, basis):
    n = 2 * K * ncopy
    qc = QuantumCircuit(n, n, name=f"b{K}_{basis}_t{tau}")
    for p in range(0, n, 2):
        qc.h(p); qc.cx(p, p + 1)
    qc.barrier(); qc.delay(tau, range(n), unit="us"); qc.barrier()
    if basis == "X": qc.h(range(n))
    qc.measure(range(n), range(n))
    return qc

def bellK_stats(bz, bx, K, ncopy):
    res = {m: [] for m in (1, 3, 5) if m <= K}
    ev = lambda v: float(np.mean(1 - 2 * v))
    for k in range(ncopy):
        a = 2 * K * k
        zs = np.stack([bz[:, a + 2 * i] ^ bz[:, a + 2 * i + 1] for i in range(K)], 1).astype(int)
        xs = np.stack([bx[:, a + 2 * i] ^ bx[:, a + 2 * i + 1] for i in range(K)], 1).astype(int)
        for m in res:
            sel = list(range(K // 2 - m // 2, K // 2 - m // 2 + m))      # centre pairs
            res[m].append((ev(zs[:, sel].sum(1) * 2 > m) + ev(xs[:, sel].sum(1) * 2 > m)) / 2)
    return {f"{m}_pairs": round(float(np.mean(v)), 4) for m, v in res.items()}
