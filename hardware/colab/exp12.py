# QuBIOS round 18 — Qubitpage.  From DETECTION to CORRECTION without discarding shots:
# three Bell pairs on a 6-qubit path, stabilizers Z-parity(pair i) x Z-parity(pair j) and the same for X.
# Logical <ZZ>, <XX> = MAJORITY of the three pair parities (corrects one bad pair per basis, keeps 100% of shots).
import numpy as np
from qiskit import QuantumCircuit

def bell3_circuit(ncopy, tau, basis):
    n = 6 * ncopy
    qc = QuantumCircuit(n, n, name=f"b3_{basis}_t{tau}")
    for k in range(ncopy):
        for p in (6 * k, 6 * k + 2, 6 * k + 4):
            qc.h(p); qc.cx(p, p + 1)
    qc.barrier(); qc.delay(tau, range(n), unit="us"); qc.barrier()
    if basis == "X": qc.h(range(n))
    qc.measure(range(n), range(n))
    return qc

def bell3_stats(bz, bx, ncopy):
    r = {"bare": [], "detect2": [], "keep2": [], "majority3": []}
    for k in range(ncopy):
        a = 6 * k
        P = lambda b, i: b[:, a + 2 * i] ^ b[:, a + 2 * i + 1]
        ev = lambda v: float(np.mean(1 - 2 * v))
        zs = [P(bz, i) for i in range(3)]; xs = [P(bx, i) for i in range(3)]
        r["bare"].append((ev(zs[0]) + ev(xs[0])) / 2)
        kz, kx = ~(zs[0] ^ zs[1]), ~(xs[0] ^ xs[1])
        r["detect2"].append((ev(zs[0][kz]) + ev(xs[0][kx])) / 2); r["keep2"].append(float((kz.mean() + kx.mean()) / 2))
        mz = (zs[0].astype(int) + zs[1] + zs[2]) >= 2; mx = (xs[0].astype(int) + xs[1] + xs[2]) >= 2
        r["majority3"].append((ev(mz) + ev(mx)) / 2)
    return {k: round(float(np.mean(v)), 4) for k, v in r.items()}
