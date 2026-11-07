# QuBIOS round 17 — Qubitpage.  First BOTH-BASIS protection: logical Bell state of the [[4,2,2]] code.
# On a 4-qubit path p0-p1-p2-p3 the [[4,2,2]] logical Bell state equals Bell(p0,p1) x Bell(p2,p3); stabilizers
# XXXX and ZZZZ (products of the two pair parities) DETECT any single error in either basis.
# Logical <ZZ>_L = <Z0Z1>, logical <XX>_L = <X0X1>, kept only when the stabilizer measured in that basis is +1.
# Fidelity lower bound (valid for any state): F(Phi+) >= (<XX> + <ZZ>)/2.   Baseline: the bare Bell pair (p0,p1).
import numpy as np
from qiskit import QuantumCircuit

def bell422_circuit(ncopy, tau, basis):
    n = 4 * ncopy
    qc = QuantumCircuit(n, n, name=f"b422_{basis}_t{tau}")
    for k in range(ncopy):
        a = 4 * k
        for p, q in ((a, a + 1), (a + 2, a + 3)):
            qc.h(p); qc.cx(p, q)
    qc.barrier(); qc.delay(tau, range(n), unit="us"); qc.barrier()
    if basis == "X": qc.h(range(n))
    qc.measure(range(n), range(n))
    return qc

def bell422_stats(bz, bx, ncopy):
    out = {"bare": [], "code": [], "keep": []}
    for k in range(ncopy):
        a = 4 * k
        zp1 = bz[:, a] ^ bz[:, a + 1]; zp2 = bz[:, a + 2] ^ bz[:, a + 3]
        xp1 = bx[:, a] ^ bx[:, a + 1]; xp2 = bx[:, a + 2] ^ bx[:, a + 3]
        ez = lambda v: float(np.mean(1 - 2 * v))
        bare = (ez(zp1) + ez(xp1)) / 2
        kz = ~(zp1 ^ zp2); kx = ~(xp1 ^ xp2)
        code = (ez(zp1[kz]) + ez(xp1[kx])) / 2
        out["bare"].append(bare); out["code"].append(code); out["keep"].append(float((kz.mean() + kx.mean()) / 2))
    return {k: round(float(np.mean(v)), 4) for k, v in out.items()} | {"best_bare": round(float(max(out["bare"])), 4)}
