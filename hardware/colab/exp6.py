# QuBIOS hardware round 9 — CONFIRMATION of the "conveyor beats stay-put" effect. Qubitpage.
# Harder test, designed to KILL the effect if it is an artefact:
#  * stronger DD for everyone: XY4 (4 pulses, robust to pulse errors) instead of a 2-pulse echo
#  * two different quantum states: |+> and |+i>  (an arbitrary-state memory, not only one basis)
#  * identical physical chains for conveyor and stay-put; stay-put also gets a dummy mid-circuit
#    measurement on its chain neighbour every step (same measurement-induced dephasing)
#  * two different processors (ibm_marrakesh, ibm_kingston)
import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister


def _xy4(qc, qs, t):
    for q in qs:
        qc.delay(t / 8, q, unit="us"); qc.x(q); qc.delay(t / 4, q, unit="us"); qc.y(q)
        qc.delay(t / 4, q, unit="us"); qc.x(q); qc.delay(t / 4, q, unit="us"); qc.y(q); qc.delay(t / 8, q, unit="us")


def _prep(qc, q, st):
    qc.h(q)
    if st == "+i": qc.s(q)


def _meas(qc, q, c, st):
    if st == "+i": qc.sdg(q)
    qc.h(q); qc.measure(q, c)


def hop_circuit(H, st, nc, mode, wait=10.0):
    """mode 'conv': state hops to a fresh qubit every step (one-bit teleportation, software Z frame).
       mode 'stay': state parked on chain qubit 0; chain qubit 1 gets a dummy measurement each step."""
    L = H + 1
    q = QuantumRegister(nc * L, "q")
    ret = ClassicalRegister(nc * H, "ret"); fin = ClassicalRegister(nc, "fin")
    qc = QuantumCircuit(q, ret, fin, name=f"{mode}_{st}_H{H}")
    ch = [[k * L + i for i in range(L)] for k in range(nc)]
    for c in ch: _prep(qc, c[0], st)
    for s in range(H):
        qc.barrier()
        _xy4(qc, [c[s] if mode == "conv" else c[0] for c in ch], wait)
        qc.barrier()
        for k, c in enumerate(ch):
            if mode == "conv":
                qc.cx(c[s], c[s + 1]); qc.h(c[s]); qc.measure(c[s], ret[k * H + s])
            else:
                qc.measure(c[1], ret[k * H + s])
    qc.barrier()
    for k, c in enumerate(ch):
        _meas(qc, c[H] if mode == "conv" else c[0], fin[k], st)
    return qc


def hop_values(ret, fin, H, nc, mode):
    v = fin.copy()
    if mode == "conv":
        for k in range(nc):
            v[:, k] ^= (ret[:, k * H:(k + 1) * H].sum(1) % 2 == 1)
    return v                                   # 0 = correct


def vote3(v):
    g = v.shape[1] // 3
    return round(float(np.mean([(v[:, 3 * i:3 * i + 3].sum(1) >= 2).mean() for i in range(g)])), 4)


def wilson(p, n, z=1.96):
    d = 1 + z * z / n; c = (p + z * z / (2 * n)) / d; h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return round(float(c - h), 4), round(float(c + h), 4)
