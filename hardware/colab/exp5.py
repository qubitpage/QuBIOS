# QuBIOS hardware round 8 — Qubitpage. Needs exp2/exp3/exp4 functions.
# Controls from the round-7 self-critique:
#  * SAME-QUBIT control: conveyor vs stay-put run on the identical physical chain (separate circuits, same layout)
#  * REDUNDANT CONVEYOR: 3 parallel conveyors carrying |+>, final majority vote (no parity checks at all)
#  * per-copy breakdown for the large bit-flip codes
import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister


def stay_circuit(H, nconv, where, wait=10.0, dd=True):
    """Same register shape as conveyor_circuit(H,'X',nconv): |+> parked on chain qubit `where` for H steps.
    Every step also spends the same CX+measure time window (the other chain qubits are idle)."""
    L = H + 1
    q = QuantumRegister(nconv * L + nconv, "q")
    ret = ClassicalRegister(max(1, nconv * H), "ret")
    fin = ClassicalRegister(2 * nconv, "fin")
    qc = QuantumCircuit(q, ret, fin, name=f"stay{where}_H{H}")
    chains = [[k * L + i for i in range(L)] for k in range(nconv)]
    bare = [nconv * L + k for k in range(nconv)]
    hold = [c[0] if where == "first" else c[H] for c in chains]
    for h in hold + bare: qc.h(h)
    for s in range(H):
        qc.barrier(); _wait(qc, hold + bare, wait, dd); qc.barrier()
        # mimic the hop's measurement window with a dummy measurement on a spare chain qubit
        for k, c in enumerate(chains):
            spare = c[1] if where == "first" else c[0]
            qc.measure(spare, ret[k * H + s])
    qc.barrier()
    for k, h in enumerate(hold):
        qc.h(h); qc.measure(h, fin[k])
    for k, b in enumerate(bare):
        qc.h(b); qc.measure(b, fin[nconv + k])
    return qc


def analyse_stay(fin, nconv):
    return round(float(fin[:, :nconv].mean()), 4), round(float(fin[:, nconv:].mean()), 4)


def conveyor_values(ret, fin, H, nconv):
    """Frame-corrected X-basis value (0 = correct |+>) for each conveyor: [shots, nconv]."""
    v = fin[:, :nconv].copy()
    for k in range(nconv):
        if H > 0:
            v[:, k] ^= (ret[:, k * H:(k + 1) * H].sum(1) % 2 == 1)
    return v


def redundant_vote(vals, group=3):
    n = vals.shape[1] // group
    errs = [float((vals[:, g * group:(g + 1) * group].sum(1) * 2 > group).mean()) for g in range(n)]
    return round(float(np.mean(errs)), 4)


def decode_per_copy(syn, fin, copies, R, basis):
    """Matching error for each copy separately (list in copy order)."""
    res, f, s0 = [], 0, 0
    nsyn = sum(n - 1 for n in copies)
    exp = 1 if basis == "Z" else 0
    for n in copies:
        fb = fin[:, f:f + n].astype(np.uint8); f += n
        if n == 1:
            res.append(round(float((fb[:, 0] != exp).mean()), 4)); continue
        J = n - 1
        S = np.stack([syn[:, r * nsyn + s0: r * nsyn + s0 + J] for r in range(R)], 1).astype(np.uint8); s0 += J
        layers = np.concatenate([S, (fb[:, :-1] ^ fb[:, 1:])[:, None, :]], 1)
        prev = np.concatenate([np.zeros_like(layers[:, :1]), layers[:, :-1]], 1)
        pred = _matcher(n, R).decode_batch((layers ^ prev).reshape(len(fb), -1))[:, 0]
        res.append(round(float(((fb[:, 0] ^ pred) != exp).mean()), 4))
    return res
