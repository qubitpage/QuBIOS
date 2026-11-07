# QuBIOS hardware round 6 — Qubitpage. Needs exp2 functions (pick_chains_auto, decoder_table, rep_circuit, analyse_rep).
# (C) CONVEYOR: the user's "hand the information to a fresh qubit and retire the old one" idea, done with
#     one-bit teleportation hops (1 CNOT + 1 X-measurement, frame tracked in software, no feed-forward).
#     Question: does refreshing qubits beat a qubit that stays put, for the same total time?
# (P) PHASE-FLIP code protecting a SUPERPOSITION |+_L> = |+++>: X-parity checks with escort ancillas,
#     real-time feedback (Z corrections), with and without MANUAL dynamical decoupling inside the waits
#     (IBM refuses automatic DD on dynamic circuits).
import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister


def _wait(qc, qubits, t, dd):
    if not dd:
        qc.delay(t, qubits, unit="us"); return
    for q in qubits:                       # CPMG-style echo: t/4 - X - t/2 - X - t/4
        qc.delay(t / 4, q, unit="us"); qc.x(q); qc.delay(t / 2, q, unit="us"); qc.x(q); qc.delay(t / 4, q, unit="us")


def conveyor_circuit(H, basis, nconv, wait=10.0, dd=False):
    """nconv conveyors of length H+1 plus nconv bare qubits held for the same time.
    basis 'Z': store |1>.  basis 'X': store |+>.  Each step: wait, CX(old->fresh), measure old in X (retire)."""
    L = H + 1
    q = QuantumRegister(nconv * L + nconv, "q")
    ret = ClassicalRegister(max(1, nconv * H), "ret")
    fin = ClassicalRegister(2 * nconv, "fin")      # [conveyor ends..., bare...]
    qc = QuantumCircuit(q, ret, fin, name=f"conv{basis}_H{H}{'_dd' if dd else ''}")
    chains = [[k * L + i for i in range(L)] for k in range(nconv)]
    bare = [nconv * L + k for k in range(nconv)]
    for c in chains:
        (qc.x if basis == "Z" else qc.h)(c[0])
    for b in bare:
        (qc.x if basis == "Z" else qc.h)(b)
    for s in range(H):
        qc.barrier()
        _wait(qc, [c[s] for c in chains] + bare, wait, dd)
        qc.barrier()
        for k, c in enumerate(chains):
            qc.cx(c[s], c[s + 1]); qc.h(c[s]); qc.measure(c[s], ret[k * H + s])
    qc.barrier()
    for k, c in enumerate(chains):
        if basis == "X": qc.h(c[H])
        qc.measure(c[H], fin[k])
    for k, b in enumerate(bare):
        if basis == "X": qc.h(b)
        qc.measure(b, fin[nconv + k])
    return qc


def analyse_conveyor(ret, fin, basis, H, nconv):
    """ret: [shots, nconv*H], fin: [shots, 2*nconv] bools. Expected: Z-basis -> 1, X-basis -> 0 (|+>)."""
    conv, bare = [], []
    for k in range(nconv):
        v = fin[:, k].copy()
        if basis == "X" and H > 0:                    # Pauli frame: each retirement outcome 1 = Z on the new qubit
            v ^= (ret[:, k * H:(k + 1) * H].sum(1) % 2 == 1)
        exp = True if basis == "Z" else False
        conv.append(float((v != exp).mean()))
        bare.append(float((fin[:, nconv + k] != exp).mean()))
    return round(float(np.mean(conv)), 4), round(float(np.mean(bare)), 4)


def phase_code_circuit(R, copies, feedback=True, dd=False, wait=10.0):
    """Phase-flip repetition code holding |+_L>=|+..+>; copies list of n (1 = bare |+>)."""
    sizes = [2 * n - 1 for n in copies]
    q = QuantumRegister(sum(sizes), "q")
    regs, syn = [], {}
    for k, n in enumerate(copies):
        for r in range(R):
            if n > 1:
                syn[(k, r)] = ClassicalRegister(n - 1, f"s{k}_{r}"); regs.append(syn[(k, r)])
    fin = ClassicalRegister(sum(copies), "fin")
    qc = QuantumCircuit(q, *regs, fin, name=f"ph{'_fb' if feedback else '_off'}{'_dd' if dd else ''}_R{R}")
    off = np.cumsum([0] + sizes)
    data = [[int(off[k]) + 2 * i for i in range(n)] for k, n in enumerate(copies)]
    anc = [[int(off[k]) + 2 * i + 1 for i in range(n - 1)] for k, n in enumerate(copies)]
    tabs = {n: decoder_table(n) for n in set(copies) if n > 1}
    alld = [d for D in data for d in D]
    for d in alld:
        qc.h(d)
    for r in range(R):
        qc.barrier(); _wait(qc, alld, wait, dd); qc.barrier()
        for k, n in enumerate(copies):
            if n == 1: continue
            for d in data[k]: qc.h(d)                       # X-parity checks
            for j, a in enumerate(anc[k]):
                qc.cx(data[k][j], a); qc.cx(data[k][j + 1], a)
            for d in data[k]: qc.h(d)
            for j, a in enumerate(anc[k]): qc.measure(a, syn[(k, r)][j])
            for a in anc[k]: qc.reset(a)
            if feedback:
                for v, flips in tabs[n].items():
                    with qc.if_test((syn[(k, r)], v)):
                        for i in flips: qc.z(data[k][i])
    qc.barrier()
    f = 0
    for k, n in enumerate(copies):
        for d in data[k]:
            qc.h(d); qc.measure(d, fin[f]); f += 1
    return qc


def analyse_phase(bits, copies):
    out, f = {}, 0
    for n in copies:
        blk = bits[:, f:f + n]; f += n
        wrong = blk.sum(1) * 2 > n                         # expected all 0 (= |+>)
        out.setdefault(n, []).append(float(wrong.mean()))
    return {n: round(float(np.mean(v)), 4) for n, v in out.items()}
