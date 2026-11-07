# QuBIOS hardware round 5 — Qubitpage
# (A) "Cloning" test: CNOT fan-out of |+> onto 3 qubits, measured in X and Z.
# (B) Repetition codes n=3 and n=5 (escort ancillas between data), real-time feedback vs none,
#     side by side in the SAME circuit with bare |1> qubits, on hand-picked qubit chains.
import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister

WAIT = 10.0  # us between syndrome rounds


# ---------- layout: disjoint chains of good, connected physical qubits ----------
def _graph(bm):
    adj, e2q = {}, {}
    for a, b in bm.coupling_map.get_edges():
        adj.setdefault(a, set()).add(b); adj.setdefault(b, set()).add(a)
    t = bm.target
    for g in ("cz", "ecr", "cx"):
        if g in t.operation_names:
            for qs, p in t[g].items():
                if p is not None and p.error is not None:
                    e2q[frozenset(qs)] = p.error
    ro = {}
    for qs, p in t["measure"].items():
        ro[qs[0]] = p.error if (p is not None and p.error is not None) else 1.0
    return adj, e2q, ro


def pick_chains(bm, lengths, ro_max=0.03, e2_max=0.01, buffer=True):
    """lengths: list of chain lengths (1 = single qubit). Returns list of physical chains."""
    adj, e2q, ro = _graph(bm)
    good = {q for q in adj if ro.get(q, 1) < ro_max}
    used, blocked, out = set(), set(), []
    order = sorted(range(len(lengths)), key=lambda i: -lengths[i])
    res = [None] * len(lengths)
    for i in order:
        L = lengths[i]; found = None
        for s in sorted(good - blocked, key=lambda q: ro[q]):
            stack = [[s]]
            while stack and found is None:
                path = stack.pop()
                if len(path) == L:
                    found = path; break
                for n in sorted(adj[path[-1]], key=lambda n: ro.get(n, 1), reverse=True):
                    if n in good and n not in blocked and n not in path and \
                            e2q.get(frozenset((path[-1], n)), 1) < e2_max:
                        stack.append(path + [n])
            if found: break
        if found is None:
            return None
        res[i] = found
        for q in found:
            used.add(q); blocked.add(q)
            if buffer: blocked |= adj[q]
    return res


# ---------- repetition code ----------
def decoder_table(n):
    """syndrome value (bit j = parity of data j, j+1) -> min-weight list of data indices to flip."""
    tab = {}
    for v in range(1, 2 ** (n - 1)):
        s = [(v >> j) & 1 for j in range(n - 1)]
        e = [0]
        for j in range(n - 1):
            e.append(e[-1] ^ s[j])
        if sum(e) > n - sum(e):
            e = [1 - x for x in e]
        tab[v] = [i for i in range(n) if e[i]]
    return tab


def rep_circuit(R, feedback, copies, wait=WAIT):
    """copies: list of code sizes n (n=1 means bare qubit). Each copy is a chain d a d a ... d.
    Virtual qubit order = copies concatenated. All data prepared in |1>."""
    sizes = [2 * n - 1 for n in copies]
    q = QuantumRegister(sum(sizes), "q")
    regs, syn = [], {}
    for k, n in enumerate(copies):
        for r in range(R):
            if n > 1:
                syn[(k, r)] = ClassicalRegister(n - 1, f"s{k}_{r}"); regs.append(syn[(k, r)])
    fin = ClassicalRegister(sum(copies), "fin")
    qc = QuantumCircuit(q, *regs, fin, name=f"{'fb' if feedback else 'off'}_R{R}")
    off = np.cumsum([0] + sizes)
    data = [[off[k] + 2 * i for i in range(n)] for k, n in enumerate(copies)]
    anc = [[off[k] + 2 * i + 1 for i in range(n - 1)] for k, n in enumerate(copies)]
    tabs = {n: decoder_table(n) for n in set(copies) if n > 1}
    for k in range(len(copies)):
        for d in data[k]:
            qc.x(d)
    for r in range(R):
        qc.barrier(); qc.delay(wait, q, unit="us"); qc.barrier()
        for k, n in enumerate(copies):
            if n == 1:
                continue
            for j, a in enumerate(anc[k]):
                qc.cx(data[k][j], a); qc.cx(data[k][j + 1], a)
            for j, a in enumerate(anc[k]):
                qc.measure(a, syn[(k, r)][j])
            for a in anc[k]:
                qc.reset(a)
            if feedback:
                for v, flips in tabs[n].items():          # real-time feed-forward (if_test is supported on IBM)
                    with qc.if_test((syn[(k, r)], v)):
                        for i in flips:
                            qc.x(data[k][i])
    qc.barrier()
    f = 0
    for k, n in enumerate(copies):
        for d in data[k]:
            qc.measure(d, fin[f]); f += 1
    return qc


def analyse_rep(bits, copies):
    """bits: [shots, sum(copies)] bool. Returns dict n -> list of logical error rates per copy."""
    out, f = {}, 0
    for n in copies:
        blk = bits[:, f:f + n]; f += n
        logical1 = blk.sum(1) * 2 > n
        out.setdefault(n, []).append(float((~logical1).mean()))
    return out


# ---------- cloning test ----------
def clone_circuit(basis, ntrip, nsing):
    """ntrip triples (chain of 3): H on middle? no - H on first, CX 0->1, CX 1->2.
    nsing single qubits in |+>. Measure everything in `basis` (X or Z)."""
    q = QuantumRegister(3 * ntrip + nsing, "q")
    c = ClassicalRegister(3 * ntrip + nsing, "fin")
    qc = QuantumCircuit(q, c, name=f"clone_{basis}")
    for t in range(ntrip):
        a, b, d = 3 * t, 3 * t + 1, 3 * t + 2
        qc.h(a); qc.cx(a, b); qc.cx(b, d)
    for s in range(nsing):
        qc.h(3 * ntrip + s)
    qc.barrier()
    if basis == "X":
        qc.h(q)
    qc.measure(q, c)
    return qc


def analyse_clone(bx, bz, ntrip, nsing):
    T = 3 * ntrip
    tx = bx[:, :T].reshape(len(bx), ntrip, 3)
    tz = bz[:, :T].reshape(len(bz), ntrip, 3)
    return {
        # if three true copies of |+>, each qubit would read '+' (bit 0) ~100%
        "clone_each_plus": round(float((~tx).mean()), 4),
        "single_plus": round(float((~bx[:, T:]).mean()), 4) if nsing else None,
        # GHZ signature: XXX parity even, Z outcomes all equal
        "clone_XXX_even": round(float((tx.sum(2) % 2 == 0).mean()), 4),
        "clone_Z_all_equal": round(float(((tz.sum(2) == 0) | (tz.sum(2) == 3)).mean()), 4),
        "clone_Z_first_is_1": round(float(tz[:, :, 0].mean()), 4),
    }


def pick_chains_auto(bm, lengths):
    for buf in (True, False):
        for ro_max, e2_max in ((0.03, 0.01), (0.04, 0.015), (0.06, 0.02), (0.1, 0.05)):
            ch = pick_chains(bm, lengths, ro_max, e2_max, buf)
            if ch is not None:
                return ch, dict(buffer=buf, ro_max=ro_max, e2_max=e2_max)
    raise RuntimeError("no chain set found")
