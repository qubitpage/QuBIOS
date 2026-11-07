# QuBIOS hardware round 10 — Qubitpage. Needs exp2 (_graph), exp3 (_wait), exp4 (mem_circuit, _matcher), exp5, exp6 (_xy4, wilson).
# (1) CALIBRATION-AWARE PLACEMENT: chains chosen by a coherence-weighted score (T1, T2, readout, 2q error) instead of readout only
# (2) DIAGNOSE the stay-put qubit: what kills a parked |+>? neighbour measurement? DD type?
# (3) LEARNED DECODER: estimate every matching-graph edge probability from the measured detection-event
#     correlations (p_ij method used by Google), 2-fold cross-validated, vs the uniform-weight decoder
import numpy as np
import pymatching
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister


# ---------------- (1) scored placement ----------------
def qubit_scores(bm, wait_us=10.0):
    adj, e2q, ro = _graph(bm)
    sc = {}
    for q in adj:
        p = bm.target.qubit_properties[q] if bm.target.qubit_properties else None
        t1 = (p.t1 or 1e-6) * 1e6 if p is not None and p.t1 else 1.0
        t2 = (p.t2 or 1e-6) * 1e6 if p is not None and p.t2 else 1.0
        sc[q] = ro.get(q, 1.0) + wait_us / t2 + 0.5 * wait_us / t1      # estimated error per 10 us step + readout
    return adj, e2q, sc


def pick_chains_scored(bm, lengths, keep=0.6, e2_max=0.012, buffer=True):
    adj, e2q, sc = qubit_scores(bm)
    ranked = sorted(sc, key=sc.get)
    for frac in (keep, 0.75, 0.9, 1.0):
        good = set(ranked[:int(len(ranked) * frac)])
        for e2m in (e2_max, 0.02, 0.05):
            for buf in ((buffer, False) if buffer else (False,)):
                res, blocked = [None] * len(lengths), set()
                ok = True
                for i in sorted(range(len(lengths)), key=lambda i: -lengths[i]):
                    L, best = lengths[i], None
                    for s in sorted(good - blocked, key=sc.get):
                        stack = [([s], sc[s])]
                        while stack:
                            path, cost = stack.pop()
                            if len(path) == L:
                                if best is None or cost < best[1]: best = (path, cost)
                                continue
                            for n in adj[path[-1]]:
                                if n in good and n not in blocked and n not in path and e2q.get(frozenset((path[-1], n)), 1) < e2m:
                                    stack.append((path + [n], cost + sc[n]))
                        if best is not None and L <= 3: break       # short chains: first good start is enough
                        if best is not None and len(stack) == 0: break
                    if best is None: ok = False; break
                    res[i] = best[0]
                    for q in best[0]:
                        blocked.add(q)
                        if buf: blocked |= adj[q]
                if ok:
                    return res, dict(frac=frac, e2_max=e2m, buffer=buf,
                                     mean_score=round(float(np.mean([sc[q] for c in res for q in c])), 4))
    raise RuntimeError("no scored chain set")


# ---------------- (2) stay-put diagnosis ----------------
def _xy8(qc, qs, t):
    seq = ["x", "y", "x", "y", "y", "x", "y", "x"]
    for q in qs:
        qc.delay(t / 16, q, unit="us")
        for i, g in enumerate(seq):
            getattr(qc, g)(q); qc.delay(t / 8 if i < 7 else t / 16, q, unit="us")


def park_circuit(H, n, variant, wait=10.0):
    """n pairs [park, neighbour]; |+> parked for H steps.  variant: 'xy4_meas' (neighbour measured each step),
    'xy4' (neighbour idle), 'xy8', 'none' (no DD)."""
    q = QuantumRegister(2 * n, "q"); fin = ClassicalRegister(n, "fin")
    dm = ClassicalRegister(max(1, n * H), "dm")
    qc = QuantumCircuit(q, dm, fin, name=f"park_{variant}_H{H}")
    P = [2 * k for k in range(n)]; N = [2 * k + 1 for k in range(n)]
    for p in P: qc.h(p)
    for s in range(H):
        qc.barrier()
        if variant.startswith("xy4"): _xy4(qc, P, wait)
        elif variant == "xy8": _xy8(qc, P, wait)
        else: qc.delay(wait, P, unit="us")
        qc.barrier()
        if variant == "xy4_meas":
            for k, nb in enumerate(N): qc.measure(nb, dm[k * H + s])
        else:
            qc.delay(1.5, P + N, unit="us")      # same wall-clock as a measurement window
    qc.barrier()
    for k, p in enumerate(P): qc.h(p); qc.measure(p, fin[k])
    return qc


# ---------------- (3) learned (correlation-calibrated) decoder ----------------
def _detectors(syn, fin, copies, R):
    """per code copy: detection events [shots, (R+1)*J] and data qubit-0 final bit."""
    out, f, s0 = [], 0, 0
    nsyn = sum(n - 1 for n in copies)
    for n in copies:
        fb = fin[:, f:f + n].astype(np.uint8); f += n
        if n == 1: out.append(None); continue
        J = n - 1
        S = np.stack([syn[:, r * nsyn + s0: r * nsyn + s0 + J] for r in range(R)], 1).astype(np.uint8); s0 += J
        layers = np.concatenate([S, (fb[:, :-1] ^ fb[:, 1:])[:, None, :]], 1)
        prev = np.concatenate([np.zeros_like(layers[:, :1]), layers[:, :-1]], 1)
        out.append(((layers ^ prev).reshape(len(fb), -1), fb[:, 0], n))
    return out


def _edges(n, R):
    J = n - 1; node = lambda r, j: r * J + j; E = []
    for r in range(R + 1):
        for i in range(n):
            if i == 0: E.append((node(r, 0), None, True))
            elif i == n - 1: E.append((node(r, J - 1), None, False))
            else: E.append((node(r, i - 1), node(r, i), False))
        if r < R:
            for j in range(J): E.append((node(r, j), node(r + 1, j), False))
    return E


def learned_matcher(det, n, R):
    """Edge probabilities from detection-event statistics (pij formula), boundary edges from residual."""
    X = det.astype(float); m = X.mean(0); E = _edges(n, R)
    XX = (X.T @ X) / len(X)
    mt = pymatching.Matching()
    pb = {}
    pij_sum = np.zeros(X.shape[1])
    probs = []
    for a, b, fid in E:
        if b is None: probs.append(None); continue
        num = 4 * (XX[a, b] - m[a] * m[b]); den = 1 - 2 * m[a] - 2 * m[b] + 4 * XX[a, b]
        v = 1 - num / den if den > 1e-9 else 1.0
        p = 0.5 - 0.5 * np.sqrt(max(v, 0.0)); p = float(np.clip(p, 1e-4, 0.45))
        probs.append(p); pij_sum[a] += p; pij_sum[b] += p
    for (a, b, fid), p in zip(E, probs):
        if b is None:   # boundary: whatever detection rate the bulk edges do not explain
            p = float(np.clip((m[a] - pij_sum[a]) / max(1 - 2 * pij_sum[a], 1e-3), 1e-4, 0.45))
        w = float(np.log((1 - p) / p))
        if b is None: mt.add_boundary_edge(a, fault_ids={0} if fid else set(), weight=w, error_probability=p)
        else: mt.add_edge(a, b, fault_ids={0} if fid else set(), weight=w, error_probability=p)
    return mt


def compare_decoders(syn, fin, copies, R, basis="Z"):
    """returns per copy (uniform_matching_err, learned_matching_err) with 2-fold cross-validation."""
    exp = 1 if basis == "Z" else 0
    res = []
    for item in _detectors(syn, fin, copies, R):
        if item is None: res.append(None); continue
        det, d0, n = item
        half = len(det) // 2
        u = _matcher(n, R).decode_batch(det)[:, 0]
        uerr = float(((d0 ^ u) != exp).mean())
        errs = []
        for tr, te in ((slice(0, half), slice(half, None)), (slice(half, None), slice(0, half))):
            mt = learned_matcher(det[tr], n, R)
            pr = mt.decode_batch(det[te])[:, 0]
            errs.append(float(((d0[te] ^ pr) != exp).mean()))
        res.append((round(uerr, 4), round(float(np.mean(errs)), 4)))
    return res
