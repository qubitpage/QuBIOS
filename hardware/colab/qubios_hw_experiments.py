"""QuBIOS v1.1.1 — real-hardware experiments for IBM Quantum (run in Google Colab or locally).

Experiments (all in ONE Sampler job per DD setting):
  A. Memory vs time: 4x |1_L>, 4x |+_L> (Steane [[7,1,3]] unitary encoder, classical
     Hamming decode at readout) and 4 bare |1>, 4 bare |+>, all idling tau microseconds
     between barriers.  tau in TAUS.
  B. Escort relay (the QuBIOS 'transit ring' idea): a state either rests on one qubit for
     T_RELAY us, or is SWAP-relayed through k escort qubits with equal waits in between.
     k in HOPS (k=0 is the resting reference).  6 chains carry |1>, 6 carry |+>.

The API key is read with getpass (never printed, never stored in a file).
(c) 2026 Qubitpage | MIT
"""
import json, time, numpy as np
from qiskit import QuantumCircuit
from qiskit.transpiler import generate_preset_pass_manager

TAUS = [0, 25, 50, 100, 200]          # us
HOPS = [0, 2, 5, 10]                  # escort relays within T_RELAY
T_RELAY = 100                         # us
SHOTS = 4000
STAB = [[3, 4, 5, 6], [1, 2, 5, 6], [0, 2, 4, 6]]
PIV = {0: [2, 4, 6], 1: [2, 5, 6], 3: [4, 5, 6]}


def encode(qc, b):                    # |0_L> on qubits b..b+6
    for p in PIV: qc.h(b + p)
    for p, ts in PIV.items():
        for t in ts: qc.cx(b + p, b + t)


def memory_circuit(tau):
    nL, nB = 8, 8
    qc = QuantumCircuit(7 * nL + nB, 7 * nL + nB, name=f"mem_tau{tau}")
    for j in range(nL):
        b = 7 * j; encode(qc, b)
        for i in range(7):
            (qc.x if j < 4 else qc.h)(b + i)          # |1_L> = X_L|0_L>,  |+_L> = H^7|0_L>
    B0 = 7 * nL
    for i in range(nB):
        (qc.x if i < 4 else qc.h)(B0 + i)
    qc.barrier()
    if tau > 0:
        qc.delay(tau, range(qc.num_qubits), unit="us"); qc.barrier()
    for j in range(4, nL):
        for i in range(7): qc.h(7 * j + i)
    for i in range(4, nB): qc.h(B0 + i)
    qc.measure(range(qc.num_qubits), range(qc.num_qubits))
    return qc


def relay_circuit(k):
    n_chain, length = 12, max(k, 0) + 1
    nq = n_chain * length
    qc = QuantumCircuit(nq, nq, name=f"relay_k{k}")
    heads = [j * length for j in range(n_chain)]
    for j, h in enumerate(heads):
        (qc.x if j < 6 else qc.h)(h)
    qc.barrier()
    wait = T_RELAY / (k + 1)
    for hop in range(k + 1):
        qc.delay(wait, range(nq), unit="us"); qc.barrier()
        if hop < k:
            for h in heads: qc.swap(h + hop, h + hop + 1)
            qc.barrier()
    for j, h in enumerate(heads):
        if j >= 6: qc.h(h + k)
    qc.measure(range(nq), range(nq))
    return qc


# ── analysis ─────────────────────────────────────────────────────────────
def hamming_decode(bits7):
    s = [int(sum(bits7[i] for i in sp) % 2) for sp in STAB]
    pos = 4 * s[0] + 2 * s[1] + s[2]
    par = int(sum(bits7) % 2)
    return par, pos


def analyse_memory(m):                 # m: [shots, nq] bool, clbit order
    out = {"L1_dec": [], "Lp_dec": [], "L1_ps": [], "Lp_ps": [], "keep": [], "B1": [], "Bp": []}
    for j in range(8):
        blk = m[:, 7 * j:7 * j + 7].astype(int)
        s = np.stack([blk[:, sp].sum(1) % 2 for sp in STAB], 1)
        pos = s[:, 0] * 4 + s[:, 1] * 2 + s[:, 2]
        par = blk.sum(1) % 2
        exp = 1 if j < 4 else 0
        dec = float(((par ^ (pos > 0)) != exp).mean())
        keep = pos == 0
        ps = float((par[keep] != exp).mean()) if keep.any() else None
        out["L1_dec" if j < 4 else "Lp_dec"].append(round(dec, 4))
        out["L1_ps" if j < 4 else "Lp_ps"].append(None if ps is None else round(ps, 4))
        out["keep"].append(round(float(keep.mean()), 4))
    for i in range(8):
        exp = 1 if i < 4 else 0
        out["B1" if i < 4 else "Bp"].append(round(float((m[:, 56 + i] != exp).mean()), 4))
    return out


def analyse_relay(m, k):
    length = k + 1
    errs1, errsp = [], []
    for j in range(12):
        end = j * length + k
        exp = 1 if j < 6 else 0
        (errs1 if j < 6 else errsp).append(round(float((m[:, end] != exp).mean()), 4))
    return {"err_1": errs1, "err_plus": errsp}


def bits_from_result(res):
    arr = next(iter(res.data.values()))
    return arr.to_bool_array(order="little")


def run(backend, dd: bool, mode=None):
    from qiskit_ibm_runtime import SamplerV2
    pm = generate_preset_pass_manager(optimization_level=1, backend=backend, seed_transpiler=11)
    circs = [memory_circuit(t) for t in TAUS] + [relay_circuit(k) for k in HOPS]
    isa = [pm.run(c) for c in circs]
    sampler = SamplerV2(mode=mode or backend)
    try:
        sampler.options.dynamical_decoupling.enable = dd
        if dd: sampler.options.dynamical_decoupling.sequence_type = "XpXm"
    except Exception as e:
        print("DD option not available:", e)
    job = sampler.run(isa, shots=SHOTS)
    print("submitted", job.job_id(), "DD" if dd else "noDD")
    return job, circs, isa


def report(job, circs, isa, dd):
    res = job.result()
    rows = {"job_id": job.job_id(), "dd": dd, "memory": {}, "relay": {}, "two_qubit_gates": {}}
    for c, r, ic in zip(circs, res, isa):
        m = bits_from_result(r)
        ops = ic.count_ops(); rows["two_qubit_gates"][c.name] = int(sum(v for g, v in ops.items() if g in ("cz", "ecr", "cx")))
        rows["delays_in_isa"] = rows.get("delays_in_isa", 0) + int(ops.get("delay", 0))
        if c.name.startswith("mem"):
            rows["memory"][c.name] = analyse_memory(m)
        else:
            rows["relay"][c.name] = analyse_relay(m, int(c.name.split("k")[1]))
    return rows


if __name__ == "__main__":
    # self-test on an ideal stabilizer simulator (no noise): every error must be 0
    from qiskit_aer import AerSimulator
    sim = AerSimulator(method="stabilizer")
    for c in [memory_circuit(50), relay_circuit(5)]:
        m = sim.run(c, shots=200).result().get_memory(c) if False else None
        counts = sim.run(c, shots=200).result().get_counts()
        bits = []
        for k, v in counts.items():
            b = [ch == "1" for ch in k.replace(" ", "")[::-1]]
            bits += [b] * v
        m = np.array(bits)
        print(c.name, analyse_memory(m) if c.name.startswith("mem") else analyse_relay(m, 5))
