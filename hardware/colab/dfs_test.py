# QuBIOS — "opposed pair" (decoherence-free subspace) test vs single qubit vs aligned pair.
import json, numpy as np
from qiskit import QuantumCircuit
from qiskit.transpiler import generate_preset_pass_manager

NP = 20                     # copies of each kind
TAUS = [0, 10, 20, 40, 80, 160]

def dfs_circuit(tau):
    # qubits: [0, 2*NP)  opposed pairs |01>+|10> ; [2NP, 4NP) aligned pairs |00>+|11> ; [4NP, 5NP) single |+>
    n = 5 * NP
    qc = QuantumCircuit(n, n, name=f"dfs_tau{tau}")
    for k in range(NP):
        a, b = 2 * k, 2 * k + 1                  # opposed
        qc.x(b); qc.h(a); qc.cx(a, b)
        a, b = 2 * NP + 2 * k, 2 * NP + 2 * k + 1  # aligned
        qc.h(a); qc.cx(a, b)
        qc.h(4 * NP + k)                         # single
    qc.barrier()
    if tau > 0:
        qc.delay(tau, range(n), unit="us"); qc.barrier()
    for k in range(NP):
        for a in (2 * k, 2 * NP + 2 * k):
            qc.cx(a, a + 1); qc.h(a)             # decode: logical |+> -> first qubit 0
        qc.h(4 * NP + k)
    qc.measure(range(n), range(n))
    return qc

def analyse(m):
    out = {}
    opp_err = [float((m[:, 2 * k] != 0).mean()) for k in range(NP)]
    # opposed: partner must read 1 (leakage out of the subspace = detected error)
    keep = [(m[:, 2 * k + 1] == 1) for k in range(NP)]
    opp_ps = [float((m[keep[k], 2 * k] != 0).mean()) if keep[k].any() else None for k in range(NP)]
    ali_err = [float((m[:, 2 * NP + 2 * k] != 0).mean()) for k in range(NP)]
    keep2 = [(m[:, 2 * NP + 2 * k + 1] == 0) for k in range(NP)]
    ali_ps = [float((m[keep2[k], 2 * NP + 2 * k] != 0).mean()) if keep2[k].any() else None for k in range(NP)]
    sgl = [float((m[:, 4 * NP + k] != 0).mean()) for k in range(NP)]
    f = lambda v: round(float(np.mean([x for x in v if x is not None])), 4)
    return {"opposed": f(opp_err), "opposed_ps": f(opp_ps), "opposed_keep": round(float(np.mean([k.mean() for k in keep])), 3),
            "aligned": f(ali_err), "aligned_ps": f(ali_ps), "single": f(sgl),
            "opposed_best": round(min(opp_err), 4), "single_best": round(min(sgl), 4)}

if __name__ == "__main__":
    from qiskit_aer import AerSimulator
    sim = AerSimulator(method="stabilizer")
    c = dfs_circuit(20)
    counts = sim.run(c, shots=300).result().get_counts()
    rows = []
    for k, v in counts.items():
        rows += [[ch == "1" for ch in k[::-1]]] * v
    print(analyse(np.array(rows)))
    from qiskit_ibm_runtime.fake_provider import FakeMarrakesh
    be = FakeMarrakesh()
    pm = generate_preset_pass_manager(optimization_level=1, backend=be, seed_transpiler=3)
    ic = pm.run(dfs_circuit(0)); print("2q gates", ic.count_ops().get("cz", 0), "swaps-free?", ic.count_ops().get("cz", 0) == 4 * NP * 2)
