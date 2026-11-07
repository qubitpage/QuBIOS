# QuBIOS hardware round 13 — Qubitpage.  Does a big superposition die N times faster (independent noise)
# or N^2 times faster (correlated "superdecoherence")?  GHZ_N held for tau microseconds, X-parity coherence measured.
import numpy as np
from qiskit import QuantumCircuit

def ghz_wait_circuit(N, tau):
    qc = QuantumCircuit(N, N, name=f"ghzw{N}_t{tau}")
    m = N // 2; qc.h(m)
    for k in range(1, max(m, N - 1 - m) + 1):
        if m - k >= 0: qc.cx(m - k + 1, m - k)
        if m + k <= N - 1: qc.cx(m + k - 1, m + k)
    if tau > 0:
        qc.barrier(); qc.delay(tau, range(N), unit="us")
    # undo the GHZ (disentangle) so coherence maps onto ONE qubit: robust against readout of N qubits
    for k in range(max(m, N - 1 - m), 0, -1):
        if m + k <= N - 1: qc.cx(m + k - 1, m + k)
        if m - k >= 0: qc.cx(m - k + 1, m - k)
    qc.h(m)
    qc.measure(range(N), range(N))
    return qc

def ghz_wait_stat(bits, N):
    m = N // 2
    others = np.delete(bits, m, axis=1)
    p_center0 = float((~bits[:, m]).mean())                     # 1.0 = perfect coherence, 0.5 = none
    clean = float((~others.any(1)).mean())                       # all other qubits back to 0
    return {"P_center0": round(p_center0, 4), "coherence": round(2 * p_center0 - 1, 4), "others_clean": round(clean, 4)}
