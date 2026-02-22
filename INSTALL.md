# Installing QuBIOS
### Qubitpage® Quantum BIOS Framework
> [🌐 qubitpage.com](https://qubitpage.com)

---

## Requirements

- Python 3.10+
- pip / virtualenv
- `stim` >= 1.13 (required — the real QEC simulation engine)
- `pymatching` >= 2.1 (optional but recommended — faster syndrome decoding)
- `numpy` >= 1.26

---

## Quick Install

```bash
git clone https://github.com/qubitpage/QuBIOS.git
cd QuBIOS
pip install -r requirements.txt
```

## Core Dependencies

```
stim>=1.13
pymatching>=2.1
numpy>=1.26
```

Install:
```bash
pip install stim pymatching numpy
```

For IBM Quantum hardware support:
```bash
pip install qiskit qiskit-aer qiskit-ibm-runtime
```

---

## Verify Installation

```python
from qubios import SteaneQEC

qec = SteaneQEC(physical_error_rate=0.001)
stats = qec.run_qec_cycle(shots=1000, rounds=1)
print(f"Fidelity: {stats.fidelity_estimate:.6f}")
print(f"Logical error rate: {stats.logical_error_rate:.2e}")
# Expected: fidelity≈0.999979, logical p≈2.1e-5
```

---

## Run Tests

```bash
python tests/test_core.py
```

Full test suite takes 2–5 minutes (real Stim circuit simulation).

---

## Integration with QLang

QuBIOS powers the [QLang](https://github.com/qubitpage/QLang) quantum programming language:

```python
# QLang uses QuBIOS internally via the QuBIOS API
# No additional setup needed — QuBIOS activates automatically
# when you run QLang circuits with error correction

from qubios import QubiLogicEngine
engine = QubiLogicEngine(n_escorts=2, n_transit=3)
result = engine.benchmark_comparison("bell", shots=4096)
```

---

## Environment Variables (IBM Quantum)

| Variable | Required | Description |
|----------|----------|-------------|
| `IBM_QUANTUM_TOKEN` | No | IBM Quantum Platform token (for real hardware) |

```bash
export IBM_QUANTUM_TOKEN="your-token"  # never hardcode
```

IBM token obtained at: **https://quantum.ibm.com**

---

<div align="center">
<strong><a href="https://qubitpage.com">🌐 Qubitpage®</a></strong>
<br/><em>© 2026 Qubitpage® — All rights reserved</em>
</div>
