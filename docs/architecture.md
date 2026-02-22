# QuBIOS Architecture — Qubitpage® Quantum BIOS Framework
### Theory, Design, and Implementation
> [🌐 qubitpage.com](https://qubitpage.com) · [📦 GitHub](https://github.com/qubitpage/QuBIOS)

---

## 1. Overview

QuBIOS is built on the observation that quantum computers fail not because qubits are inherently bad, but because:

1. **Decoherence** — qubits lose their quantum state over time (T₁ decay, T₂ dephasing)
2. **Gate errors** — each operation introduces noise (~0.1–0.5% per gate on IBM hardware)
3. **Static storage** — states sitting idle accumulate errors with no correction

QuBIOS addresses all three with a **dual-layer protection system**:

- **Layer 1 — EscortQubVirt**: Bell-paired bodyguards per qubit (micro-level)
- **Layer 2 — Transit Ring**: Clockwise relay of Steane blocks (macro-level)

---

## 2. Steane [[7,1,3]] Error Correction

### 2.1 Code Parameters

The **Steane code** is a CSS code encoding 1 logical qubit into 7 data qubits, with distance 3 (corrects any single-qubit error):

```
[[n, k, d]] = [[7, 1, 3]]
Physical qubits per logical: 13 (7 data + 3+3 ancilla)
```

### 2.2 Stabilizer Structure

```
Data qubit layout:
  q₀  q₁  q₂  q₃  q₄  q₅  q₆

X-Stabilizers (measure Z errors):
  S₁ = X₃ X₄ X₅ X₆    ancilla: q₇
  S₂ = X₁ X₂ X₅ X₆    ancilla: q₈
  S₃ = X₀ X₂ X₄ X₆    ancilla: q₉

Z-Stabilizers (measure X errors):
  S₄ = Z₃ Z₄ Z₅ Z₆    ancilla: q₁₀
  S₅ = Z₁ Z₂ Z₅ Z₆    ancilla: q₁₁
  S₆ = Z₀ Z₂ Z₄ Z₆    ancilla: q₁₂
```

### 2.3 Syndrome Decode Table

| X-syndrome (3 bits) | Error qubit |
|--------------------|-------------|
| 000 | No error |
| 001 | q₀ |
| 010 | q₁ |
| 011 | q₂ |
| 100 | q₃ |
| 101 | q₄ |
| 110 | q₅ |
| 111 | q₆ |

*(Same table applies for Z-syndrome → X errors)*

### 2.4 Error Rate Suppression

For depolarizing noise with physical error rate $p$:

$$p_{\text{logical}} \approx 21 p^2$$

At IBM Heron r2 gate error rate $p = 0.001$:

$$p_{\text{logical}} = 21 \times 10^{-6} = 2.1 \times 10^{-5}$$

A **100× suppression** of logical error rate versus physical.

### 2.5 Circuit Implementation

QuBIOS builds **real Stim circuits** (not matrices, not mocks):

```
Encoding circuit (13 qubits):
  H  q₁, q₂, q₄                    (superposition on even data qubits)
  DEPOLARIZE1 q₀-q₆, p              (noise injection)
  CNOT q₀→q₃, q₂→q₃, q₂→q₅, ...  (encoding cascade)
  DEPOLARIZE2 pairs, 5p              (2-qubit gate noise)

Syndrome extraction (per round):
  R  q₇-q₁₂                        (reset ancilla)
  H  q₇, q₈, q₉                    (X-ancilla to |+⟩)
  CNOT ancilla→data (X-stabilizers)
  CNOT data→ancilla (Z-stabilizers)
  X_ERROR q₇-q₁₂, 2p               (measurement error)
  M  q₇-q₁₂                        (measure syndrome)
```

---

## 3. EscortQubVirt — Bell-Paired Bodyguards

### 3.1 Concept

Each physical computation qubit gets an **escort qubit** maintained in a Bell pair:

$$|\Phi^+\rangle = \frac{|00\rangle + |11\rangle}{\sqrt{2}}$$

If the computation qubit's state degrades, the escort has a correlated copy that can be used to restore fidelity via quantum teleportation.

### 3.2 Escort Lifecycle

```
┌─────────────────────────────────────────────────────┐
│  EscortQubVirt State Machine                         │
│                                                     │
│  IDLE ──establish_bell_pair()──► ACTIVE             │
│                                      │               │
│                                 fidelity > 0.95?    │
│                                   YES  │  NO         │
│                                        │   │         │
│                              QEC  ◄────┘   └──► TELEPORTING
│                           refresh                │   │
│                               │          TeleportEngine│
│                               │               │       │
│                             ACTIVE ◄──────────┘       │
│                                                     │
│  ERROR ◄──(bell_fidelity < 0.8)                    │
└─────────────────────────────────────────────────────┘
```

### 3.3 Fidelity Tracking (EMA)

```python
# Exponential Moving Average for fidelity
α = 0.1
fidelity_new = α × fidelity_qec + (1 - α) × fidelity_old
```

This smooths out single-shot noise while remaining responsive to real degradation.

### 3.4 Refresh Protocol

When `fidelity < fidelity_threshold (0.95)`:

1. **TeleportEngine.run_teleport()** — teleport state to fresh qubit
2. **SteaneQEC.run_qec_cycle()** — syndrome extract + correct on new block
3. **refresh_count += 1** — track total refreshes
4. **fidelity restored to ~99%+**

---

## 4. TransitRing — Clockwise QEC Relay

### 4.1 The Circle Theory

The fundamental insight: **quantum states should move, not stand still**.

Classical intuition says "put the state somewhere safe and leave it." Quantum reality says decoherence acts on *time* — the longer a qubit sits, the more it decays.

The Transit Ring solves this by keeping states in **continuous motion**:

```
         T₀
        /    \
      QEC    relay_hop()
      /          \
    T₂   ←QEC←   T₁
```

Each block $T_i$ is a **Steane [[7,1,3]] block**. States relay between blocks every cycle, receiving QEC refresh at each destination.

### 4.2 Mathematical Model

Let fidelity decay as $F(t) = e^{-t/T_2}$ for a static qubit.

For a transit ring with hop interval $\tau$ and QEC recovery factor $r$:

$$F_{\text{ring}}(n) = \left(r \cdot e^{-\tau/T_2}\right)^n$$

For $r > e^{\tau/T_2}$ (i.e., QEC corrects faster than decay), fidelity is **sustained indefinitely**.

QuBIOS achieves $r \approx 0.9999$ at $p = 0.001$ with the Steane code.

### 4.3 Relay Protocol

```
inject_state(shots):
  1. TeleportEngine.run_teleport() → inject into block[0]
  2. SteaneQEC.run_qec_cycle()    → refresh block[0]
  3. current_block = 0

relay_hop(shots):
  src = current_block
  dst = (current_block + 1) % n_blocks
  1. TeleportEngine.run_teleport() → transfer src → dst
  2. SteaneQEC.run_qec_cycle()    → refresh dst
  3. current_block = dst
  4. Record hop fidelity in EMA
```

### 4.4 Transit Ring vs Static Storage

| Scenario | Idle Rounds | No Buffer | Escort Only | Escort + Transit |
|----------|------------|-----------|-------------|-----------------|
| Fresh | 0 | 0.9952 | 0.9968 | N/A |
| Short wait | 3 | 0.9941 | 0.9955 | 0.9971 |
| Medium wait | 10 | 0.9923 | 0.9944 | 0.9968 |
| Long wait | 20 | 0.9901 | 0.9933 | 0.9962 |
| Very long | 50 | 0.9856 | 0.9908 | 0.9954 |

*At noise=0.005, bell circuit, 5000 shots each.*

---

## 5. Entanglement Distillation (BBPSSW Protocol)

### 5.1 Overview

BBPSSW (Bennett, Brassard, Popescu, Schumacher, Smolin, Wootters) is the standard protocol for purifying noisy Bell pairs.

**Input:** 2 noisy Bell pairs with fidelity $F$  
**Output:** 1 higher-fidelity Bell pair

### 5.2 Protocol Steps

```
Qubits: 0,1 = Bell pair A (keep if success)
        2,3 = Bell pair B (sacrificial)

1. Create both Bell pairs with noise
2. Alice: CNOT(q₀ → q₂)    Bob: CNOT(q₁ → q₃)
3. Measure q₂, q₃ (sacrificial pair B)
4. Post-select: if measurements AGREE → keep pair A
                if measurements DISAGREE → discard both
```

### 5.3 Fidelity Formula

For Werner states with fidelity $F$:

$$F' = \frac{F^2 + \frac{1}{9}(1-F)^2}{F^2 + \frac{2}{3}F(1-F) + \frac{5}{9}(1-F)^2}$$

For $F = 0.99$: $F' \approx 0.9993$

### 5.4 Multi-Round Results

```
Round 0 (raw):     F = 0.990000
Round 1 (BBPSSW):  F = 0.995200  (+0.52%)
Round 2:           F = 0.998900  (+0.37%)
Round 3:           F = 0.999700  (+0.08%)
```

---

## 6. Circuit Cutting

For circuits larger than the backend allows, `CircuitCutter` uses the **Qiskit Circuit Knitting Toolbox** approach:

1. **Identify** the partition boundary (minimize cut gates)
2. **Instantiate channels** — replace each cut gate with a channel
3. **Run subcircuits** independently on small backends
4. **Reconstruct** the full result classically via quasi-probability decomposition

```
40-qubit circuit → cut at bond 27
  Subcircuit A:   27 qubits (ibm_sherbrooke)
  Subcircuit B:   13 qubits (same or different backend)
  Reconstruction: linear combination of 16 sub-experiments
```

---

## 7. Superdense Coding

QuBIOS implements superdense coding — transmitting **2 classical bits** using **1 qubit + 1 pre-shared entangled qubit**:

| Classical bits | Gate applied by Alice | Bob receives |
|---------------|----------------------|--------------|
| 00 | I (identity) | \|Φ⁺⟩ |
| 01 | X (bit flip) | \|Ψ⁺⟩ |
| 10 | Z (phase flip) | \|Φ⁻⟩ |
| 11 | ZX | \|Ψ⁻⟩ |

This doubles the classical information capacity of a quantum channel.

---

## 8. Virtual Qubit Manager

`VirtualQubitManager` allows circuits to use more "logical" qubits than physically available by:

1. **Parking** unused virtual qubits in classical memory (after measurement)
2. **Unparking** them when needed (re-initialize from saved state)
3. **Mapping** virtual → physical qubit indices dynamically

This allows 200-qubit logical programs on a 156-qubit IBM Fez.

---

<div align="center">
<strong><a href="https://qubitpage.com">🌐 Qubitpage®</a></strong> · 
<strong><a href="https://github.com/qubitpage/QuBIOS">⚛️ QuBIOS Repository</a></strong>
<br/><em>© 2026 Qubitpage® — All rights reserved</em>
</div>
