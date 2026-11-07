"""Measured Steane [[7,1,3]] quantum-memory experiment — Qubitpage® QuBIOS.

https://qubitpage.com | https://github.com/qubitpage/QuBIOS

This module replaces assumed numbers with measured ones. It:

  1. builds a circuit-level noisy Stim circuit for a Steane-code memory
     (prepare a logical state, run R rounds of syndrome extraction, measure),
  2. declares DETECTORs and the logical OBSERVABLE so Stim can check that the
     circuit is a valid QEC circuit (noiseless detectors are deterministic),
  3. decodes every shot with a real decoder (BP+OSD on Stim's detector error
     model, via the `ldpc` package),
  4. measures the logical error rate and compares it with an unencoded
     physical qubit idling for the same number of time steps under the same
     noise model (a fair baseline).

Nothing in this file substitutes a formula for a measurement.

© 2026 Qubitpage® | MIT License
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field, asdict

import numpy as np
import stim

try:
    from tesseract_decoder import tesseract as _tesseract
    HAS_TESSERACT = True
except ImportError:  # pragma: no cover
    HAS_TESSERACT = False

try:
    from ldpc import BpOsdDecoder
    HAS_LDPC = True
except ImportError:  # pragma: no cover
    HAS_LDPC = False


# ── Steane code layout (Hamming [7,4,3]; qubit i ↔ binary position i+1) ──
DATA = list(range(7))
X_ANC = [7, 8, 9]        # measure X-type stabilizers
Z_ANC = [10, 11, 12]     # measure Z-type stabilizers
STABILIZERS = [          # same supports for X- and Z-type (CSS, self-dual)
    [3, 4, 5, 6],
    [1, 2, 5, 6],
    [0, 2, 4, 6],
]
LOGICAL_SUPPORT = DATA   # Z_L = Z^⊗7, X_L = X^⊗7


@dataclass
class NoiseModel:
    """Uniform circuit-level depolarizing noise.

    p is applied: after every 1-qubit gate (DEPOLARIZE1), after every CNOT
    (DEPOLARIZE2), to every reset and measurement (bit/phase flip), and to
    every qubit that idles during a time step (DEPOLARIZE1).
    Set `idle_factor` to scale idle noise relative to gate noise.
    """
    p: float = 1e-3
    idle_factor: float = 1.0

    @property
    def p_idle(self) -> float:
        return self.p * self.idle_factor


def steane_encoder(basis: str = "Z") -> stim.Circuit:
    """Noiseless unitary encoder for |0_L> (basis 'Z') or |+_L> (basis 'X').

    |0_L> = uniform superposition over the X-stabilizer group applied to
    |0000000>: Hadamard the three 'pivot' qubits (positions 1,2,4 → q0,q1,q3)
    and fan out with CNOTs along each stabilizer support.
    |+_L> is obtained by applying transversal H to |0_L> (the code is
    self-dual).
    """
    c = stim.Circuit()
    pivots = {0: [2, 4, 6], 1: [2, 5, 6], 3: [4, 5, 6]}
    c.append("H", list(pivots))
    for ctrl, targets in pivots.items():
        for t in targets:
            c.append("CNOT", [ctrl, t])
    if basis.upper() == "X":
        c.append("H", DATA)
    return c


def _schedule_sequences(seqs: list[list[tuple[int, int]]]) -> list[list[tuple[int, int]]]:
    """Greedy-pack CNOT sequences into layers. CNOTs inside one sequence keep
    their order; no qubit is used twice in the same layer."""
    layers: list[list[tuple[int, int]]] = []
    busy: list[set[int]] = []
    for seq in seqs:
        k_min = 0
        for a, b in seq:
            k = k_min
            while k < len(layers) and (a in busy[k] or b in busy[k]):
                k += 1
            if k == len(layers):
                layers.append([])
                busy.append(set())
            layers[k].append((a, b))
            busy[k].update((a, b))
            k_min = k + 1
    return layers


def _schedule(cnots: list[tuple[int, int]]) -> list[list[tuple[int, int]]]:
    """Group CNOTs by their ancilla (qubit 7..12) and schedule them."""
    seqs: dict[int, list[tuple[int, int]]] = {}
    for a, b in cnots:
        anc = a if 7 <= a <= 12 else b
        seqs.setdefault(anc, []).append((a, b))
    return _schedule_sequences(list(seqs.values()))


_X_CNOTS = [(X_ANC[i], d) for i, s in enumerate(STABILIZERS) for d in s]
_Z_CNOTS = [(d, Z_ANC[i]) for i, s in enumerate(STABILIZERS) for d in s]
X_LAYERS = _schedule(_X_CNOTS)
Z_LAYERS = _schedule(_Z_CNOTS)

# Flag qubits (Chao & Reichardt, PRL 121, 050502, 2018): one per stabilizer.
# A flag is coupled to its ancilla after the 1st and before the 4th data CNOT,
# so a single ancilla fault that spreads to 2 data qubits ("hook error") also
# flips the flag. Flags are deterministic and are added as detectors, so the
# global BP+OSD decoder uses them automatically.
X_FLAG = [13, 14, 15]    # |0>, measured in Z  (catch X faults on X-ancilla)
Z_FLAG = [16, 17, 18]    # |+>, measured in X  (catch Z faults on Z-ancilla)


def _flagged_sequences(kind: str) -> list[list[tuple[int, int]]]:
    seqs = []
    for i, sup in enumerate(STABILIZERS):
        if kind == "X":
            a, f = X_ANC[i], X_FLAG[i]
            data = [(a, d) for d in sup]
            flag = (a, f)
        else:
            a, f = Z_ANC[i], Z_FLAG[i]
            data = [(d, a) for d in sup]
            flag = (f, a)
        seqs.append([data[0], flag, data[1], data[2], flag, data[3]])
    return seqs


X_LAYERS_FLAGGED = _schedule_sequences(_flagged_sequences("X"))
Z_LAYERS_FLAGGED = _schedule_sequences(_flagged_sequences("Z"))


class _Builder:
    """Appends time steps (TICK-separated) with gate noise + idle noise."""

    def __init__(self, qubits: list[int], noise: NoiseModel | None):
        self.c = stim.Circuit()
        self.qubits = qubits
        self.noise = noise
        self.n_meas = 0
        self.ticks = 0

    def _idle(self, used: set[int]):
        if self.noise and self.noise.p_idle > 0:
            idle = [q for q in self.qubits if q not in used]
            if idle:
                self.c.append("DEPOLARIZE1", idle, self.noise.p_idle)

    def step(self, gate: str, targets: list[int], *, pairs: bool = False):
        p = self.noise.p if self.noise else 0.0
        if gate in ("M", "MX"):
            self.c.append(gate, targets, p) if p > 0 else self.c.append(gate, targets)
            self.n_meas += len(targets)
        else:
            self.c.append(gate, targets)
            if p > 0:
                if gate == "R":
                    self.c.append("X_ERROR", targets, p)
                elif gate == "RX":
                    self.c.append("Z_ERROR", targets, p)
                elif pairs:
                    self.c.append("DEPOLARIZE2", targets, p)
                else:
                    self.c.append("DEPOLARIZE1", targets, p)
        self._idle(set(targets))
        self.c.append("TICK")
        self.ticks += 1

    def cnot_layer(self, layer: list[tuple[int, int]]):
        self.step("CNOT", [q for pair in layer for q in pair], pairs=True)

    def rec(self, absolute_index: int) -> stim.GateTarget:
        return stim.target_rec(absolute_index - self.n_meas)


def syndrome_round(b: _Builder, flags: bool = False):
    """One round of syndrome extraction.

    flags=False: single ancilla per stabilizer (not fault-tolerant: hook
                 errors give circuit-level distance 2).
    flags=True:  one extra flag qubit per stabilizer.
    Returns (x_meas, z_meas, flag_meas) absolute measurement indices.
    """
    if not flags:
        b.step("R", X_ANC + Z_ANC)
        b.step("H", X_ANC)
        for layer in X_LAYERS:
            b.cnot_layer(layer)
        b.step("H", X_ANC)
        for layer in Z_LAYERS:
            b.cnot_layer(layer)
        first = b.n_meas
        b.step("M", X_ANC + Z_ANC)
        return list(range(first, first + 3)), list(range(first + 3, first + 6)), []

    b.step("R", X_ANC + Z_ANC + X_FLAG + Z_FLAG)
    b.step("H", X_ANC + Z_FLAG)
    for layer in X_LAYERS_FLAGGED:
        b.cnot_layer(layer)
    b.step("H", X_ANC)
    for layer in Z_LAYERS_FLAGGED:
        b.cnot_layer(layer)
    b.step("H", Z_FLAG)
    first = b.n_meas
    b.step("M", X_ANC + Z_ANC + X_FLAG + Z_FLAG)
    return (list(range(first, first + 3)), list(range(first + 3, first + 6)),
            list(range(first + 6, first + 12)))


def _qubits(flags: bool) -> list[int]:
    return DATA + X_ANC + Z_ANC + (X_FLAG + Z_FLAG if flags else [])


def round_duration_ticks(flags: bool = False) -> int:
    """Number of time steps in one syndrome round (used for the baseline)."""
    b = _Builder(_qubits(flags), None)
    syndrome_round(b, flags)
    return b.ticks


def build_memory_circuit(rounds: int, noise: NoiseModel | None,
                         basis: str = "Z", flags: bool = False,
                         logical_one: bool = False) -> stim.Circuit:
    """Steane memory experiment with detectors and a logical observable.

    basis 'Z': prepare |0_L>, final Z measurement, protects against X errors
               (Z-type stabilizers are the error-detecting checks).
    basis 'X': prepare |+_L>, final X measurement, protects against Z errors.

    Preparation: reset data in the product state (|0>^7 or |+>^7), which is
    already a +1 eigenstate of the same-type stabilizers and of the logical
    operator; the first syndrome round then projects into the code space.
    """
    basis = basis.upper()
    assert basis in ("Z", "X")
    b = _Builder(_qubits(flags), noise)
    b.step("R" if basis == "Z" else "RX", DATA)
    if logical_one:   # |1_L> = X_L|0_L>  or  |-_L> = Z_L|+_L>  (transversal)
        b.step("X" if basis == "Z" else "Z", DATA)

    prev_x = prev_z = None
    for r in range(rounds):
        mx, mz, mf = syndrome_round(b, flags)
        for j, mi in enumerate(mf):
            b.c.append("DETECTOR", [b.rec(mi)], [j, r, 2])
        same, other = (mz, mx) if basis == "Z" else (mx, mz)
        prev_same, prev_other = (prev_z, prev_x) if basis == "Z" else (prev_x, prev_z)
        for i in range(3):
            if r == 0:   # same-type checks are deterministic from the start
                b.c.append("DETECTOR", [b.rec(same[i])], [i, r, 0])
            else:
                b.c.append("DETECTOR", [b.rec(same[i]), b.rec(prev_same[i])], [i, r, 0])
                b.c.append("DETECTOR", [b.rec(other[i]), b.rec(prev_other[i])], [i, r, 1])
        prev_x, prev_z = mx, mz

    first = b.n_meas
    b.step("M" if basis == "Z" else "MX", DATA)
    last_same = prev_z if basis == "Z" else prev_x
    for i, s in enumerate(STABILIZERS):
        targets = [b.rec(first + d) for d in s] + [b.rec(last_same[i])]
        b.c.append("DETECTOR", targets, [i, rounds, 0])
    b.c.append("OBSERVABLE_INCLUDE", [b.rec(first + d) for d in LOGICAL_SUPPORT], 0)
    return b.c


def build_physical_baseline(rounds: int, noise: NoiseModel,
                            basis: str = "Z", flags: bool = False) -> stim.Circuit:
    """One unencoded qubit idling for exactly as many time steps as the
    Steane memory, with the same reset / idle / measurement noise."""
    c = stim.Circuit()
    p = noise.p
    if basis.upper() == "Z":
        c.append("R", [0]); c.append("X_ERROR", [0], p)
    else:
        c.append("RX", [0]); c.append("Z_ERROR", [0], p)
    for _ in range(rounds * round_duration_ticks(flags)):
        c.append("DEPOLARIZE1", [0], noise.p_idle)
    c.append("M" if basis.upper() == "Z" else "MX", [0], p)
    c.append("OBSERVABLE_INCLUDE", [stim.target_rec(-1)], 0)
    return c


# ── Decoding ─────────────────────────────────────────────────────────────

class DemDecoder:
    """Most-likely-error decoder built from a Stim detector error model.

    Default backend: Tesseract (Google, 2025), an A*-search decoder made for
    hypergraph error models such as colour/Steane codes.
    Fallback: BP+OSD from `ldpc`. NOTE: on these small, highly degenerate
    Steane models BP+OSD was measured to mis-decode some *single* faults,
    so it is only used if Tesseract is unavailable.
    """

    def __init__(self, circuit: stim.Circuit, backend: str = "auto"):
        dem = circuit.detector_error_model(decompose_errors=False)
        self.dem = dem
        self.n_obs = dem.num_observables
        if backend == "auto":
            backend = "tesseract" if HAS_TESSERACT else "bposd"
        self.backend = backend
        self._cache: dict[bytes, np.ndarray] = {}
        # Build check (H) and observable (L) matrices — also used by tests.
        cols_d, cols_o, priors = [], [], []
        for inst in dem.flattened():
            if inst.type != "error":
                continue
            d, o = [], []
            for t in inst.targets_copy():
                if t.is_relative_detector_id():
                    d.append(t.val)
                elif t.is_logical_observable_id():
                    o.append(t.val)
            cols_d.append(d); cols_o.append(o); priors.append(inst.args_copy()[0])
        self.priors = np.array(priors)
        self.H = np.zeros((dem.num_detectors, len(priors)), dtype=np.uint8)
        self.L = np.zeros((self.n_obs, len(priors)), dtype=np.uint8)
        for j, (d, o) in enumerate(zip(cols_d, cols_o)):
            self.H[d, j] = 1
            self.L[o, j] = 1
        if backend == "tesseract":
            if not HAS_TESSERACT:
                raise RuntimeError("pip install tesseract-decoder")
            cfg = _tesseract.TesseractConfig(dem=dem)
            self._tess = _tesseract.TesseractDecoder(cfg)
        elif backend == "bposd":
            if not HAS_LDPC:
                raise RuntimeError("pip install ldpc")
            self._bposd = BpOsdDecoder(self.H, error_channel=list(priors), max_iter=100,
                                       bp_method="product_sum", osd_method="osd_cs",
                                       osd_order=7)
        else:
            raise ValueError(backend)

    def decode(self, syndrome: np.ndarray) -> np.ndarray:
        syndrome = np.ascontiguousarray(syndrome, dtype=np.uint8)
        if not syndrome.any():
            return np.zeros(self.n_obs, dtype=np.uint8)
        key = np.packbits(syndrome).tobytes()
        hit = self._cache.get(key)
        if hit is None:
            if self.backend == "tesseract":
                hit = np.asarray(self._tess.decode(syndrome.astype(bool)), dtype=np.uint8)
            else:
                err = self._bposd.decode(syndrome)
                hit = (self.L @ err) % 2
            self._cache[key] = hit
        return hit


def decode_failures(dec: DemDecoder, dets: np.ndarray, obs: np.ndarray) -> int:
    """Count shots where the decoder's predicted logical flip is wrong.
    Each distinct syndrome is decoded once (identical shots share a result)."""
    packed = np.packbits(dets, axis=1)
    uniq, inv = np.unique(packed, axis=0, return_inverse=True)
    inv = inv.reshape(-1)
    n_det = dets.shape[1]
    pred = np.empty(len(uniq), dtype=np.uint8)
    for i, row in enumerate(uniq):
        syn = np.unpackbits(row)[:n_det]
        pred[i] = dec.decode(syn)[0]
    return int((pred[inv] ^ obs[:, 0].astype(np.uint8)).sum())


# ── Experiment ───────────────────────────────────────────────────────────

@dataclass
class MemoryResult:
    p: float
    rounds: int
    basis: str
    shots: int
    logical_errors: int
    logical_error_rate: float
    logical_error_rate_ci95: tuple[float, float]
    shots_with_detection: int
    physical_errors: int
    physical_error_rate: float
    circuit_distance: int | None = None
    extra: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 1.0)
    ph = k / n
    den = 1 + z * z / n
    c = (ph + z * z / (2 * n)) / den
    h = z * math.sqrt(ph * (1 - ph) / n + z * z / (4 * n * n)) / den
    return (max(0.0, c - h), min(1.0, c + h))


def circuit_distance(circuit: stim.Circuit) -> int:
    """Smallest number of circuit faults that flip the logical observable
    without triggering any detector (Stim heuristic search)."""
    errs = circuit.search_for_undetectable_logical_errors(
        dont_explore_detection_event_sets_with_size_above=4,
        dont_explore_edges_with_degree_above=9999,
        dont_explore_edges_increasing_symptom_degree=False,
        canonicalize_circuit_errors=True,
    )
    return len(errs)


def run_memory(p: float = 1e-3, rounds: int = 3, shots: int = 100_000,
               basis: str = "Z", idle_factor: float = 1.0,
               seed: int | None = None, with_distance: bool = False,
               flags: bool = False) -> MemoryResult:
    """Run the Steane memory and the matched physical baseline; return
    measured logical and physical error rates."""
    noise = NoiseModel(p=p, idle_factor=idle_factor)
    circ = build_memory_circuit(rounds, noise, basis, flags)
    dec = DemDecoder(circ)
    dets, obs = circ.compile_detector_sampler(seed=seed).sample(shots, separate_observables=True)
    fails = decode_failures(dec, dets, obs)
    detected = int(dets.any(axis=1).sum())

    base = build_physical_baseline(rounds, noise, basis, flags)
    _, bobs = base.compile_detector_sampler(seed=seed).sample(shots, separate_observables=True)
    pfails = int(bobs[:, 0].sum())

    return MemoryResult(
        p=p, rounds=rounds, basis=basis, shots=shots,
        logical_errors=fails, logical_error_rate=fails / shots,
        logical_error_rate_ci95=wilson_ci(fails, shots),
        shots_with_detection=detected,
        physical_errors=pfails, physical_error_rate=pfails / shots,
        circuit_distance=circuit_distance(circ) if with_distance else None,
        extra={"flags": flags,
               "ticks_per_round": round_duration_ticks(flags),
               "num_detectors": circ.num_detectors,
               "decoder": f"{dec.backend} on Stim DEM"},
    )


def per_round_error(total_error: float, rounds: int) -> float:
    """Convert a total flip probability after R rounds into a per-round rate,
    assuming independent rounds: 1-2P = (1-2ε)^R."""
    x = 1 - 2 * total_error
    if x <= 0:
        return 0.5
    return 0.5 * (1 - x ** (1.0 / rounds))
