"""QuBIOS — Qubitpage® Quantum BIOS Framework.

https://qubitpage.com | https://github.com/qubitpage/QuBIOS

Example (measured Steane memory vs. unencoded qubit):
    from qubios import run_memory

    r = run_memory(p=1e-3, rounds=3, shots=100_000, flags=True)
    print(r.logical_error_rate, r.physical_error_rate)

© 2026 Qubitpage® | MIT License
"""
from qubios.core import (
    QubiLogicEngine,
    VirtualCircuitEngine,
    SuperdenseEngine,
    EntanglementDistiller,
    VirtualQubitManager,
    CircuitCutter,
    TransitRing,
    SteaneQEC,
    TeleportEngine,
    EscortQubVirt,
    SteaneBlock,
    PauliFrame,
    QECStats,
    TeleportResult,
    RelayStats,
    BlockState,
)
from qubios.memory import (
    run_memory,
    build_memory_circuit,
    build_physical_baseline,
    steane_encoder,
    circuit_distance,
    per_round_error,
    DemDecoder,
    NoiseModel,
    MemoryResult,
)

__version__ = "1.2.2"
__author__ = "Qubitpage®"
__url__ = "https://qubitpage.com"

__all__ = [
    "QubiLogicEngine",
    "VirtualCircuitEngine",
    "SuperdenseEngine",
    "EntanglementDistiller",
    "VirtualQubitManager",
    "CircuitCutter",
    "TransitRing",
    "SteaneQEC",
    "TeleportEngine",
    "EscortQubVirt",
    "SteaneBlock",
    "PauliFrame",
    "QECStats",
    "TeleportResult",
    "RelayStats",
    "BlockState",
    "run_memory",
    "build_memory_circuit",
    "build_physical_baseline",
    "steane_encoder",
    "circuit_distance",
    "per_round_error",
    "DemDecoder",
    "NoiseModel",
    "MemoryResult",
]
