"""QuBIOS — Qubitpage® Quantum BIOS Framework.

https://qubitpage.com | https://github.com/qubitpage/QuBIOS

Example:
    from qubios import QubiLogicEngine, SteaneQEC, TransitRing

    engine = QubiLogicEngine(n_escorts=2, n_transit=3, error_rate=0.001)
    result = engine.benchmark_comparison(circuit_type="bell", shots=4096)

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

__version__ = "1.0.0"
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
]
