# Nura

Nura is an ongoing engineering effort focused on the design of a long-lived,
memory-centric AI system with strict architectural boundaries.

## Architecture Status

### Phase 1 — Foundation & Boundary Enforcement (Completed)

Phase 1 focused exclusively on enforcing strict engine boundaries and isolating
temporal responsibility across the system.

Completed work includes:
- Removal of temporal reasoning from MemoryEngine
- Centralized engine instantiation and lifecycle control
- Isolation of temporal parsing logic within TemporalEngine
- Elimination of cross-engine dependency violations

Phase 1 was completed with no new features added and no intended runtime behavior
changes, establishing a stable architectural foundation for subsequent phases.

> Phase 1 completion timestamp: **December 20th 2025**
