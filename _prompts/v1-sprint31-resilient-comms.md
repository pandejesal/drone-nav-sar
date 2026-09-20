# Sprint 31: Resilient Comms (CETC-like DTN Mesh)

## Goal
Build a resilient, delay-tolerant mesh communication layer for swarm operations in contested/denied environments. Rivals CETC/PLA swarm comms.

## Scope (modify ONLY these)
- src/comms/dtn_mesh.py: NEW — DTN Bundle Protocol (RFC 5050) implementation, custody transfer, expiration, fragmentation
- src/comms/lpi_lpd.py: EXTEND — LPI/LPD waveform simulation, frequency hopping, spread spectrum
- src/comms/anti_jam.py: EXTEND — Anti-jam techniques: FHSS, DSSS, adaptive nulling
- src/comms/mesh_c2.py: EXTEND — Integrate DTN + LPI/LPD + anti-jam into MeshC2
- tests/test_resilient_comms.py: EXTEND — 4 tests (DTN delivery, LPI/LPD detection, anti-jam, mesh resilience)

## Architecture
```
Application (Mission/C2)
    ↓
DTN Bundle Protocol (RFC 5050) — custody transfer, expiration, fragmentation
    ↓
LPI/LPD Link Layer — FHSS/DSSS, adaptive power, low probability of intercept
    ↓
Anti-Jam Layer — FHSS hopping, DSSS spreading, adaptive nulling
    ↓
Physical — Simulated RF (path loss, fading, jamming)
```

## DTN Bundle Protocol (RFC 5050)
| Feature | Implementation |
|---------|----------------|
| Bundle format | Primary block + payload block + optional blocks |
| Custody transfer | ACK/NACK with retransmission |
| Expiration | TTL-based expiration, fragmentation |
| Routing | Epidemic/Prophet/PRoPHET for intermittent connectivity |
| Security | Bundle Authentication Block (BAB), Payload Integrity Block (PIB) |

## LPI/LPD Waveform
| Parameter | Value |
|-----------|-------|
| FHSS | 75 channels, 100 hops/sec, 1 MHz spacing |
| DSSS | BPSK, 11-chip Barker, 11 Mcps |
| Power control | Adaptive: -10 to +20 dBm |
| LPI threshold | < -100 dBm at 1km |

## Anti-Jam
| Technique | Implementation |
|-----------|----------------|
| FHSS | 75-channel hopping, 100 hops/sec |
| DSSS | 11-chip Barker, 11 Mcps |
| Adaptive nulling | 4-element array, LMS algorithm |
| CJAM detection | Energy detection + cyclostationary |

## Acceptance
- DTN bundle delivery: > 99% at 30% packet loss, 50% link availability
- LPI/LPD: < -100 dBm detectability at 1km
- Anti-jam: Maintain link at 30 dB J/S
- Mesh resilience: 10-drone mesh survives 50% node loss

## DONE
DONE-31 | files: src/comms/dtn_mesh.py,src/comms/lpi_lpd.py,src/comms/anti_jam.py,src/comms/mesh_c2.py,tests/test_resilient_comms.py | tests: 4/4 green | metrics: {dtn_delivery: >0.99, lpi_dbm: <-100, jam_margin_db: >30}