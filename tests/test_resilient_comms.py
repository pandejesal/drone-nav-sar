#!/usr/bin/env python3
"""Sprint 23: Resilient comms tests (SAR-only). 4 tests."""

import numpy as np

from src.comms.anti_jam import AntiJamLink
from src.comms.dtn_mesh import DTNNode
from src.comms.lpi_lpd import LpiLpdLink, dsss_despread, dsss_spread
from src.comms.mesh_c2 import ResilientMeshC2


def test_dtn_delivery():
    """DTN delivery > 99% at 30% loss, 50% availability (custody+retry)."""
    a = DTNNode("a", link_availability=0.5, loss=0.3, seed=1)
    b = DTNNode("b", link_availability=0.5, loss=0.3, seed=2)
    n = 100
    ok = 0
    for i in range(n):
        bd = a.create_bundle("b", {"skill": "navigate_to", "wp": i})
        # repeated epidemic rounds emulate intermittent contacts
        for _ in range(12):
            if a.custody_transfer(bd.bundle_id, b, max_retries=8):
                break
            b.epidemic_exchange(a)
        ok += int(bd.bundle_id in b.delivered)
    ratio = ok / n
    assert ratio > 0.99, f"dtn delivery {ratio}"
    # expiration path: TTL=0 bundle must expire
    e = a.create_bundle("b", {"skill": "hover"}, lifetime_s=0.0)
    import time

    assert e.is_expired(time.time() + 1.0)
    assert a.expire_bundles(now=time.time() + 10.0) >= 1


def test_lpi_lpd_detection():
    """LPI/LPD: interceptor power < -100 dBm at 1 km; DSSS roundtrip."""
    link = LpiLpdLink(tx_power_dbm=0.0)
    link.adapt_power(50.0)  # short friendly link -> power drops
    p1km = link.interceptor_power_dbm(1000.0)
    assert p1km < -100.0, f"lpi {p1km} dBm"
    assert not link.is_detectable(1000.0)
    bits = np.array([1.0, -1.0, 1.0, 1.0])
    rx = dsss_despread(dsss_spread(bits))
    assert list(rx) == list(bits)
    assert len(link.hop_sequence(75)) == 75


def test_anti_jam():
    """Anti-jam: margin > 30 dB; link maintained at 30 dB J/S."""
    aj = AntiJamLink(seed=7)
    assert aj.jamming_margin_db() > 30.0, f"margin {aj.jamming_margin_db()}"
    assert aj.link_maintained_at_js(js_db=30.0)
    spec = np.ones(75)
    spec[10] = 50.0
    det = aj.detect_jammer(spec)
    assert det["jammed"] and 10 in det["jammer_bins"]


def test_mesh_resilience():
    """10-drone mesh survives 50% node loss (isolate 5 of 10)."""
    mesh = ResilientMeshC2(node_id=0, n_nodes=10, comm_range_m=200.0,
                           packet_loss=0.0, seed=3)
    rng = np.random.default_rng(11)
    pos = [rng.uniform(0, 60, 3).astype(np.float32) for _ in range(10)]
    info = mesh.form_mesh(pos)
    assert info["formed"]
    # Isolate nodes 5..9: cut all their links (50% node loss).
    for n in range(5, 10):
        for m in list(mesh.adj.get(n, set())):
            mesh.remove_link(n, m)
    mesh.heal()
    # Survivor sub-mesh (0..4) must stay connected and deliver.
    sub = {i: mesh.adj[i] & set(range(5)) for i in range(5)}
    seen, stack = {0}, [0]
    while stack:
        u = stack.pop()
        for v in sub[u]:
            if v not in seen:
                seen.add(v)
                stack.append(v)
    assert len(seen) == 5, f"survivors connected: {seen}"
    b = mesh.resilient_send(4, {"skill": "hover"}, src=0)
    for _ in range(5):
        mesh.dtn_tick()
    assert b.delivered, "DTN delivery within surviving mesh"
    rep = mesh.resilience_report()
    assert rep["lpi_1km_dbm"] < -100.0
    assert rep["aj_margin_db"] > 30.0
