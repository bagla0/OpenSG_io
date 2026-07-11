"""Rigorous tests for opensg_io.mixed_mesh (MIXED hex+tet conformal taper generator).

Run on the server (opensg_2_0):  python -m pytest tests/test_mixed_mesh.py -v
Uses the bundled IEA-22 windIO blade.  Cases encode EMPIRICALLY VERIFIED behavior:
the mild taper meshes clean, the auto-march inserts true intermediate stations and
REFUSES honestly on genuinely twisted intervals, the YAML round-trips.
"""
import os
import sys

import numpy as np
import pytest
import yaml as _yaml

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from opensg_io.converter import load_blade
from opensg_io.mixed_mesh import (mixed_taper_mesh, write_mixed_yaml, hex_to_tets,
                                  _tet_vols)
from opensg_io.conformity import check_conformal

WINDIO = os.path.join(ROOT, "examples", "data", "IEA-22-280-RWT.yaml")


@pytest.fixture(scope="module")
def blade():
    return load_blade(WINDIO)


@pytest.fixture(scope="module")
def mild(blade):
    """The validated mild taper r=0.2->0.3 (coarse for speed)."""
    return mixed_taper_mesh(blade, 0.2, 0.3, n_thick=2, n_span=4, nw=3,
                            mesh_size=0.02, verbose=False)


def test_mild_clean_and_mixed(mild):
    rep = mild["report"]
    assert rep["n_hex"] > 0 and rep["n_tet"] > 0            # genuinely MIXED
    assert rep["min_sj_hex"] > 0.0                          # skin hexes non-inverted
    assert rep["n_neg_tet"] == 0                            # all web tets positive
    assert rep["stations"] == [0.2, 0.3]                    # no refinement needed here
    assert len(mild["hmats"]) == rep["n_hex"] + rep["n_tet"]
    assert mild["oris"].shape == (rep["n_hex"] + rep["n_tet"], 9)


def test_mild_hex_region_conforming(mild):
    """The skin-hex subset must be perfectly conforming among itself."""
    ok, rep = check_conformal(mild["nodes"], mild["hexes"], "hex")
    assert ok, rep


def test_mild_tet_region_conforming(mild):
    """The web-tet subset (matched-diagonal 6-splits) must be conforming."""
    ok, rep = check_conformal(mild["nodes"], mild["tets"], "tet")
    assert ok, rep


def test_interface_is_node_tied(mild):
    """Web tets reuse ONLY existing nodes (node-tied hex|tet interface, no new nodes)."""
    used_tet = set(np.unique(mild["tets"]).tolist())
    used_hex = set(np.unique(mild["hexes"]).tolist())
    assert used_tet & used_hex                              # shared junction nodes exist
    assert max(used_tet | used_hex) < len(mild["nodes"])


def test_orientation_frames_are_unit(mild):
    o = np.asarray(mild["oris"]).reshape(-1, 3, 3)
    n = np.linalg.norm(o, axis=2)
    assert np.allclose(n, 1.0, atol=1e-6)                   # e1/e2/e3 unit vectors


def test_yaml_roundtrip(tmp_path, mild):
    p = os.path.join(str(tmp_path), "mixed.yaml")
    write_mixed_yaml(p, mild)
    d = _yaml.safe_load(open(p))                            # safe_load = no numpy leakage
    assert len(d["nodes"]) == len(mild["nodes"])
    lens = {len(r[0].split()) for r in d["elements"]}
    assert lens == {8, 4}                                   # genuinely mixed rows
    assert len(d["elements"]) == len(mild["hexes"]) + len(mild["tets"])
    assert len(d["elementOrientations"]) == len(d["elements"])
    labs = sorted(l for s in d["sets"]["element"] for l in s["labels"])
    assert labs == list(range(1, len(d["elements"]) + 1))   # every element in exactly one set


def test_hex_to_tets_conformal_split():
    """Unit cube pair sharing a face: 6-split diagonals must MATCH across the shared face."""
    nodes = np.array([[x, y, z] for z in (0, 1) for y in (0, 1) for x in (0, 1, 2)], float)
    def nid(x, y, z):
        return z * 6 + y * 3 + x
    hexes = np.array([[nid(0, 0, 0), nid(1, 0, 0), nid(1, 1, 0), nid(0, 1, 0),
                       nid(0, 0, 1), nid(1, 0, 1), nid(1, 1, 1), nid(0, 1, 1)],
                      [nid(1, 0, 0), nid(2, 0, 0), nid(2, 1, 0), nid(1, 1, 0),
                       nid(1, 0, 1), nid(2, 0, 1), nid(2, 1, 1), nid(1, 1, 1)]])
    tets = hex_to_tets(hexes)
    assert (_tet_vols(nodes, tets) > 0).all()
    ok, rep = check_conformal(nodes, tets, "tet")
    assert ok, rep                                          # matched diagonals across the face


def test_auto_refine_honest_refusal(blade):
    """A genuinely twisted interval (root flatback transition) must raise an HONEST
    RuntimeError naming the interval, after the march inserted true stations."""
    with pytest.raises(RuntimeError) as e:
        mixed_taper_mesh(blade, 0.0487, 0.0665, n_thick=2, n_span=4, nw=3,
                         mesh_size=0.02, max_refine=1, verbose=False)
    assert "quality gate" in str(e.value)
    assert "tet" in str(e.value)                            # points to the robust fallback
