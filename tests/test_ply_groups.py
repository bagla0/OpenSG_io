"""Tests for TaperLaminate ply-group layer conformity (the sandwich-skin fix)."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from opensg_io.layup import TaperLaminate

SAND = [("gelcoat", 0.5e-3, 0.0), ("glass_triax", 3e-3, 0.0),
        ("medium_density_foam", 70e-3, 0.0), ("glass_triax", 3e-3, 0.0)]


def test_sandwich_nr4_layers_are_plies():
    tl = TaperLaminate.from_stations(SAND, SAND)
    cuts = tl.group_cuts(4)
    mats = [tl.group_material(cuts, l, 0.5)[0] for l in range(4)]
    assert mats == ["gelcoat", "glass_triax", "medium_density_foam", "glass_triax"]
    fr = tl.group_fractions(cuts, 0.5)
    assert len(fr) == 5 and fr[0] == 0.0 and abs(fr[-1] - 1.0) < 1e-12
    # skins survive: the interfaces sit at the ply boundaries, not at l/nr
    assert abs(fr[1] - 0.5e-3 / 76.5e-3) < 1e-9
    assert abs(fr[3] - 73.5e-3 / 76.5e-3) < 1e-9


def test_sandwich_nr3_merges_thin_pair_not_skin():
    tl = TaperLaminate.from_stations(SAND, SAND)
    cuts = tl.group_cuts(3)
    mats = [tl.group_material(cuts, l, 0.5)[0] for l in range(3)]
    # gelcoat merges into the outer triax (smallest adjacent pair); skins survive
    assert mats == ["glass_triax", "medium_density_foam", "glass_triax"]


def test_single_ply_nr4_splits_internally():
    tl = TaperLaminate.from_stations([("uni", 0.01, 0.0)], [("uni", 0.01, 0.0)])
    cuts = tl.group_cuts(4)
    fr = tl.group_fractions(cuts, 0.5)
    assert len(fr) == 5
    assert all(tl.group_material(cuts, l, 0.5)[0] == "uni" for l in range(4))
    # internal splits give near-uniform layers
    w = [fr[l + 1] - fr[l] for l in range(4)]
    assert max(w) < 0.55 and min(w) > 0.1


def test_fractions_move_with_taper():
    thin = [("skin", 2e-3, 0.0), ("core", 20e-3, 0.0), ("skin", 2e-3, 0.0)]
    thick = [("skin", 4e-3, 0.0), ("core", 60e-3, 0.0), ("skin", 4e-3, 0.0)]
    tl = TaperLaminate.from_stations(thick, thin)
    cuts = tl.group_cuts(3)
    f0, f1 = tl.group_fractions(cuts, 0.0), tl.group_fractions(cuts, 1.0)
    assert abs(f0[1] - 4.0 / 68.0) < 1e-9                  # thick end: skin frac
    assert abs(f1[1] - 2.0 / 24.0) < 1e-9                  # thin end: skin frac


if __name__ == "__main__":
    for fn in [test_sandwich_nr4_layers_are_plies, test_sandwich_nr3_merges_thin_pair_not_skin,
               test_single_ply_nr4_splits_internally, test_fractions_move_with_taper]:
        fn(); print("PASS", fn.__name__)
