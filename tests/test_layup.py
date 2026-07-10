"""Tests for opensg_io.layup: alignment, linear interpolation, and ply drops."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from opensg_io.layup import TaperLaminate


def test_same_stack_linear_interp():
    lamL = [("triax", 4.0, 0.0), ("uniax", 20.0, 0.0), ("triax", 4.0, 0.0)]
    lamR = [("triax", 2.0, 0.0), ("uniax", 8.0, 0.0), ("triax", 2.0, 0.0)]
    tl = TaperLaminate.from_stations(lamL, lamR)
    assert len(tl.plies) == 3 and not tl.dropped()
    assert abs(tl.thickness(0.0) - 28.0) < 1e-9
    assert abs(tl.thickness(1.0) - 12.0) < 1e-9
    mid = tl.at(0.5)                                        # (28+12)/2 = 20 total
    assert abs(sum(t for _m, t, _a in mid) - 20.0) < 1e-9
    assert abs(mid[1][1] - 14.0) < 1e-9                    # uniax: (20+8)/2


def test_ply_drop_ramps_to_zero():
    # a carbon uniax reinforcement present at the root, dropped by the tip
    lamL = [("triax", 4.0, 0.0), ("carbon", 30.0, 0.0), ("triax", 4.0, 0.0)]
    lamR = [("triax", 4.0, 0.0), ("triax", 4.0, 0.0)]
    tl = TaperLaminate.from_stations(lamL, lamR)
    drops = tl.dropped()
    assert len(drops) == 1 and drops[0].material == "carbon"
    assert abs(tl.thickness(0.0) - 38.0) < 1e-9
    assert abs(tl.thickness(1.0) - 8.0) < 1e-9
    # carbon present near the root, gone at the tip
    assert any(m == "carbon" for m, _t, _a in tl.at(0.1))
    assert all(m != "carbon" for m, _t, _a in tl.at(1.0))
    # halfway the carbon is 15 mm
    car = [t for m, t, _a in tl.at(0.5) if m == "carbon"]
    assert car and abs(car[0] - 15.0) < 1e-9


def test_ply_of_depth_tracks_stack():
    lamL = [("gel", 0.5, 0.0), ("triax", 4.0, 45.0), ("foam", 20.0, 0.0), ("triax", 4.0, 45.0)]
    tl = TaperLaminate.from_stations(lamL, lamL)
    assert tl.ply_of_depth(0.0, 0.01)[0] == "gel"          # outer skin
    assert tl.ply_of_depth(0.0, 0.5)[0] == "foam"          # core
    assert tl.ply_of_depth(0.0, 0.99)[0] == "triax"        # inner skin
    assert tl.ply_of_depth(0.0, 0.1)[1] == 45.0            # angle carried through (in the triax)


def test_added_ply_toward_tip():
    lamL = [("triax", 4.0, 0.0), ("triax", 4.0, 0.0)]
    lamR = [("triax", 4.0, 0.0), ("uniax", 10.0, 0.0), ("triax", 4.0, 0.0)]
    tl = TaperLaminate.from_stations(lamL, lamR)
    assert len(tl.plies) == 3 and len(tl.dropped()) == 1
    assert all(m != "uniax" for m, _t, _a in tl.at(0.0))
    assert any(m == "uniax" for m, _t, _a in tl.at(1.0))


if __name__ == "__main__":
    for fn in [test_same_stack_linear_interp, test_ply_drop_ramps_to_zero,
               test_ply_of_depth_tracks_stack, test_added_ply_toward_tip]:
        fn(); print("PASS", fn.__name__)
