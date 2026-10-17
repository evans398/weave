"""
Tests for the weave model. Run with:  python tests/test_weave.py   (or pytest, if installed)

Properties checked:
  - every cut (repeat or hand-off) lands on a zero crossing, in the right direction
  - every play of a loop equals the source stream from the segment start (up to polarity)
  - parameter changes made between crossings are applied only at the next crossing
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "model"))
from weave_model import run_v2, join_ok, chains, check_chain, sr  # noqa: E402

T = np.arange(3 * sr) / sr
A = 0.6 * np.sin(2 * np.pi * 440 * T)            # 20 crossings = 1002 samples
B = 0.6 * np.sin(2 * np.pi * 330 * T + 1)


def _chain(ev, start, cur=0):
    return [c for c in chains(ev) if c["start"] == start and c["cur"] == cur][0]


def _baseline(reps=30, z=20):
    out, ev = run_v2(A, B, zA=z, zB=2, repA=reps, repB=0)
    cs = [c for c in chains(ev) if c["cur"] == 0 and len(c["cuts"]) == reps + 2][0]
    return cs


def _change_mid_play(cs, k):
    """a sample in the middle of play k (not on a crossing)"""
    return (cs["cuts"][k] + cs["cuts"][k + 1]) // 2


def _run_chain(changes, reps=30, z=20):
    cs = _baseline(reps, z)
    out, ev = run_v2(A, B, zA=z, zB=2, repA=reps, repB=0, changes=changes(cs))
    c = _chain(ev, cs["start"])
    return out, ev, c, np.diff(c["cuts"])


def test_constant_params_cut_on_crossings():
    out, ev = run_v2(A, B, zA=2, zB=2, repA=3, repB=1)
    ok, tot = join_ok(out, ev)
    assert tot > 100 and ok == tot


def test_repeat_structure():
    cs = _baseline(reps=30)
    d = np.diff(cs["cuts"])
    assert len(d) == 31                      # first play + 30 repeats
    assert len(set(d)) == 1                  # identical length every time


def test_content_matches_source():
    out, ev = run_v2(A, B, zA=20, zB=2, repA=30, repB=0)
    assert all(check_chain(out, c, A, B) == 0 for c in chains(ev))


def test_decrease_mid_repeat():
    out, ev, c, d = _run_chain(lambda cs: [(_change_mid_play(cs, 5), "zA", 10)])
    L = int(np.diff(_baseline()["cuts"])[0])
    assert len(d) == 31                      # repeat count unaffected
    assert abs(int(d[-1]) - L // 2) <= 2     # later loops are 10 crossings
    assert check_chain(out, c, A, B) == 0
    ok, tot = join_ok(out, ev)
    assert ok == tot


def test_increase_mid_repeat_extends_into_following_audio():
    out, ev, c, d = _run_chain(lambda cs: [(_change_mid_play(cs, 5), "zA", 40)], reps=20)
    L = int(np.diff(_baseline(reps=20)["cuts"])[0])
    assert abs(int(d[-1]) - 2 * L) <= 3      # later loops are 40 crossings
    assert check_chain(out, c, A, B) == 0    # content is the source stream continuing
    ok, tot = join_ok(out, ev)
    assert ok == tot


def test_restore_after_decrease():
    def ch(cs):
        return [(_change_mid_play(cs, 5), "zA", 10), (_change_mid_play(cs, 8), "zA", 20)]
    out, ev, c, d = _run_chain(ch)
    L = int(np.diff(_baseline()["cuts"])[0])
    assert abs(int(d[-1]) - L) <= 2
    assert check_chain(out, c, A, B) == 0


def test_repeat_count_lowered_finishes_loop_then_hands_off():
    out, ev, c, d = _run_chain(lambda cs: [(_change_mid_play(cs, 5), "repA", 3)])
    assert len(d) == 6                       # plays 0..5, then hand-off
    out, ev, c, d = _run_chain(lambda cs: [(_change_mid_play(cs, 5), "repA", 8)])
    assert len(d) == 9                       # first play + 8 repeats
    ok, tot = join_ok(out, ev)
    assert ok == tot


def test_change_during_first_play():
    out, ev, c, d = _run_chain(lambda cs: [(cs["cuts"][0] + 300, "zA", 5)])
    L = int(np.diff(_baseline()["cuts"])[0])
    assert abs(int(d[-1]) - L // 4) <= 2
    ok, tot = join_ok(out, ev)
    assert ok == tot


def test_fuzz_random_parameter_changes():
    rng = np.random.default_rng(1)
    noise_a, noise_b = 0.4 * rng.standard_normal(len(T)), 0.4 * rng.standard_normal(len(T))
    for x, y in [(A, B), (noise_a, noise_b)]:
        changes = [(int(rng.integers(1000, len(x) - 1000)), str(rng.choice(["zA", "zB", "repA", "repB"])),
                    int(rng.integers(1, 60))) for _ in range(20)]
        out, ev = run_v2(x, y, zA=int(rng.integers(1, 30)), zB=int(rng.integers(1, 30)),
                         repA=int(rng.integers(0, 20)), repB=int(rng.integers(0, 20)), changes=changes)
        assert np.all(np.isfinite(out))
        ok, tot = join_ok(out, ev)
        assert ok == tot
        assert sum(check_chain(out, c, x, y) for c in chains(ev)) == 0


def test_silent_and_dc_second_source():
    for other in (np.zeros(len(T)), 0.3 * np.ones(len(T))):
        out, ev = run_v2(A, other, zA=10, zB=10, repA=4, repB=4)
        assert np.all(np.isfinite(out))


def test_reverse_every_second_repeat():
    out, ev = run_v2(A, B, zA=5, zB=2, repA=6, repB=0, revA=1)
    assert join_ok(out, ev)[0] == join_ok(out, ev)[1]
    c = [c for c in chains(ev) if c["cur"] == 0 and len(c["cuts"]) == 8][0]
    assert c["rvs"] == [0, 1, 0, 1, 0, 1, 0]          # play, then reversed, forward, reversed, ...
    assert check_chain(out, c, A, B) == 0


def test_reverse_off_changes_nothing():
    o1, _ = run_v2(A, B, zA=5, zB=3, repA=4, repB=4)
    o2, e2 = run_v2(A, B, zA=5, zB=3, repA=4, repB=4, revA=0, revB=0)
    assert np.array_equal(o1, o2) and all(e["rv"] == 0 for e in e2)


def test_reverse_per_source():
    out, ev = run_v2(A, B, zA=4, zB=4, repA=4, repB=4, revA=1, revB=0)
    assert any(e["rv"] for e in ev if e["cur"] == 0) and not any(e["rv"] for e in ev if e["cur"] == 1)
    out, ev = run_v2(A, B, zA=4, zB=4, repA=4, repB=4, revA=0, revB=1)
    assert any(e["rv"] for e in ev if e["cur"] == 1) and not any(e["rv"] for e in ev if e["cur"] == 0)
    assert join_ok(out, ev)[0] == join_ok(out, ev)[1]


def test_reverse_toggle_and_zx_change_mid_repeat():
    cs = _baseline(30, 20)
    for chg in ([("revA", 1)], [("zA", 7)], [("zA", 7), ("revA", 1)], [("revA", 1), ("zA", 40)]):
        k = 3
        m = (cs["cuts"][k] + cs["cuts"][k + 1]) // 2
        out, ev = run_v2(A, B, zA=20, zB=2, repA=30, repB=0,
                         changes=[(m + j * 17, n, v) for j, (n, v) in enumerate(chg)])
        assert join_ok(out, ev)[0] == join_ok(out, ev)[1]
        assert sum(check_chain(out, c, A, B) for c in chains(ev)) == 0


def test_reverse_fuzz():
    rng = np.random.default_rng(5)
    nz = [0.4 * rng.standard_normal(len(T)) for _ in range(2)]
    for x, y in [(A, B), (nz[0], nz[1])]:
        changes = [(int(rng.integers(1000, len(x) - 1000)),
                    str(rng.choice(["zA", "zB", "repA", "repB", "revA", "revB"])), int(rng.integers(0, 40)))
                   for _ in range(25)]
        out, ev = run_v2(x, y, zA=int(rng.integers(1, 30)), zB=int(rng.integers(1, 30)),
                         repA=int(rng.integers(0, 20)), repB=int(rng.integers(0, 20)),
                         revA=1, revB=1, changes=changes)
        assert np.all(np.isfinite(out))
        assert join_ok(out, ev)[0] == join_ok(out, ev)[1]
        assert sum(check_chain(out, c, x, y) for c in chains(ev)) == 0


if __name__ == "__main__":
    failed = 0
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)]
    for name, fn in tests:
        try:
            fn()
            print("PASS", name)
        except AssertionError as e:
            failed += 1
            print("FAIL", name, e)
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)
