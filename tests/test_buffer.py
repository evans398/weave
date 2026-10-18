"""
Tests for the buffer-version model.  Run with:  python tests/test_buffer.py

Properties checked (per channel):
  - the range start is a crossing, the range end is a crossing travelling the same way
  - every join between chunks has the output travelling through zero (no gap, no skipped sample)
  - each source advances contiguously: its chunks, in order, are the range with nothing skipped
  - every chunk is the source samples times its polarity
  - every non-final chunk spans exactly zero_x crossings (zero_x - 1 strictly inside it)
  - one-shot appends what is left of the longer range; loop wraps; play off stops at a crossing
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "model"))
from buffer_model import render, snap, crossings, sr  # noqa: E402

T = np.arange(2 * sr) / sr
A = 0.6 * np.sin(2 * np.pi * 220 * T)
B = 0.5 * np.sin(2 * np.pi * 330 * T + 1) + 0.2 * np.sin(2 * np.pi * 700 * T)
SEL_A = (3000, 30000)
SEL_B = (4000, 20000)


def _ch(x, c):
    return x if x.ndim == 1 else x[:, min(c, x.shape[1] - 1)]


def check_channel(out, info, xa, xb, zx, loop):
    """the structural checks for one channel; zx = (zA, zB) constant"""
    src = [xa, xb]
    rg = [info["ra"], info["rb"]]
    chunks = info["chunks"]
    nxt = [0, 0]
    for ci, c in enumerate(chunks):
        s = c["src"]
        s0, e0 = rg[s]
        L = e0 - s0
        # contiguous advance
        assert c["k0"] == nxt[s], ("gap in source", s, c["k0"], nxt[s])
        nxt[s] = c["k1"]
        # samples are the source times the polarity
        idx = s0 + (np.arange(c["k0"], c["k1"]) % L if loop else np.arange(c["k0"], c["k1"]))
        ref = c["pol"] * src[s][idx]
        assert np.allclose(out[c["n0"]:c["n1"]], ref, atol=1e-12), ("chunk content", ci)
        # crossings inside the chunk
        inside = int(np.sum((ref[:-1] < 0) != (ref[1:] < 0))) if len(ref) > 1 else 0
        last_of_src = ci == len(chunks) - 1 or all(d["src"] != s for d in chunks[ci + 1:])
        # a chunk starts just after a crossing and ends at the zero_x-th crossing after it,
        # so zero_x - 1 crossings sit strictly inside it
        if ci < len(chunks) - 1 and not last_of_src:
            assert inside == zx[s] - 1, ("crossings in chunk", ci, inside, zx[s])
        elif ci < len(chunks) - 1:
            assert inside <= zx[s] - 1                      # a source's final chunk before the other one's last
        # the very last chunk may be the appended tail of the longer range: any number of crossings
        # join: the output keeps travelling through zero
        if ci > 0:
            n = c["n0"]
            p, q = out[n - 1], out[n]
            if c["nld"] == 1:
                assert p <= 0 and q >= 0, ("join", ci, p, q)
            else:
                assert p >= 0 and q <= 0, ("join", ci, p, q)


def test_snap():
    s0, e0 = snap(A, 3000, 30000)
    cr = crossings(A)
    assert s0 in cr and e0 in cr
    assert np.argmin(np.abs(cr - 3000)) == np.searchsorted(cr, s0)
    assert (A[s0] >= 0) == (A[e0] >= 0)                    # same direction
    assert e0 > s0
    # a start between two crossings goes to the nearer one, a tie goes to the earlier one
    i, j = cr[10], cr[11]
    assert snap(A, (i + j) // 2, 40000)[0] in (i, j)
    assert snap(A, i, 40000)[0] == i
    assert snap(np.zeros(100), 10, 50) is None             # no crossings


def test_one_shot_mono():
    out, info = render(A, B, SEL_A, SEL_B, zA=2, zB=2)
    inf = info[0]
    LA = inf["ra"][1] - inf["ra"][0]
    LB = inf["rb"][1] - inf["rb"][0]
    assert inf["length"] == LA + LB                        # nothing skipped, nothing added
    assert np.all(out[inf["length"]:] == 0)
    check_channel(out, inf, A, B, (2, 2), False)
    # concatenated A chunks are the A range, in order (up to polarity)
    for s, x in ((0, A), (1, B)):
        s0, e0 = inf["ra"] if s == 0 else inf["rb"]
        got = np.concatenate([np.abs(out[c["n0"]:c["n1"]]) for c in inf["chunks"] if c["src"] == s])
        assert np.allclose(got, np.abs(x[s0:e0]))


def test_odd_and_mixed_counts():
    for zA, zB in [(1, 1), (3, 3), (1, 4), (5, 2), (7, 3), (20, 20)]:
        out, info = render(A, B, SEL_A, SEL_B, zA=zA, zB=zB)
        check_channel(out, info[0], A, B, (zA, zB), False)
        inf = info[0]
        assert inf["length"] == (inf["ra"][1] - inf["ra"][0]) + (inf["rb"][1] - inf["rb"][0])


def test_self_interleave_doubles_length():
    out, info = render(A, A, SEL_A, SEL_A, zA=2, zB=2)
    inf = info[0]
    L = inf["ra"][1] - inf["ra"][0]
    assert inf["length"] == 2 * L
    check_channel(out, inf, A, A, (2, 2), False)


def test_append_the_longer_range():
    out, info = render(A, B, (3000, 60000), (4000, 12000), zA=2, zB=2)
    inf = info[0]
    last_b = max(i for i, c in enumerate(inf["chunks"]) if c["src"] == 1)
    assert last_b < len(inf["chunks"]) - 1                  # A chunks follow B's last chunk
    assert all(c["src"] == 0 for c in inf["chunks"][last_b + 1:])
    check_channel(out, inf, A, B, (2, 2), False)
    assert inf["length"] == (inf["ra"][1] - inf["ra"][0]) + (inf["rb"][1] - inf["rb"][0])


def test_loop_wraps_seamlessly():
    n = 60000
    out, info = render(A, B, (3000, 12000), (4000, 9000), zA=2, zB=3, loop=True, n_out=n)
    inf = info[0]
    assert inf["length"] == n
    check_channel(out, inf, A, B, (2, 3), True)
    for s0, e0 in (inf["ra"], inf["rb"]):                   # the wrap travels the same way as the start
        x = A if (s0, e0) == inf["ra"] else B
        assert (x[s0] >= 0) == (x[e0] >= 0)
    # a source played more than one lap
    assert max(c["k1"] for c in inf["chunks"] if c["src"] == 1) > inf["rb"][1] - inf["rb"][0]


def test_stop_waits_for_a_crossing():
    for loop in (True, False):
        for off in (7000, 12345, 20001):
            out, info = render(A, B, (3000, 40000), (4000, 30000), zA=4, zB=4, loop=loop,
                               n_out=50000, off_at=off)
            inf = info[0]
            m = inf["length"]
            assert off <= m < off + 1000, (loop, off, m)
            assert np.all(out[m:] == 0)
            c = inf["chunks"][-1]
            x = A if c["src"] == 0 else B
            s0, e0 = inf["ra"] if c["src"] == 0 else inf["rb"]
            L = e0 - s0
            kk = c["k1"]
            before = x[s0 + (kk - 1) % L]
            after = x[s0 + kk % L] if loop else x[min(s0 + kk, len(x) - 1)]
            assert (before < 0) != (after < 0)               # the stop is exactly on a crossing


def test_stereo_and_mono_channels():
    st = np.stack([A, 0.7 * np.roll(A, 37)], axis=1)
    out, info = render(st, B, SEL_A, SEL_B)                  # stereo A, mono B
    assert out.shape[1] == 2 and len(info) == 2
    for c in range(2):
        o1, i1 = render(st[:, c], B, SEL_A, SEL_B)
        assert np.array_equal(out[:len(o1), c], o1)
    out, info = render(A, B, SEL_A, SEL_B)
    assert out.ndim == 1                                     # mono + mono stays mono
    out, info = render(A, np.stack([B, B], axis=1), SEL_A, SEL_B)
    assert out.shape[1] == 2                                 # mono A feeds both channels
    assert np.array_equal(out[:, 0][:5000], out[:, 1][:5000]) # identical channels give identical output
    # fully stereo
    stb = np.stack([B, 0.6 * np.roll(B, 91)], axis=1)
    out, info = render(st, stb, SEL_A, SEL_B)
    for c in range(2):
        check_channel(out[:, c], info[c], st[:, c], stb[:, c], (2, 2), False)


def test_source_without_crossings():
    out, info = render(np.full(len(A), 0.3), B, SEL_A, SEL_B)
    inf = info[0]
    assert inf["ra"] is None and inf["rb"] is not None
    s0, e0 = inf["rb"]
    assert np.allclose(np.abs(out[:e0 - s0]), np.abs(B[s0:e0]))
    out, info = render(np.zeros(1000), np.zeros(1000), (0, 10), (0, 10))
    assert np.all(out == 0)


def test_zero_x_change_only_at_crossings():
    changes = [(9001, "zA", 6), (15555, "zB", 1)]
    out, info = render(A, B, SEL_A, SEL_B, zA=2, zB=3, changes=changes)
    inf = info[0]
    src = [A, B]
    full = [c for c in inf["chunks"][:-1]]
    seen = {0: False, 1: False}
    for c in full:
        s0, e0 = inf["ra"] if c["src"] == 0 else inf["rb"]
        ref = src[c["src"]][s0 + c["k0"]: s0 + c["k1"]]
        inside = int(np.sum((ref[:-1] < 0) != (ref[1:] < 0)))
        if c["src"] == 0 and c["n0"] > 9001 and c["k1"] - c["k0"] > 0 and c is not full[-1]:
            assert inside <= 5                                # a chunk that starts after the change uses the new count
            seen[0] = seen[0] or inside == 5
        if c["src"] == 1 and c["n0"] > 15555 and c is not full[-1]:
            assert inside == 0
            seen[1] = True
        if c["n0"] < 9001 and c["src"] == 0 and c["n1"] < 9001:
            assert inside == 1                                # before the change: the old count
    assert seen[0] and seen[1]
    for ci, c in enumerate(inf["chunks"]):                  # joins stay clean
        if ci:
            p, q = out[c["n0"] - 1], out[c["n0"]]
            assert (p <= 0 and q >= 0) if c["nld"] == 1 else (p >= 0 and q <= 0)


def test_fuzz():
    rng = np.random.default_rng(3)
    for t in range(40):
        n = int(rng.integers(8000, 40000))
        a = 0.4 * rng.standard_normal(n) if t % 2 else A[:n]
        b = 0.4 * rng.standard_normal(n) if t % 3 else B[:n]
        if t % 5 == 0:
            a = np.stack([a, np.roll(a, 11)], axis=1)
        if t % 7 == 0:
            b = np.stack([b, np.roll(b, 5)], axis=1)
        sa = tuple(sorted(rng.integers(0, n, 2)))
        sb = tuple(sorted(rng.integers(0, n, 2)))
        zA, zB = int(rng.integers(1, 12)), int(rng.integers(1, 12))
        loop = bool(t % 4 == 0)
        n_out = 30000 if loop else None
        off = int(rng.integers(1000, 25000)) if rng.random() < 0.4 else None
        out, info = render(a, b, sa, sb, zA=zA, zB=zB, loop=loop, n_out=n_out, off_at=off)
        oc = 1 if out.ndim == 1 else out.shape[1]
        assert np.all(np.isfinite(out))
        for c in range(oc):
            inf = info[c]
            if inf["ra"] is None or inf["rb"] is None:
                continue
            o = out if out.ndim == 1 else out[:, c]
            check_channel(o, inf, _ch(a, c), _ch(b, c), (zA, zB), loop)


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
