"""
Checks that the gen~ code (gen/weave_buffer.genexpr) does what the verified model does.
gen_port() below is the gen~ code translated statement by statement into Python (locals pattern, same
control flow, same SCAN loops); the tests compare it with buffer_model.render on random cases.
Run with:  python tests/test_buffer_gen.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "model"))
from buffer_model import render, sr  # noqa: E402

SCAN = 100000


def gen_port(xa, xb, a_start, a_end, b_start, b_end, zA, zB, loop_on, off_at, n_out):
    """one channel of the gen~ code; xa, xb are the (already channel-picked) buffer contents"""
    src = [xa, xb]
    nA, nB = len(xa), len(xb)
    running = 0; pplay = 0; cur = 0; pol = 1.0; pv = 0.0; hpv = 0; zcount = 0; kA = 0; kB = 0; lp = 0
    s0A = e0A = okA = s0B = e0B = okB = 0
    out = np.zeros(n_out)
    for t in range(n_out):
        play_on = 1 if (off_at is None or t < off_at) else 0
        vrun, vcur, vpol, vpv, vhpv, vz, vkA, vkB, vlp = running, cur, pol, pv, hpv, zcount, kA, kB, lp
        vs0A, ve0A, vokA, vs0B, ve0B, vokB = s0A, e0A, okA, s0B, e0B, okB
        if play_on >= 1 and pplay < 1:
            vlp = loop_on
            for w in range(2):
                n = nA if w == 0 else nB
                sel1 = a_start if w == 0 else b_start
                sel2 = a_end if w == 0 else b_end
                x = src[w]

                def cross(i):
                    xi, xp = x[i], x[i - 1]
                    return (xp < 0 and xi >= 0) or (xp >= 0 and xi < 0)
                s = -1
                for m in range(SCAN):
                    i = sel1 - m
                    if i >= 1 and i < n:
                        if cross(i): s = i; break
                    i = sel1 + m
                    if m > 0 and i >= 1 and i < n:
                        if cross(i): s = i; break
                e = -1
                if s >= 0:
                    d0 = 1 if x[s] >= 0 else -1
                    for m in range(SCAN):
                        i = sel2 - m
                        if i > s and i < n:
                            di = 1 if x[i] >= 0 else -1
                            if cross(i) and di == d0: e = i; break
                        i = sel2 + m
                        if m > 0 and i > s and i < n:
                            di = 1 if x[i] >= 0 else -1
                            if cross(i) and di == d0: e = i; break
                if w == 0:
                    vs0A, ve0A, vokA = s, e, 1 if (s >= 0 and e >= 0) else 0
                else:
                    vs0B, ve0B, vokB = s, e, 1 if (s >= 0 and e >= 0) else 0
            vkA = 0; vkB = 0; vz = 0; vhpv = 0; vpol = 1
            vcur = 0 if vokA == 1 else 1
            vrun = 1 if (vokA == 1 or vokB == 1) else 0
        y = 0.0
        if vrun == 1:
            s = vcur
            L = (ve0A - vs0A) if s == 0 else (ve0B - vs0B)
            ks = vkA if s == 0 else vkB
            base = vs0A if s == 0 else vs0B
            exhausted = (vlp == 0 and ks >= L)
            idx = base + (ks % L) if vlp == 1 else base + ks
            x = src[s][idx]
            zc = (vhpv == 1) and ((vpv < 0 and x >= 0) or (vpv >= 0 and x < 0))
            event = (zc or exhausted)
            if play_on < 1 and event:
                vrun = 0
            else:
                z = vz
                cut = 0
                if zc:
                    tz = zA if s == 0 else zB
                    z += 1
                    if z >= tz: cut = 1
                if exhausted: cut = 1
                if cut == 1:
                    od = 0
                    if s == 0:
                        if vokB == 0 or (vlp == 0 and vkB >= (ve0B - vs0B)): od = 1
                    else:
                        if vokA == 0 or (vlp == 0 and vkA >= (ve0A - vs0A)): od = 1
                    if od == 1:
                        if exhausted:
                            vrun = 0
                        else:
                            y = vpol * x; vpv = x; vhpv = 1; vz = 0
                            if s == 0: vkA = ks + 1
                            else: vkB = ks + 1
                    else:
                        nld = vpol * (1 if x >= 0 else -1)
                        o = 1 - s
                        ko = vkA if o == 0 else vkB
                        Lo = (ve0A - vs0A) if o == 0 else (ve0B - vs0B)
                        bo = vs0A if o == 0 else vs0B
                        io = bo + (ko % Lo) if vlp == 1 else bo + ko
                        xo = src[o][io]
                        dO = 1 if xo >= 0 else -1
                        vpol = nld * dO
                        y = vpol * xo; vpv = xo; vhpv = 1; vz = 0; vcur = o
                        if o == 0: vkA = ko + 1
                        else: vkB = ko + 1
                else:
                    y = vpol * x; vpv = x; vhpv = 1; vz = z
                    if s == 0: vkA = ks + 1
                    else: vkB = ks + 1
        running, cur, pol, pv, hpv, zcount, kA, kB, lp = vrun, vcur, vpol, vpv, vhpv, vz, vkA, vkB, vlp
        s0A, e0A, okA, s0B, e0B, okB = vs0A, ve0A, vokA, vs0B, ve0B, vokB
        pplay = play_on
        out[t] = y
    return out


def _compare(a, b, sa, sb, zA, zB, loop, n_out, off_at):
    ref, info = render(a, b, sa, sb, zA=zA, zB=zB, loop=loop, n_out=n_out, off_at=off_at)
    ref = ref.reshape(len(ref), -1)
    ca = 1 if a.ndim == 1 else a.shape[1]
    cb = 1 if b.ndim == 1 else b.shape[1]
    for c in range(ref.shape[1]):
        xa = a if a.ndim == 1 else a[:, min(c, ca - 1)]
        xb = b if b.ndim == 1 else b[:, min(c, cb - 1)]
        got = gen_port(xa, xb, sa[0], sa[1], sb[0], sb[1], zA, zB, 1 if loop else 0, off_at, len(ref))
        assert np.array_equal(got, ref[:, c]), ("gen port differs from the model", c, int(np.argmax(got != ref[:, c])))


def test_port_matches_model_basic():
    T = np.arange(2 * sr) / sr
    A = 0.6 * np.sin(2 * np.pi * 220 * T)
    B = 0.5 * np.sin(2 * np.pi * 330 * T + 1) + 0.2 * np.sin(2 * np.pi * 700 * T)
    for loop, off in [(False, None), (True, None), (True, 20000), (False, 15000)]:
        _compare(A, B, (3000, 30000), (4000, 20000), 2, 3, loop, 60000, off)


def test_port_matches_model_fuzz():
    rng = np.random.default_rng(11)
    for t in range(30):
        n = int(rng.integers(6000, 20000))
        a = 0.4 * rng.standard_normal(n)
        b = 0.4 * rng.standard_normal(n)
        if t % 4 == 0:
            a = np.stack([a, np.roll(a, 7)], axis=1)
        if t % 5 == 0:
            b = np.stack([b, np.roll(b, 3)], axis=1)
        sa = tuple(int(v) for v in sorted(rng.integers(0, n, 2)))
        sb = tuple(int(v) for v in sorted(rng.integers(0, n, 2)))
        loop = bool(t % 3 == 0)
        off = int(rng.integers(500, 18000)) if rng.random() < 0.5 else None
        _compare(a, b, sa, sb, int(rng.integers(1, 10)), int(rng.integers(1, 10)), loop, 25000, off)


def test_port_silent_source():
    T = np.arange(sr) / sr
    B = 0.5 * np.sin(2 * np.pi * 330 * T)
    _compare(np.full(sr, 0.2), B, (0, 3000), (1000, 20000), 2, 2, False, 30000, None)
    _compare(B, np.zeros(sr), (1000, 20000), (0, 3000), 2, 2, True, 30000, 12000)


if __name__ == "__main__":
    failed = 0
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)]
    for name, fn in tests:
        try:
            fn(); print("PASS", name)
        except AssertionError as e:
            failed += 1; print("FAIL", name, e)
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)
