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


def gen_port(xa, xb, a_start, a_end, b_start, b_end, zA, zB, loop_t, trig_t, n_out):
    """one channel of the gen~ code; xa, xb are the (already channel-picked) buffer contents;
    loop_t and trig_t are the loop_on and trig parameter values at every sample"""
    src = [xa, xb]
    nA, nB = len(xa), len(xb)
    running = 0; ploop = 0; ptrig = 0; cur = 0; pol = 1.0; pv = 0.0; hpv = 0; zcount = 0; kA = 0; kB = 0; lp = 0
    s0A = e0A = okA = s0B = e0B = okB = 0
    psA = peA = psB = peB = 0
    pend = 0; q0A = q1A = qokA = q0B = q1B = qokB = 0
    a_start, a_end, b_start, b_end = [np.broadcast_to(np.asarray(v), (n_out,)) for v in (a_start, a_end, b_start, b_end)]
    out = np.zeros(n_out)
    for t in range(n_out):
        loop_on = loop_t[t]; trig = trig_t[t]
        vpend, vq0A, vq1A, vqokA, vq0B, vq1B, vqokB = pend, q0A, q1A, qokA, q0B, q1B, qokB
        vrun, vcur, vpol, vpv, vhpv, vz, vkA, vkB, vlp = running, cur, pol, pv, hpv, zcount, kA, kB, lp
        vs0A, ve0A, vokA, vs0B, ve0B, vokB = s0A, e0A, okA, s0B, e0B, okB
        loopedge = (loop_on >= 1 and ploop < 1)
        trigedge = (trig >= 1 and ptrig < 1 and loop_on < 1)
        startnow = (vrun == 0 and (loopedge or trigedge))
        selchg = (a_start[t] != psA or a_end[t] != peA or b_start[t] != psB or b_end[t] != peB)
        resnap = (vrun == 1 and vlp == 1 and selchg)
        if startnow or resnap:
            if startnow:
                vlp = 1 if loopedge else 0
            for w in range(2):
                n = nA if w == 0 else nB
                sel1 = a_start[t] if w == 0 else b_start[t]
                sel2 = a_end[t] if w == 0 else b_end[t]
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
                    vq0A, vq1A, vqokA = s, e, 1 if (s >= 0 and e >= 0) else 0
                else:
                    vq0B, vq1B, vqokB = s, e, 1 if (s >= 0 and e >= 0) else 0
            vpend = 1
            if startnow:
                vs0A, ve0A, vokA, vs0B, ve0B, vokB = vq0A, vq1A, vqokA, vq0B, vq1B, vqokB
                vpend = 0
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
            if vlp == 1 and loop_on < 1 and event:
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
                    if vlp == 1 and vpend == 1 and vqokA == 1 and vqokB == 1:
                        vs0A, ve0A, vs0B, ve0B = vq0A, vq1A, vq0B, vq1B
                        vkA = 0; vkB = 0; vpend = 0
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
        ploop = loop_on; ptrig = trig
        pend, q0A, q1A, qokA, q0B, q1B, qokB = vpend, vq0A, vq1A, vqokA, vq0B, vq1B, vqokB
        psA, peA, psB, peB = a_start[t], a_end[t], b_start[t], b_end[t]
        out[t] = y
    return out


def _params(loop, off_at, n):
    """parameter trajectories for one run started at sample 0: a loop run (loop_on up, optionally switched
    off at off_at) or a one-shot (loop_on stays 0, trig goes up)"""
    lt = np.zeros(n); tt = np.zeros(n)
    if loop:
        lt[:] = 1
        if off_at is not None:
            lt[off_at:] = 0
    else:
        tt[:] = 1
    return lt, tt


def _compare(a, b, sa, sb, zA, zB, loop, n_out, off_at):
    ref, info = render(a, b, sa, sb, zA=zA, zB=zB, loop=loop, n_out=n_out, off_at=off_at)
    ref = ref.reshape(len(ref), -1)
    ca = 1 if a.ndim == 1 else a.shape[1]
    cb = 1 if b.ndim == 1 else b.shape[1]
    for c in range(ref.shape[1]):
        xa = a if a.ndim == 1 else a[:, min(c, ca - 1)]
        xb = b if b.ndim == 1 else b[:, min(c, cb - 1)]
        lt, tt = _params(loop, off_at, len(ref))
        got = gen_port(xa, xb, sa[0], sa[1], sb[0], sb[1], zA, zB, lt, tt, len(ref))
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


def _setup():
    T = np.arange(2 * sr) / sr
    A = 0.6 * np.sin(2 * np.pi * 220 * T)
    B = 0.5 * np.sin(2 * np.pi * 330 * T + 1) + 0.2 * np.sin(2 * np.pi * 700 * T)
    return A, B, (3000, 30000), (4000, 20000)


def test_nothing_plays_without_a_trigger():
    A, B, sa, sb = _setup()
    n = 20000
    z = np.zeros(n)
    out = gen_port(A, B, sa[0], sa[1], sb[0], sb[1], 2, 2, z, z, n)
    assert np.all(out == 0)                                  # loop off, no trigger: silence
    out = gen_port(A, B, sa[0], sa[1], sb[0], sb[1], 2, 2, z, np.ones(n) * 0, n)
    assert np.all(out == 0)


def test_trigger_plays_once_and_can_be_repeated():
    A, B, sa, sb = _setup()
    ref, info = render(A, B, sa, sb, zA=2, zB=2, loop=False)
    L = info[0]["length"]
    n = 3 * L + 5000
    lt = np.zeros(n); tt = np.zeros(n)
    tt[10:20] = 1                                            # a trigger: 0 -> 1 -> 0
    tt[L // 2:L // 2 + 5] = 1                                # a trigger while it is playing: ignored
    second = L + 1500
    tt[second:second + 5] = 1                                # a trigger after it has ended: plays again
    out = gen_port(A, B, sa[0], sa[1], sb[0], sb[1], 2, 2, lt, tt, n)
    assert np.array_equal(out[10:10 + L], ref[:L])
    assert np.all(out[:10] == 0) and np.all(out[10 + L:second] == 0)
    assert np.array_equal(out[second:second + L], ref[:L])
    assert np.all(out[second + L:] == 0)


def test_loop_on_starts_and_off_stops_and_trigger_is_ignored_while_looping():
    A, B, sa, sb = _setup()
    n = 40000
    lt = np.zeros(n); tt = np.zeros(n)
    lt[100:25000] = 1
    tt[5000:5005] = 1                                        # a trigger while the loop is on: nothing extra
    out = gen_port(A, B, sa[0], sa[1], sb[0], sb[1], 2, 2, lt, tt, n)
    assert np.all(out[:100] == 0) and np.any(out[100:200] != 0)
    nz = np.flatnonzero(out != 0)
    stop = nz[-1] + 1
    assert 25000 <= stop < 26500                             # stopped at the first crossing after loop_on went to 0
    assert np.all(out[stop:] == 0)
    ref, info = render(A, B, sa, sb, zA=2, zB=2, loop=True, n_out=n)
    assert np.array_equal(out[100:stop], ref[:stop - 100])   # and up to then it is the same as the model


def test_moving_a_selection_while_looping_takes_over_at_the_next_hand_off():
    A, B, sa, sb = _setup()
    n = 60000
    lt = np.ones(n); tt = np.zeros(n)
    t0 = 20000
    na_s = np.full(n, sa[0]); na_e = np.full(n, sa[1]); nb_s = np.full(n, sb[0]); nb_e = np.full(n, sb[1])
    na_s[t0:] = 40000; na_e[t0:] = 52000                     # A's selection moves while the loop runs
    out = gen_port(A, B, na_s, na_e, nb_s, nb_e, 2, 2, lt, tt, n)
    old = gen_port(A, B, sa[0], sa[1], sb[0], sb[1], 2, 2, lt, tt, n)
    assert np.array_equal(out[:t0], old[:t0])                # nothing changes before the move
    assert not np.array_equal(out, old)                      # and the move has an effect
    ra = render(A, B, (40000, 52000), sb, loop=True, n_out=n)[1][0]["ra"]
    rb = render(A, B, sa, sb, loop=True, n_out=n)[1][0]["rb"]
    aset = set(np.abs(A[ra[0]:ra[1]]).tolist()); bset = set(np.abs(B[rb[0]:rb[1]]).tolist())
    tail = out[n - 8000:]
    assert np.any(tail != 0)
    assert all((abs(v) in aset) or (abs(v) in bset) for v in tail if v != 0)   # only the new A range and B are heard
    old_only = set(np.abs(A[3000:30000]).tolist()) - aset
    assert not any(abs(v) in old_only for v in tail)                           # the old A range is gone
    assert np.max(np.abs(np.diff(out))) <= 1.5 * np.max(np.abs(np.diff(old)))   # the takeover join is no rougher than the others


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
