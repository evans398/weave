"""
Python model of the weave gen~ codebox (gen/weave.genexpr), written line for line
so the logic can be tested without Max.

run_v2(a, b, ...) takes two mono arrays and returns (output, events).
Parameter changes can be scheduled with changes=[(sample, name, value), ...]
(names: zA, zB, repA, repB). They are read only at zero crossings, as in the codebox.
"""
import numpy as np

sr = 44100

def run_v2(a, b, zA=2, zB=2, repA=0, repB=0, revA=0, revB=0, changes=(), win_ms=20, N=8192, SN=262144, ZMAX=100, slope_match=True):
    """Line-for-line model of the new gen~ code: continuous loop capture, params read only at crossings."""
    P = dict(zA=zA, zB=zB, repA=repA, repB=repB, revA=revA, revB=revB)
    ch = sorted(changes); ci = 0
    n = len(a)
    buf = np.zeros((N, 2)); seg = np.zeros(SN); wp = 0
    rp = 0; rn = 0; pv = 0.0; cur = 0; zcount = 0; pol = 1.0
    sd = 0; rd = 0; ph = 0; ov = 0; rv = 0
    cxi = np.zeros(128, dtype=int)
    clen = 0; cz = 0; cpv = 0.0; cdone = 0; lz = -1
    lim = min(int(win_ms*sr/1000), N - 16)
    out = np.zeros(n); ev = []; abs_start = 0
    for i in range(n):
        while ci < len(ch) and ch[ci][0] <= i:
            k, v = ch[ci][1], ch[ci][2]
            P[k] = (min(max(v, 1), 100) if k in ("zA", "zB")
                    else min(max(v, 0), 1) if k in ("revA", "revB") else min(max(v, 0), 1000))
            ci += 1
        buf[wp, 0] = a[i]; buf[wp, 1] = b[i]
        xc = buf[rp, cur]
        x = xc if ph == 0 else (seg[rn] if rn >= 0 else 0.0)
        # ---- continuous capture of the playing source's stream into the loop buffer
        nclen, ncz, ncpv, ncdone, nlz, nov = clen, cz, cpv, cdone, lz, ov
        if cdone == 0:
            cr = (cpv < 0 and xc >= 0) or (cpv >= 0 and xc < 0)
            seg[clen] = xc
            if cr: ncz = cz + 1; nlz = clen; cxi[ncz] = clen
            ncpv = xc; nclen = clen + 1
            if ncz >= ZMAX or nclen >= SN: ncdone = 1
        # ---- zero crossing of the played signal; params are read ONLY here
        rise = (pv < 0 and x >= 0); fall = (pv >= 0 and x < 0)
        zc = rise or fall
        if rv == 1: zc = False                            # no crossing counting inside a reversed play
        dr = 1 if rise else -1
        z = zcount; cut = 0
        if zc:
            tz = P['zA'] if cur == 0 else P['zB']
            z += 1
            if z >= tz: cut = 1
        if ph == 1 and rv == 0 and ncdone and rn >= nlz: cut = 1      # end of what was stored
        if ph == 1 and rv == 1 and rn < 0: cut = 1                    # reversed play has reached its start
        if ph == 0 and ncdone and cut == 0: nov = 1       # segment too long to store: no repeats
        y = pol * x; npv = x; nrp = (rp + 1) % N; nrn = (rn - 1) if rv == 1 else (rn + 1)
        ncur = cur; nz = z; npol = pol; nsd = sd; nrd = rd; nph = ph; nrv = rv
        if cut:
            nld = pol * (-sd) if (ph == 1 and rv == 1) else pol * dr
            reps = P['repA'] if cur == 0 else P['repB']   # read at a crossing
            if rd < reps and sd != 0 and ov == 0:
                rdn = rd + 1
                revon = P['revA'] if cur == 0 else P['revB']
                tz2 = P['zA'] if cur == 0 else P['zB']
                cT = min(tz2, ncz)
                use_rev = revon >= 1 and (rdn % 2) == 1 and cT >= 1
                if use_rev:
                    bb = cxi[cT]
                    dB = 1 if seg[bb] >= 0 else -1
                    npol = nld * (-dB)
                    ys = seg[bb - 1]; y = npol * ys; npv = ys
                    nrn = bb - 2; nrv = 1
                else:
                    npol = nld * sd
                    ys = seg[0]; y = npol * ys; npv = ys
                    nrn = 1; nrv = 0
                nph = 1; nz = 0; nrd = rdn
                ev.append(dict(i=i, kind="repeat", nld=nld, start=abs_start, cur=cur, rv=nrv))
            else:
                other = 1 - cur
                idx = -1; idx2 = -1; d2 = 0
                osl = abs(x - pv)                                  # slope the output is leaving with
                if ph == 1 and rv == 1: osl = abs(pv - seg[1])
                best = 1e9
                v1 = buf[wp, other]
                for k in range(lim):
                    j = (wp - k - 1 + N) % N
                    v0 = buf[j, other]
                    if (v0 < 0 and v1 >= 0) or (v0 >= 0 and v1 < 0):
                        dd = 1 if v1 >= 0 else -1
                        ps = (wp - k + N) % N
                        if idx2 < 0: idx2 = ps; d2 = dd
                        if dd == nld:
                            if not slope_match: idx = ps; break
                            cs = abs(abs(v1 - v0) - osl)              # how far this crossing's slope is from ours
                            if cs < best: best = cs; idx = ps         # ties keep the newest
                    v1 = v0
                si = wp; spol = 1.0; ssd = 0
                if idx >= 0: si = idx; ssd = nld
                elif idx2 >= 0: si = idx2; spol = nld * d2; ssd = d2
                ys = buf[si, other]
                y = spol * ys; npv = ys; nrp = (si + 1) % N
                ncur = other; npol = spol; nz = 0; nsd = ssd; nrd = 0
                seg[0] = ys; nclen = 1; ncz = 0; ncpv = ys; ncdone = 0; nlz = -1; nov = 0
                nph = 0; nrn = 0; nrv = 0
                abs_start = i - ((wp - si) % N)
                ev.append(dict(i=i, kind="handoff", nld=nld, start=abs_start, cur=ncur, rv=0))
        rp = nrp; rn = nrn; pv = npv; cur = ncur; zcount = nz; pol = npol
        sd = nsd; rd = nrd; ph = nph; ov = nov; rv = nrv
        clen = nclen; cz = ncz; cpv = ncpv; cdone = ncdone; lz = nlz
        out[i] = y
        wp = (wp + 1) % N
    return out, ev

def join_ok(out, ev):
    ok = tot = 0
    for e in ev:
        i, ld = e["i"], e["nld"]
        if i < 1: continue
        p, q = out[i-1], out[i]; tot += 1
        if (ld == -1 and p >= 0 and q <= 0) or (ld == 1 and p <= 0 and q >= 0): ok += 1
    return ok, tot

def chains(ev):
    """list of dict(start, cur, cuts) ; cuts[0]=handoff index, then repeat indices, then the next handoff index"""
    res = []; curc = None
    for e in ev:
        if e["kind"] == "handoff":
            if curc is not None:
                curc["cuts"].append(e["i"]); res.append(curc)
            curc = dict(start=e["start"], cur=e["cur"], cuts=[e["i"]], rvs=[0])
        elif curc is not None:
            curc["cuts"].append(e["i"]); curc["rvs"].append(e["rv"])
    return res

def check_chain(out, ch, a, b):
    """each play must equal the source stream from the chain start, up to polarity"""
    src = a if ch["cur"] == 0 else b
    bad = 0
    for k in range(len(ch["cuts"]) - 1):
        s, e = ch["cuts"][k], ch["cuts"][k+1]
        L = e - s
        seg = out[s:e]
        ref = src[ch["start"]: ch["start"] + L]
        if ch["rvs"][k]: ref = ref[::-1]          # a reversed play is the segment read backwards
        if len(ref) < L or not np.allclose(np.abs(seg), np.abs(ref), atol=1e-12): bad += 1
    return bad

