"""
Python model of the buffer version of weave (CDP-style interleave of two buffers).

Design (all decided with the user):
  - two sources A and B, each mono or stereo, each with a selected range (start, end) in samples
  - channels are independent; a mono source feeds every output channel
  - the range start snaps to the nearest zero crossing; the range end snaps to the nearest
    crossing that travels in the same direction as the start (so the range holds an even number of crossings)
  - a chunk is zero_x crossings long; a source only advances while it plays (CDP-style)
  - at each join the polarity of the new chunk is chosen so the output keeps travelling through zero
  - one-shot: each range plays once, what is left of the longer one is appended
    loop: both ranges wrap until play goes off
  - play off takes effect at the next zero crossing (the output goes silent there)

A "crossing at index i" means the sign changes between samples i-1 and i; sample i is the first
sample after the crossing.  A chunk starts on such a sample.
"""
import numpy as np

sr = 44100


def crossings(x):
    """indices i >= 1 where the sign changes between x[i-1] and x[i]"""
    return np.flatnonzero((x[:-1] < 0) != (x[1:] < 0)) + 1


def snap(x, sel_start, sel_end):
    """-> (s0, e0) or None.  s0 = nearest crossing to sel_start; e0 = nearest crossing to sel_end
    that travels in the same direction as s0 and lies after it."""
    cr = crossings(x)
    if len(cr) < 2:
        return None
    s0 = int(cr[np.argmin(np.abs(cr - sel_start))])        # ties keep the earlier one
    d0 = x[s0] >= 0
    cand = cr[(cr > s0) & ((x[cr] >= 0) == d0)]
    if len(cand) == 0:
        return None
    e0 = int(cand[np.argmin(np.abs(cand - sel_end))])
    return s0, e0


def _channel(xa, xb, ra, rb, zA, zB, loop, n_out, off_at, changes):
    src = [xa, xb]
    rg = [ra, rb]                                          # (s0, e0) or None
    L = [None if r is None else r[1] - r[0] for r in rg]
    P = dict(zA=zA, zB=zB)
    ch = sorted(changes); ci = 0
    out = np.zeros(n_out)
    chunks = []                                            # dict(src, k0, k1, pol, n0)
    k = [0, 0]

    def done(s):
        return rg[s] is None or ((not loop) and k[s] >= L[s])

    def get(s, kk):
        """sample at stream position kk of source s (kk == L is the crossing sample after the range end)"""
        s0, e0 = rg[s]
        if loop:
            return src[s][s0 + kk % L[s]]
        return src[s][s0 + kk]                            # kk == L[s] reads x[e0]

    if rg[0] is None and rg[1] is None:
        return out, chunks, 0
    cur = 0 if rg[0] is not None else 1
    pol = 1.0
    pv = None
    zcount = 0
    cstart = dict(src=cur, k0=0, pol=pol, n0=0)
    length = n_out
    play = True
    for n in range(n_out):
        while ci < len(ch) and ch[ci][0] <= n:
            P[ch[ci][1]] = ch[ci][2]; ci += 1
        if off_at is not None and n >= off_at:
            play = False
        s = cur
        ks = k[s]
        exhausted = (not loop) and ks >= L[s]
        x = get(s, ks)                                     # when exhausted this is x[e0], the sample after the end
        zc = (pv is not None) and ((pv < 0) != (x < 0))
        event = zc or exhausted                            # both are zero crossings of the output
        if (not play) and event:
            length = n
            break
        z = zcount
        cut = False
        if zc:
            tz = P['zA'] if s == 0 else P['zB']            # the count is read only at a crossing
            z += 1
            if z >= tz:
                cut = True
        if exhausted:
            cut = True
        if cut:
            o = 1 - s
            if done(o):
                if exhausted:                              # both ranges are used up
                    length = n
                    break
                # the other source is finished: carry on with this one (the append)
                out[n] = pol * x; pv = x; k[s] = ks + 1; zcount = 0
                continue
            nld = pol * (1 if x >= 0 else -1)              # direction the output is travelling through zero
            chunks.append(dict(cstart, k1=ks, n1=n))
            j = k[o]
            xo = get(o, j)
            do = 1 if xo >= 0 else -1
            pol = nld * do
            out[n] = pol * xo
            pv = xo; k[o] = j + 1; cur = o; zcount = 0
            cstart = dict(src=o, k0=j, pol=pol, n0=n, nld=nld)
        else:
            out[n] = pol * x; pv = x; k[s] = ks + 1; zcount = z
    else:
        length = n_out
    chunks.append(dict(cstart, k1=k[cstart['src']], n1=length))
    return out, chunks, length


def render(a, b, sel_a, sel_b, zA=2, zB=2, loop=False, n_out=None, off_at=None, changes=()):
    """a, b: (n,) mono or (n, 2) stereo float arrays; sel_*: (start, end) in samples.
    returns (out, info) with out of shape (n,) or (n, channels)."""
    a = np.asarray(a, float); b = np.asarray(b, float)
    ca = 1 if a.ndim == 1 else a.shape[1]
    cb = 1 if b.ndim == 1 else b.shape[1]
    oc = max(ca, cb)
    if n_out is None:
        n_out = (len(a) + len(b)) + 16
    outs, info = [], []
    for c in range(oc):
        xa = a if a.ndim == 1 else a[:, min(c, ca - 1)]
        xb = b if b.ndim == 1 else b[:, min(c, cb - 1)]
        ra = snap(xa, *sel_a)
        rb = snap(xb, *sel_b)
        o, chunks, length = _channel(xa, xb, ra, rb, zA, zB, loop, n_out, off_at, changes)
        outs.append(o)
        info.append(dict(ra=ra, rb=rb, chunks=chunks, length=length))
    out = np.stack(outs, axis=1)
    if oc == 1:
        out = out[:, 0]
    return out, info
