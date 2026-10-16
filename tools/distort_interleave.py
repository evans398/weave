#!/usr/bin/env python3
"""
distort_interleave.py - waveset interleave, after CDP "DISTORT INTERACT" mode 1.

Each mono input is cut into wavesets (pseudo-wavecycles): a waveset runs from
one zero crossing, through the next, to the one after that (i.e. one positive
lobe plus one negative lobe, or the reverse). The output alternates them:

    A1 B1 A2 B2 A3 B3 ...

Each file's read position only advances when one of its wavesets is used, so
interleaving a sound with itself gives a half-speed version (every waveset
twice), exactly as described in the CDP documentation.

Usage:
    python distort_interleave.py a.wav b.wav out.wav [--stop-at-shorter]
                                 [--skipcycles N] [--normalize]

Requires numpy and scipy.  Mono WAV files only (as in CDP).
"""
import argparse
import sys
import numpy as np
from scipy.io import wavfile


# ----------------------------------------------------------------- I/O ------
def read_mono(path):
    sr, data = wavfile.read(path)
    if data.ndim != 1:
        sys.exit(f"{path}: mono files only (CDP DISTORT INTERACT is mono-only). "
                 "Split channels first, or process each channel separately.")
    if data.dtype == np.int16:
        x = data.astype(np.float64) / 32768.0
    elif data.dtype == np.int32:
        x = data.astype(np.float64) / 2147483648.0
    elif data.dtype == np.uint8:
        x = (data.astype(np.float64) - 128.0) / 128.0
    else:
        x = data.astype(np.float64)
    return sr, x


# ------------------------------------------------------------ wavesets ------
def waveset_bounds(x):
    """
    Boundaries (sample indices) of the wavesets of x, as a sorted array so that
    waveset k is x[b[k]:b[k+1]].

    A zero crossing is a sign change between consecutive samples (a sample of
    exactly 0 takes the sign of the sample before it, so it is not counted as
    a separate crossing). A waveset spans two sign changes. The start and end
    of the file are used as outer boundaries so no samples are lost.
    """
    s = np.sign(x)
    # carry the previous sign through exact zeros
    for i in range(1, len(s)):
        if s[i] == 0:
            s[i] = s[i - 1]
    # leading zeros: take the first non-zero sign
    nz = np.flatnonzero(s)
    if len(nz) == 0:
        return np.array([0, len(x)])
    s[: nz[0]] = s[nz[0]]

    cross = np.flatnonzero(s[1:] != s[:-1]) + 1      # index of first sample after the flip
    inner = cross[::2]                                # every second crossing = waveset start/end
    b = np.concatenate(([0], inner, [len(x)]))
    return np.unique(b)


# ---------------------------------------------------------- interleave ------
def interleave(a, b, skipcycles=0, stop_at_shorter=False):
    ba = waveset_bounds(a)
    bb = waveset_bounds(b)
    na, nb = len(ba) - 1, len(bb) - 1

    out = []

    # optional: keep the first N wavesets of A untouched (protects the attack)
    ia = 0
    if skipcycles > 0:
        ia = min(skipcycles, na)
        out.append(a[ba[0]:ba[ia]])
    ib = 0

    while ia < na and ib < nb:
        out.append(a[ba[ia]:ba[ia + 1]]); ia += 1
        out.append(b[bb[ib]:bb[ib + 1]]); ib += 1

    if not stop_at_shorter:                           # append whatever is left
        if ia < na:
            out.append(a[ba[ia]:])
        if ib < nb:
            out.append(b[bb[ib]:])

    return np.concatenate(out) if out else np.zeros(0)


# ----------------------------------------------------------------- main -----
def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("infile1")
    p.add_argument("infile2")
    p.add_argument("outfile")
    p.add_argument("--stop-at-shorter", action="store_true",
                   help="end when either file runs out of wavesets "
                        "(default: append the remaining wavesets of the longer file)")
    p.add_argument("--skipcycles", type=int, default=0,
                   help="number of wavesets of file 1 to pass through untouched first")
    p.add_argument("--normalize", action="store_true",
                   help="scale the output to a 0.99 peak (default: leave level alone)")
    args = p.parse_args()

    sr1, a = read_mono(args.infile1)
    sr2, b = read_mono(args.infile2)
    if sr1 != sr2:
        sys.exit(f"sample rates differ ({sr1} vs {sr2}); resample one first.")

    y = interleave(a, b, args.skipcycles, args.stop_at_shorter)

    peak = np.max(np.abs(y)) if len(y) else 0.0
    if args.normalize and peak > 0:
        y = y * (0.99 / peak)
    elif peak > 1.0:
        print(f"warning: output peak is {peak:.2f}; use --normalize or lower the "
              "input levels (output is float32, so nothing is clipped in the file)")

    wavfile.write(args.outfile, sr1, y.astype(np.float32))
    print(f"wavesets: A={len(waveset_bounds(a)) - 1}  B={len(waveset_bounds(b)) - 1}  "
          f"-> {len(y)} samples ({len(y) / sr1:.3f} s)")


if __name__ == "__main__":
    main()
