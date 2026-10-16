# weave

Real-time interleaving of two audio sources in Max (`mc.gen~`). One source plays while the other is silent, and every switch happens on a zero crossing. A played segment can be repeated before the hand-off.

Inspired by CDP's `DISTORT INTERACT` (wavecycle interleave), but live: nothing is stored beyond a short memory window, so each source loses the audio that passes while the other plays.

> Status: the logic is tested in a Python model (`tests/`). The `gen~` code itself has not been run in Max yet.

## Setup in Max

1. Create `[mc.gen~]` with **2 inlets and 1 outlet** and open its editor.
2. Paste the contents of `gen/weave.genexpr` into a codebox.
3. Inlet 1 = source A, inlet 2 = source B (multichannel cables with the same channel count). Each channel runs its own independent copy, so a stereo pair is a 2-channel cable.
4. Send parameters as messages to the **left inlet**, e.g. `[a_zero_x 4(`.

## Parameters

| Name | Range | Default | Meaning |
|---|---|---|---|
| `a_zero_x`, `b_zero_x` | 1-100 | 2 | zero crossings a source plays before switching (2 = one cycle of a clean sine) |
| `a_repeats`, `b_repeats` | 0-1000 | 0 | extra times the segment is replayed before handing off; the other source stays silent |
| `win_ms` | 1-100 | 20 | how far back to search for a matching crossing when a source takes over |

## How it works

- **Memory window.** Both inputs are recorded into an 8192-sample ring (about 186 ms at 44.1 kHz per channel). The playing source is read back from it.
- **Hand-off.** When a source reaches its crossing count, the other starts at once, from the most recent crossing in its memory that runs the same direction as the cut. So the output passes through zero smoothly in one direction, with no gap. If the window has no such crossing it falls back to inverting a crossing of the other direction, and then to the newest sample.
- **Repeats.** The segment is copied into a loop buffer as it plays, and repeats replay that copy. The loop buffer keeps recording past the end of the segment, up to 100 crossings (or 262144 samples, about 5.9 s), so raising `zero_x` mid-repeat extends the loop into the audio that followed.
- **Updates happen at crossings.** `zero_x` is read only when a crossing is reached, and the repeat counts only when a loop ends. A change between crossings waits for the next one. Lowering `zero_x` shortens the loop at that crossing; raising it lengthens it.

## Limits

- A source below about 8 Hz cannot fill 100 crossings in the loop buffer. The loop then ends at the last stored crossing.
- If the loop buffer fills during a long stretch with no crossings, the end of the loop cannot land on a crossing. The tests did not hit this.
- If the memory window holds no crossing of a source (very low pitch, DC, silence), the hand-off falls back to the newest sample and may click. Raise `win_ms` for low sources.
- A source that never crosses zero stalls on its turn.

## Tests

```
python tests/test_weave.py
```

Needs `numpy`. The model in `model/weave_model.py` mirrors the codebox line for line.

## Tools

`tools/distort_interleave.py` is an offline Python version of CDP's wavecycle interleave (`python tools/distort_interleave.py a.wav b.wav out.wav`; needs `numpy` and `scipy`).
