// bufinfo.js - helper for weave (Max js object)
//   srcA a / srcB b   : reports mono/stereo and whether L and R differ (same messages you send to mc.gen~)
//   lowpass a 4000    : low-passes buffer "a" in place, before weave reads it (zero phase, about 24 dB/oct)
//   restore a         : puts back the audio that lowpass replaced (valid for the file that was filtered)
//   bang              : re-check the last name
inlets = 1;
outlets = 2;

var last = "";
var backup = {};        // name -> { n: frames, ch: [array per channel] }

function anything() {
    if ((messagename == "srcA" || messagename == "srcB") && arguments.length > 0) {
        last = String(arguments[0]);
    } else {
        last = messagename;
    }
    check();
}

function bang() { check(); }

function check() {
    if (last == "") return;
    var b = new Buffer(last);
    var c = b.channelcount();
    var n = b.framecount();
    if (n <= 1 || c < 1) {
        b.freepeer();
        outlet(0, last + ": empty (not loaded yet)");
        return;
    }
    var txt = last + ": " + (c == 1 ? "mono" : c == 2 ? "stereo" : c + " channels");
    if (c >= 2) {
        var CH = 32768;
        var maxdiff = 0, diffcount = 0, sumL = 0, sumR = 0, sumLR = 0;
        for (var i = 0; i < n; i += CH) {
            var cnt = Math.min(CH, n - i);
            var L = b.peek(1, i, cnt);
            var R = b.peek(2, i, cnt);
            for (var k = 0; k < cnt; k++) {
                var d = Math.abs(L[k] - R[k]);
                if (d > maxdiff) maxdiff = d;
                if (d > 1e-6) diffcount++;
                sumL += L[k] * L[k];
                sumR += R[k] * R[k];
                sumLR += L[k] * R[k];
            }
        }
        var corr = (sumL > 0 && sumR > 0) ? sumLR / Math.sqrt(sumL * sumR) : 0;
        if (maxdiff <= 1e-6) {
            txt += " | L and R are IDENTICAL (dual mono)";
        } else {
            txt += " | L and R differ (" + Math.round(100 * diffcount / n) + "% of samples, max diff "
                 + maxdiff.toFixed(4) + ", correlation " + corr.toFixed(3) + ")";
        }
    }
    b.freepeer();
    outlet(0, txt);
    outlet(1, c);
}

// second-order Butterworth low-pass, run forward and then backward (zero phase)
function lp2(x, fc, sr) {
    var n = x.length;
    var w0 = 2 * Math.PI * Math.min(fc, 0.45 * sr) / sr;
    var cw = Math.cos(w0), al = Math.sin(w0) / (2 * 0.70710678);
    var a0 = 1 + al;
    var b0 = ((1 - cw) / 2) / a0, b1 = (1 - cw) / a0, b2 = b0;
    var a1 = (-2 * cw) / a0, a2 = (1 - al) / a0;
    var y = new Array(n);
    var x1 = 0, x2 = 0, y1 = 0, y2 = 0, i, v;
    for (i = 0; i < n; i++) {                       // forward
        v = b0 * x[i] + b1 * x1 + b2 * x2 - a1 * y1 - a2 * y2;
        x2 = x1; x1 = x[i]; y2 = y1; y1 = v; y[i] = v;
    }
    x1 = x2 = y1 = y2 = 0;
    for (i = n - 1; i >= 0; i--) {                  // backward
        v = b0 * y[i] + b1 * x1 + b2 * x2 - a1 * y1 - a2 * y2;
        x2 = x1; x1 = y[i]; y2 = y1; y1 = v; y[i] = v;
    }
    return y;
}

function readAll(b, ch, n) {
    var out = [], CH = 32768;
    for (var i = 0; i < n; i += CH) {
        var part = b.peek(ch, i, Math.min(CH, n - i));
        for (var k = 0; k < part.length; k++) out.push(part[k]);
    }
    return out;
}

function writeAll(b, ch, arr) {
    var CH = 32768;
    for (var i = 0; i < arr.length; i += CH) b.poke(ch, i, arr.slice(i, i + CH));
}

// lowpass <buffername> <cutoff in Hz>
function lowpass(name, hz) {
    var b = new Buffer(name);
    var c = b.channelcount(), n = b.framecount(), ms = b.length();
    if (n <= 1 || c < 1 || ms <= 0) { b.freepeer(); outlet(0, name + ": empty, nothing filtered"); return; }
    var sr = n * 1000 / ms;
    if (!(hz > 0)) { b.freepeer(); outlet(0, "lowpass: give a cutoff in Hz"); return; }
    var chans = [];
    for (var ch = 1; ch <= c; ch++) chans.push(readAll(b, ch, n));
    backup[name] = { n: n, ch: chans };             // the unfiltered audio, for restore
    for (ch = 1; ch <= c; ch++) writeAll(b, ch, lp2(chans[ch - 1], hz, sr));
    b.freepeer();
    outlet(0, name + ": low-passed at " + hz + " Hz (sample rate " + Math.round(sr) + ")");
}

function restore(name) {
    var k = backup[name];
    if (!k) { outlet(0, name + ": nothing to restore"); return; }
    var b = new Buffer(name);
    if (b.framecount() != k.n) { b.freepeer(); outlet(0, name + ": the buffer changed since it was filtered, not restored"); return; }
    for (var ch = 1; ch <= k.ch.length; ch++) writeAll(b, ch, k.ch[ch - 1]);
    b.freepeer();
    outlet(0, name + ": original audio restored");
}
