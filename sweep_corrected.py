"""Correct Exp 10: exact-amplitude retention for every format/axis, phase-averaged.

fit_amp is validated against a known amplitude first, so a silent estimator bias
cannot pass as a physical result -- that is exactly what went wrong in the first pass.
"""
import json
import numpy as np

W, H = 1280, 720
yy, xx = np.mgrid[0:H, 0:W].astype(np.float64)
AMP = 50.0


def box_h(c):
    return c[:, : c.shape[1] // 2 * 2].reshape(c.shape[0], -1, 2).mean(axis=2)


def box_v(c):
    return c[: c.shape[0] // 2 * 2].reshape(-1, 2, c.shape[1]).mean(axis=1)


def box_both(c):
    return box_v(box_h(c))


def fit_amp(prof, period):
    """Least-squares amplitude of a sinusoid of known period. Period must be
    resolvable on this grid (>=4 samples per period) or the fit is degenerate."""
    n = np.arange(len(prof))
    w = 2 * np.pi * n / period
    A = np.column_stack([np.ones_like(w), np.cos(w), np.sin(w)])
    coef, *_ = np.linalg.lstsq(A, prof - prof.mean(), rcond=None)
    return float(np.hypot(coef[1], coef[2]))


# ---- estimator validation on a clean sinusoid of known amplitude, at each period
print("=== estimator validation (no filtering, true amplitude 50) ===")
bad = []
for P in (5, 6, 8, 12, 16, 24, 48, 96):
    got = fit_amp(128 + AMP * np.sin(2 * np.pi * np.arange(W) / P), P)
    if abs(got - AMP) > 0.02:
        bad.append((P, got))
print("  fit_amp error > 0.02 at:", bad or "none -- estimator is unbiased")

PHASES = np.linspace(0, 2 * np.pi, 48, endpoint=False)
rows = []
for P in (5, 6, 8, 10, 12, 16, 24, 32, 48, 64, 96):
    rec = {"P": P, "f": round(1.0 / P, 4)}
    for axis in ("X", "Y"):          # chroma varies along this axis
        A = (lambda c: fit_amp(c.mean(axis=0), P)) if axis == "X" else \
            (lambda c: fit_amp(c.mean(axis=1), P))
        Aout = (lambda c: fit_amp(c.mean(axis=0), P / 2.0)) if axis == "X" else \
               (lambda c: fit_amp(c.mean(axis=1), P / 2.0))
        r422, r420 = [], []
        for ph in PHASES:
            coord = xx if axis == "X" else yy
            c = 128 + AMP * np.sin(2 * np.pi * coord / P + ph)
            a0 = A(c)
            # 4:2:2 boxes X only; along Y it must be an untouched pass-through
            r422.append(100 * Aout(box_h(c)) / a0 if axis == "X" else 100.0)
            r420.append(100 * Aout(box_v(box_h(c))) / a0)
        rec[axis + "_422"] = round(float(np.mean(r422)), 2)
        rec[axis + "_420"] = round(float(np.mean(r420)), 2)
        rec[axis + "_sd"] = round(float(np.std(r420)), 2)
    rows.append(rec)
    print(f"P={P:>3} f={rec['f']:.4f} | 422X {rec['X_422']:6.2f}  420X {rec['X_420']:6.2f} "
          f"(sd {rec['X_sd']:4.2f}) | 422Y {rec['Y_422']:6.2f}  420Y {rec['Y_420']:6.2f}")

print("\n=== closed form: 2-tap box DTFT is |cos(pi f)|, NOT |sinc(2f)| ===")
print(f"{'f':>7} {'measured':>9} {'|cos(pi f)|':>12} {'diff':>7} {'|sinc(2f)|':>11} {'wrong by':>9}")
for r in rows:
    f = r["f"]
    cosf = 100 * abs(np.cos(np.pi * f))
    sincf = 100 * abs(np.sin(2 * np.pi * f) / (2 * np.pi * f))
    print(f"{f:>7.4f} {r['X_420']:>8.2f}% {cosf:>11.2f}% {r['X_420']-cosf:>+6.2f} "
          f"{sincf:>10.2f}% {sincf-cosf:>+8.2f}")

json.dump(rows, open("sweep_corrected.json", "w"), indent=1)
print("\nsaved sweep_corrected.json")

# ---- what actually happens AT and ABOVE the chroma Nyquist: real folding
print("\n=== folding: frequencies at/above chroma Nyquist (f_c = 0.25 cyc/px) ===")
print(f"{'P':>6} {'f':>7} {'420 X':>8} {'sd':>6} {'pred |cos(pi f)|':>16} {'alias into f<0.25':>18}")
for P in (4.0, 3.6, 3.2, 3.0, 2.8):
    rets = []
    for ph in PHASES:
        c = 128 + AMP * np.sin(2 * np.pi * xx / P + ph)
        a0 = fit_amp(c.mean(axis=0), P)
        d = box_both(c)
        # after decimation the surviving frequency is |f - nearest integer|,
        # i.e. f folds about 0.5 on the decimated grid; its period there is
        # 1/|2f - round(2f)| when that is resolvable
        alias = abs(P / 2.0 - round(P / 2.0))
        period_out = 1.0 / abs(1.0 / (P / 2.0) - round(1.0 / (P / 2.0))) if alias > 1e-9 else None
        rets.append(100 * (fit_amp(d.mean(axis=0), period_out) / a0
                           if period_out and period_out >= 4 else np.nan))
    pred = 100 * abs(np.cos(np.pi * (1.0 / P)))
    print(f"{P:>6.1f} {1/P:>7.4f} {np.nanmean(rets):>7.2f}% {np.nanstd(rets):>6.2f} "
          f"{pred:>15.2f}% {'degenerate (period<4 samples)' if np.isnan(rets).all() else ''}")
