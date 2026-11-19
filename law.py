"""Exact law for a 2-tap box, validated numerically.

y[n] = (x[2n] + x[2n+1])/2 with x[m] = sin(2*pi*f*m + phi) gives exactly
    y[n] = cos(pi*f) * sin(2*pi*f*2n + phi + pi*f)
so the envelope scales by |cos(pi*f)|, phase-independently, for every f.

At f = 0.25 the output is an alternating sequence whose amplitude is
|cos(pi/4)| * |sin(phi + pi/4)| -> phase dependent, ceiling 70.71%, floor 0.

Estimator: project onto cos/sin at the OUTPUT frequency (2f, folded if >0.5),
which is the only estimator that stays valid when samples/period gets small.
"""
import numpy as np

W, H = 1280, 720
xx = np.mgrid[0:H, 0:W].astype(np.float64)[1]
AMP = 50.0


def box_both(c):
    c = c[: c.shape[0] // 2 * 2, : c.shape[1] // 2 * 2]
    return c.reshape(c.shape[0] // 2, 2, c.shape[1] // 2, 2).mean(axis=(1, 3))


def project(prof, freq):
    """Amplitude of the component at `freq` cyc/sample, by least squares."""
    n = np.arange(len(prof))
    w = 2 * np.pi * freq * n
    A = np.column_stack([np.ones_like(w), np.cos(w), np.sin(w)])
    coef, *_ = np.linalg.lstsq(A, prof - prof.mean(), rcond=None)
    return float(np.hypot(coef[1], coef[2]))


PHASES = np.linspace(0, 2 * np.pi, 48, endpoint=False)
print(f"{'P':>5} {'f':>7} {'out freq':>9} {'measured':>9} {'|cos(pi f)|':>11} {'diff':>7}")
worst = 0.0
for P in (5, 6, 7, 8, 10, 12, 15, 16, 24, 32, 48, 64, 96):
    f = 1.0 / P
    out_f = 2 * f if 2 * f <= 0.5 else 1 - 2 * f        # folding
    rets = []
    for ph in PHASES:
        c = 128 + AMP * np.sin(2 * np.pi * xx / P + ph)
        a0 = project(c.mean(axis=0), f)
        rets.append(100 * project(box_both(c).mean(axis=0), out_f) / a0)
    meas = float(np.mean(rets))
    pred = 100 * abs(np.cos(np.pi * f))
    worst = max(worst, abs(meas - pred))
    print(f"{P:>5} {f:>7.4f} {out_f:>9.4f} {meas:>8.2f}% {pred:>10.2f}% {meas-pred:>+6.2f} "
          f"(sd {np.std(rets):.2f})")
print(f"  -> max deviation from |cos(pi f)|: {worst:.3f} pp")

print("\n=== at exactly chroma Nyquist (f=0.25): phase-dependent, ceiling 70.71% ===")
P = 4.0
vals = []
for ph in PHASES:
    c = 128 + AMP * np.sin(2 * np.pi * xx / P + ph)
    # output is (-1)^n * cos(pi f) * sin(phi + pi f): alternating, so amplitude at
    # 0.5 cyc/sample. Envelope = |cos(pi f) * sin(phi + pi f)|.
    vals.append(100 * abs(np.cos(np.pi / 4) * np.sin(ph + np.pi / 4)))
print(f"  analytic envelope: min {min(vals):.2f}%  max {max(vals):.2f}%  "
      f"mean {np.mean(vals):.2f}%  spread {max(vals)-min(vals):.2f} pp")
# numeric cross-check on the actual boxed array via its two alternating levels
num = []
for ph in PHASES:
    c = 128 + AMP * np.sin(2 * np.pi * xx / P + ph)
    prof = box_both(c).mean(axis=0)
    num.append(100 * (prof.max() - prof.min()) / AMP)     # alternating seq: ptp = 2*env
print(f"  numeric  (ptp/AMP): min {min(num):.2f}%  max {max(num):.2f}%  "
      f"mean {np.mean(num):.2f}%")
print(f"  check ceiling <= 70.71%: {max(num) <= 70.72}")
print(f"  check ceiling <= 100%  : {max(num) <= 100.01}  <- amplification is impossible")

print("\n=== above Nyquist: amplitude law still holds, frequency folds ===")
for P in (3.6, 3.2, 3.0, 2.8, 2.5):
    f = 1.0 / P
    out_f = 1 - 2 * f
    rets = []
    for ph in PHASES:
        c = 128 + AMP * np.sin(2 * np.pi * xx / P + ph)
        rets.append(100 * project(box_both(c).mean(axis=0), out_f) / AMP)
    pred = 100 * abs(np.cos(np.pi * f))
    print(f"  P={P:>4} f={f:.4f} folds to {out_f:.4f} | measured {np.mean(rets):6.2f}% "
          f"pred {pred:6.2f}% diff {np.mean(rets)-pred:+.2f}")
