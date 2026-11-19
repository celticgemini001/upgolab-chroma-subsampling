import numpy as np

W, H = 1280, 720
xx = np.mgrid[0:H, 0:W].astype(float)[1]
AMP = 50.0


def box_h(c):
    return c[:, :c.shape[1]//2*2].reshape(c.shape[0], -1, 2).mean(axis=2)


def box_both(c):
    c = c[:c.shape[0]//2*2, :c.shape[1]//2*2]
    return c.reshape(c.shape[0]//2, 2, c.shape[1]//2, 2).mean(axis=(1, 3))


def amp(prof, period):
    # least-squares amplitude -- exact for a clean sinusoid, any phase.
    # Needs >=4 samples per period; below that cos/sin are collinear and the
    # fit explodes instead of failing, so guard it.
    assert len(prof) / period >= 4, (len(prof), period)
    n = np.arange(len(prof))
    w = 2*np.pi*n/period
    A = np.column_stack([np.ones_like(w), np.cos(w), np.sin(w)])
    coef, *_ = np.linalg.lstsq(A, prof - prof.mean(), rcond=None)
    return float(np.hypot(coef[1], coef[2]))


for P in (5, 6, 8, 12, 16, 24, 48, 96):
    # after the box the sinusoid survives at period P/2 OUTPUT pixels, so
    # measure there: input and output then have the same samples/period.
    r422, r420 = [], []
    for ph in np.linspace(0, 2*np.pi, 48, endpoint=False):
        c = 128 + AMP*np.sin(2*np.pi*xx/P + ph)
        a0 = amp(c.mean(axis=0), P)
        r422.append(100*amp(box_h(c).mean(axis=0), P/2)/a0)
        r420.append(100*amp(box_both(c).mean(axis=0), P/2)/a0)
    print(f"P={P:>3} f={1/P:.4f}  422 {np.mean(r422):6.2f}%  420 {np.mean(r420):6.2f}%  "
          f"|cos(pi f)| {100*abs(np.cos(np.pi/P)):6.2f}%")
