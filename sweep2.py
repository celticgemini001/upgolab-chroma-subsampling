import numpy as np, json

W, H = 1280, 720
yy, xx = np.mgrid[0:H, 0:W].astype(np.float64)
AMP = 50.0

def box_h(c):
    h, w = c.shape; return c[:, :w-w%2].reshape(h, w//2, 2).mean(axis=2)
def box_both(c):
    h, w = c.shape
    return c[:h-h%2, :w-w%2].reshape(h//2,2,w//2,2).mean(axis=(1,3))

def amp_x(c):          # content varies along X -> profile across columns
    p = c.mean(axis=0); return float(np.percentile(p,99) - np.percentile(p,1))
def amp_y(c):          # content varies along Y -> profile across rows
    p = c.mean(axis=1); return float(np.percentile(p,99) - np.percentile(p,1))

PHASES = np.linspace(0, 2*np.pi, 12, endpoint=False)
PERIODS = [4,5,6,8,10,12,16,24,32,48,64,96]

rows = []
for P in PERIODS:
    for axis in ("X", "Y"):
        f_full, f422, f420, sd = [], [], [], []
        for ph in PHASES:
            coord = xx if axis == "X" else yy
            c = 128 + AMP*np.sin(2*np.pi*coord/P + ph)
            A = amp_x if axis == "X" else amp_y
            a0 = A(c)
            a4 = A(box_h(c)) if axis == "X" else a0      # 4:2:2 boxes X only
            a2 = A(box_both(c))                           # 4:2:0 boxes both
            f_full.append(a0); f422.append(a4); f420.append(a2)
        base = np.mean(f_full)
        rows.append(dict(P=P, axis=axis, f=round(1.0/P,4),
                         a444=round(100*base/AMP,2),
                         a422=round(100*np.mean(f422)/base,2),
                         a420=round(100*np.mean(f420)/base,2),
                         sd420=round(float(np.std(100*np.array(f420)/base)),2)))
    rX, rY = rows[-2], rows[-1]
    print(f"P={P:>3} f={rX['f']:.4f} | along X: 422 {rX['a422']:6.2f}%  420 {rX['a420']:6.2f}% "
          f"(sd {rX['sd420']:4.1f}) | along Y: 422 {rY['a422']:6.2f}%  420 {rY['a420']:6.2f}% "
          f"(sd {rY['sd420']:4.1f})")

def cross(axis, key):
    pts = [(r["f"], r[key]) for r in rows if r["axis"] == axis]
    out = []
    for (f1,a1),(f2,a2) in zip(pts, pts[1:]):
        if (a1-50)*(a2-50) <= 0 and a1 != a2:
            out.append(round(f1 + (50-a1)*(f2-f1)/(a2-a1), 4))
    return out

print("50% amplitude crossing, along X  -> 4:2:2:", cross("X","a422"), " 4:2:0:", cross("X","a420"))
print("50% amplitude crossing, along Y  -> 4:2:2:", cross("Y","a422"), " 4:2:0:", cross("Y","a420"))
json.dump(rows, open("sweep_phase.json","w"), indent=1)
