import numpy as np, subprocess, os, json

W, H = 1280, 720
yy, xx = np.mgrid[0:H, 0:W].astype(np.float64)
rng = np.random.default_rng(41)
AMP = 50.0

def write_ref(name, y, cb, cr):
    yy_ = np.clip(y,16,235).astype(np.uint8); uu_ = np.clip(cb,16,240).astype(np.uint8)
    vv_ = np.clip(cr,16,240).astype(np.uint8)
    raw = f"src/x_{name}.yuv"
    np.concatenate([yy_.ravel(), uu_.ravel(), vv_.ravel()]).tofile(raw)
    f = f"src/x_{name}_ref.mkv"
    subprocess.run(["ffmpeg","-y","-v","error","-f","rawvideo","-pix_fmt","yuv444p",
                    "-s",f"{W}x{H}","-i",raw,"-c:v","ffv1",f], check=True)
    return f

def decode(p, w, h, n):
    raw = subprocess.run(["ffmpeg","-v","error","-i",p,"-pix_fmt","yuv444p",
                          "-f","rawvideo","-"], capture_output=True, check=True).stdout
    a = np.frombuffer(raw, np.uint8).reshape(3, h, w).astype(float)
    return a

def amp_x(c):
    p = c.mean(axis=0); return float(np.percentile(p,99)-np.percentile(p,1))
def amp_y(c):
    p = c.mean(axis=1); return float(np.percentile(p,99)-np.percentile(p,1))

PERIODS = [6, 8, 12, 16, 24, 32, 48]
FMTS = [("yuv420p","high"),("yuv422p","high422"),("yuv444p","high444")]

rows = []
for P in PERIODS:
    for axis in ("X","Y"):
        coord = xx if axis == "X" else yy
        # luma carries an unrelated, easy gradient so the encoder is not luma-starved
        y = 128 + 12*np.sin(2*np.pi*(xx/W)*2.0)
        cb = 128 + AMP*np.sin(2*np.pi*coord/P)
        ref = write_ref(f"{axis}{P}", y, cb, np.full((H,W),128.0))
        A = amp_x if axis == "X" else amp_y
        rec = dict(P=P, axis=axis)
        for fmt, prof in FMTS:
            p = f"enc2/x_{axis}{P}_{fmt}.mp4"
            subprocess.run(["ffmpeg","-y","-v","error","-i",ref,"-pix_fmt",fmt,
                            "-profile:v",prof,"-c:v","libx264","-crf","18","-preset","slow",
                            "-threads","1",p], check=True)
            d = decode(p, W, H, 3)
            rec[fmt] = round(100.0*A(d[1])/AMP, 2)
            rec[fmt+"_b"] = os.path.getsize(p)
        rows.append(rec)
        print(f"P={P:>3} along {rec['axis']}: 420 {rec['yuv420p']:6.2f}%  "
              f"422 {rec['yuv422p']:6.2f}%  444 {rec['yuv444p']:6.2f}%  "
              f"| bytes 420 {rec['yuv420p_b']:7d} 422 {rec['yuv422p_b']:7d} "
              f"444 {rec['yuv444p_b']:7d}")
json.dump(rows, open("res8.json","w"), indent=1)
