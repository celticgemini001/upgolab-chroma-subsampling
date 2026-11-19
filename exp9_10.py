import numpy as np, subprocess, os, json

W, H = 1280, 720
os.makedirs("src", exist_ok=True); os.makedirs("enc", exist_ok=True)
rng = np.random.default_rng(23)
yy, xx = np.mgrid[0:H, 0:W].astype(np.float64)

def write_ref(name, y, cb, cr):
    buf = np.dstack([np.clip(y,16,235), np.clip(cb,16,240), np.clip(cr,16,240)]).astype(np.uint8)
    raw = f"src/{name}.yuv"; buf.tofile(raw)
    f = f"src/{name}_ref.mkv"
    subprocess.run(["ffmpeg","-y","-v","error","-f","rawvideo","-pix_fmt","yuv444p",
                    "-s",f"{W}x{H}","-i",raw,"-c:v","ffv1",f], check=True)
    d = subprocess.run(["ffmpeg","-v","error","-i",f,"-pix_fmt","yuv444p","-f","rawvideo","-"],
                       capture_output=True, check=True).stdout
    assert d == buf.tobytes(), f"{name}: ref not lossless"
    return np.frombuffer(d, np.uint8).reshape(H, W, 3).astype(np.float64)

def decode(path):
    raw = subprocess.run(["ffmpeg","-v","error","-i",path,"-pix_fmt","yuv444p",
                          "-f","rawvideo","-"], capture_output=True, check=True).stdout
    return np.frombuffer(raw, np.uint8).reshape(H, W, 3).astype(np.float64)

def psnr(a, b):
    m = np.mean((a-b)**2)
    return 99.0 if m <= 1e-12 else 10*np.log10(255.0**2/m)

def hf_energy(c):
    """mean |Laplacian| -- high-frequency energy of one plane"""
    return float(np.mean(np.abs(4*c[1:-1,1:-1] - c[:-2,1:-1] - c[2:,1:-1] - c[1:-1,:-2] - c[1:-1,2:])))

# ---------- content for the QP-offset sweep ----------
tex = np.zeros((H, W))
for f, a in ((3,28),(7,16),(13,10),(29,6),(61,3),(127,2)):
    tex += a*np.sin(2*np.pi*(f*xx/W + 0.37*f*yy/H) + rng.uniform(0,6.28))
PHOTO = write_ref("photo", 138 + tex,
                  108 + 36*np.tanh((xx-640)/120.0), 128 + 26*np.sin(2*np.pi*yy/H*1.5))

cb = np.full((H,W),128.0); cr = np.full((H,W),128.0)
COLS = list(range(80, W-40, 160))
for i,x0 in enumerate(COLS):
    cb[:,x0] = 128 + (58 if i%2==0 else 44); cr[:,x0] = 128 - (38 if i%2==0 else 30)
STROKE = write_ref("stroke", np.full((H,W),128.0), cb, cr)

CBARS = write_ref("cbars", np.full((H,W),128.0),
                  128 + 58*np.sign(np.sin(2*np.pi*xx/16.0)),
                  128 + 48*np.sign(np.cos(2*np.pi*xx/24.0)))

# ---------- content for the AXIS experiment ----------
# vdetail: chroma alternates along Y (horizontal bands) -> detail lives on the VERTICAL axis
vy = 128 + 55*np.sign(np.sin(2*np.pi*yy/16.0))
vx = 128 + 30*np.cos(2*np.pi*yy/22.0)
VDETAIL = write_ref("vdetail", 128 + 14*np.sin(2*np.pi*xx/300.0), vy, vx)
# hdetail: same amplitude, chroma alternates along X (vertical bars) -> detail on HORIZONTAL axis
hx = 128 + 55*np.sign(np.sin(2*np.pi*xx/16.0))
hz = 128 + 30*np.cos(2*np.pi*xx/22.0)
HDETAIL = write_ref("hdetail", 128 + 14*np.sin(2*np.pi*yy/300.0), hx, hz)

REFS = {"photo":PHOTO,"stroke":STROKE,"cbars":CBARS,"vdetail":VDETAIL,"hdetail":HDETAIL}
def psnr_c(name, d):
    r = REFS[name]
    return psnr(np.dstack([r[...,1],r[...,2]]), np.dstack([d[...,1],d[...,2]]))

# ---------- Exp A: chroma QP offset sweep at 4:2:0 ----------
OFFS = [-16,-12,-8,-6,-3,0,3,6,12]
A = {}
for name in ("photo","stroke","cbars"):
    rows = []
    for off in OFFS:
        p = f"enc/{name}_o{off}.mp4"
        subprocess.run(["ffmpeg","-y","-v","error","-i",f"src/{name}_ref.mkv",
                        "-pix_fmt","yuv420p","-c:v","libx264","-crf","20","-preset","slow",
                        "-threads","1","-chromaoffset",str(off),p], check=True)
        d = decode(p)
        rows.append(dict(off=off, bytes=os.path.getsize(p), psnr_c=round(psnr_c(name,d),4),
                         psnr_y=round(psnr(REFS[name][...,0], d[...,0]),4)))
    A[name] = rows
    print(name, [(r["off"], r["bytes"], r["psnr_c"]) for r in rows])

# ---------- Exp B: axis asymmetry, 4:2:0 vs 4:2:2 vs 4:4:4 ----------
FMTS = [("yuv420p","high"),("yuv422p","high422"),("yuv444p","high444")]
B = {}
for name in ("vdetail","hdetail"):
    rows = []
    for fmt, prof in FMTS:
        p = f"enc/{name}_{fmt}.mp4"
        subprocess.run(["ffmpeg","-y","-v","error","-i",f"src/{name}_ref.mkv",
                        "-pix_fmt",fmt,"-profile:v",prof,"-c:v","libx264","-crf","20",
                        "-preset","slow","-threads","1",p], check=True)
        d = decode(p)
        rows.append(dict(fmt=fmt, bytes=os.path.getsize(p), psnr_c=round(psnr_c(name,d),4),
                         hf_c=round(hf_energy(d[...,1]),4),
                         ref_hf=round(hf_energy(REFS[name][...,1]),4)))
    B[name] = rows
    print(name, [(r["fmt"], r["bytes"], r["psnr_c"], r["hf_c"], r["ref_hf"]) for r in rows])

json.dump(dict(offset=A, axis=B, ref_hf_photo=hf_energy(PHOTO[...,1]),
               stroke_cols=len(COLS)), open("res.json","w"), indent=1)
print("saved res.json")
