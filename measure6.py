import numpy as np, subprocess, os, json

W, H = 1280, 720
R = np.load("refs2.npy", allow_pickle=True).item()

def dec420(p):
    raw = subprocess.run(["ffmpeg","-v","error","-i",p,"-pix_fmt","yuv420p",
                          "-f","rawvideo","-"], capture_output=True, check=True).stdout
    cw, ch = W//2, H//2
    y = np.frombuffer(raw[:W*H], np.uint8).reshape(H, W)
    u = np.frombuffer(raw[W*H:W*H+cw*ch], np.uint8).reshape(ch, cw)
    v = np.frombuffer(raw[W*H+cw*ch:], np.uint8).reshape(ch, cw)
    return y.astype(float), u.astype(float), v.astype(float)

def box_both(c):
    h, w = c.shape
    return c[:h-h%2, :w-w%2].reshape(h//2,2,w//2,2).mean(axis=(1,3))

def psnr(a, b):
    m = np.mean((a-b)**2)
    return 99.0 if m <= 1e-12 else 10*np.log10(255.0**2/m)

def cpsnr(n, u, v):
    return (psnr(box_both(R[n][1]), u) + psnr(box_both(R[n][2]), v))/2

OFFS = [-24,-20,-16,-12,-10,-8,-6,-4,-2,0,2,4,6,8,10,12,16,20]
CRFS = [18, 20, 23]
out = {}
for crf in CRFS:
    for n in ("photo","chroma"):
        rows = []
        for off in OFFS:
            p = f"enc2/{n}_c{crf}_o{off}.mp4"
            subprocess.run(["ffmpeg","-y","-v","error","-i",f"src/{n}_ref.mkv",
                "-pix_fmt","yuv420p","-c:v","libx264","-crf",str(crf),"-preset","slow",
                "-threads","1","-chromaoffset",str(off),p], check=True)
            y,u,v = dec420(p)
            rows.append(dict(off=off, bytes=os.path.getsize(p),
                             c=round(cpsnr(n,u,v),4),
                             y_=round(psnr(R[n][0], y),4)))
        out[f"{crf}_{n}"] = rows
        # find the plateau (clamp) and the argmax
        best = max(rows, key=lambda r: r["c"])
        b0 = [r for r in rows if r["off"]==0][0]
        # bytes identical => x264 clamped the offset
        same_bytes = [r["off"] for r in rows if r["bytes"] == rows[0]["bytes"]]
        print(f"CRF{crf} {n:6s} off0 {b0['c']:7.3f}dB {b0['bytes']:7d}B | "
              f"peak off={best['off']:+3d} {best['c']:7.3f}dB {best['bytes']:7d}B "
              f"({100*(best['bytes']/b0['bytes']-1):+5.1f}% bytes) outside={100*(best['bytes']/b0['bytes']-1):+5.1f}% | "
              f"gain {best['c']-b0['c']:+.3f}dB | luma {b0['y_']:.3f}->{best['y_']:.3f}")
json.dump(out, open("res6.json","w"), indent=1)
print("saved res6.json")
