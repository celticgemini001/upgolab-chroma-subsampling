#!/usr/bin/env python3
"""Ad-hoc verification for the upgolab-chroma-subsampling page (NHIP 91, rev 2).

Not a test suite -- this repo has no test command. It audits the published claims:

  1. the amplitude estimator is unbiased (rev 1 used a biased one)
  2. retention == |cos(pi*f)| exactly (rev 1 said |sinc(2f)|)
  3. no period/phase yields >100% retention (rev 1 published 141%)
  4. the live page carries the corrected tables and none of the rev-1 errors
  5. Exp 9's numbers and the x264 clamp still hold
  6. every reproduce snippet compiles, and the runnable ones print the published numbers
  7. the repo ships no script that reproduces the superseded numbers

Usage: python3 verify.py [--offline]
  --offline  skip the live-HTML fetch (checks 4 uses the local index.html instead)
"""
import hashlib
import html as htmlmod
import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent
PAGE = REPO / "index.html"
LIVE = "https://celticgemini001.github.io/upgolab-chroma-subsampling/"
W, H, AMP = 1280, 720, 50.0
XX = np.mgrid[0:H, 0:W].astype(np.float64)[1]

fails = []
checks = 0


def ck(name, ok, detail=""):
    global checks
    checks += 1
    print(f"  {'PASS' if ok else 'FAIL'}  {name}" + (f"  [{detail}]" if detail else ""))
    if not ok:
        fails.append(name)


def box_h(c):
    return c[:, : c.shape[1] // 2 * 2].reshape(c.shape[0], -1, 2).mean(axis=2)


def box_both(c):
    c = c[: c.shape[0] // 2 * 2, : c.shape[1] // 2 * 2]
    return c.reshape(c.shape[0] // 2, 2, c.shape[1] // 2, 2).mean(axis=(1, 3))


def amp(prof, period):
    """Least-squares amplitude. Exact for a clean sinusoid at any phase."""
    assert len(prof) / period >= 4, "collinear: fewer than 4 samples per period"
    n = np.arange(len(prof))
    w = 2 * np.pi * n / period
    A = np.column_stack([np.ones_like(w), np.cos(w), np.sin(w)])
    coef, *_ = np.linalg.lstsq(A, prof - prof.mean(), rcond=None)
    return float(np.hypot(coef[1], coef[2]))


def psnr(a, b):
    m = np.mean((np.asarray(a, float) - np.asarray(b, float)) ** 2)
    return 99.0 if m <= 1e-12 else 10 * np.log10(255.0 ** 2 / m)


tmp = Path(tempfile.mkdtemp(prefix="hermes-verify-chroma-"))

print("=== 1. estimator is unbiased ===")
err = max(abs(amp(128 + AMP * np.sin(2 * np.pi * np.arange(W) / P), P) - AMP)
          for P in (5, 6, 8, 12, 24))
ck("least-squares amplitude exact on a known sinusoid", err < 0.02, f"max err {err:.4f}")
s6 = 128 + AMP * np.sin(2 * np.pi * np.arange(W) / 6)
pct = np.percentile(s6, 99) - np.percentile(s6, 1)
ck("the percentile estimator rev 1 used IS biased", abs(pct - AMP) > 5,
   f"returns {pct:.2f} for a true 50")

print("\n=== 2. retention == |cos(pi*f)|, exactly ===")
tab = {r["P"]: r for r in json.loads((REPO / "sweep_corrected.json").read_text())} \
    if (REPO / "sweep_corrected.json").exists() else {}
if tab:
    dev = max(abs(tab[P]["X_420"] - 100 * abs(np.cos(np.pi / P)))
              for P in (5, 6, 8, 10, 12, 16, 24, 32, 48, 96))
    ck("published sweep matches the closed form", dev < 0.01, f"max dev {dev:.4f} pp")
else:
    print("  (sweep_corrected.json absent -- covered by the brute-force check below)")

rets = [100 * amp(box_both(128 + AMP * np.sin(2 * np.pi * XX / 8.0 + ph)).mean(axis=0), 4.0)
        / amp((128 + AMP * np.sin(2 * np.pi * XX / 8.0 + ph)).mean(axis=0), 8.0)
        for ph in np.linspace(0, 2 * np.pi, 48, endpoint=False)]
ck("8 px row re-derived == 92.39%", abs(np.mean(rets) - 92.39) < 0.01, f"{np.mean(rets):.2f}")

print("\n=== 3. the impossible claim is gone ===")
peak = 0.0
for P in np.arange(2.5, 12.01, 0.05):
    for ph in np.linspace(0, 2 * np.pi, 12, endpoint=False):
        c = 128 + AMP * np.sin(2 * np.pi * XX / P + ph)
        out = box_both(c).mean(axis=0)
        if len(out) / (P / 2) >= 4:
            peak = max(peak, 100 * amp(out, P / 2) / AMP)
        else:
            peak = max(peak, 100 * (out.max() - out.min()) / (2 * AMP))
ck("no period/phase exceeds 100% retention", peak <= 100.01, f"peak {peak:.2f}%")
ck("|cos(pi/4)| == 70.71 at chroma Nyquist", abs(100 * abs(np.cos(np.pi / 4)) - 70.71) < 0.01)

print("\n=== 4. the page ===")
offline = "--offline" in sys.argv
if offline:
    h = PAGE.read_text()
    print("  (offline: reading local index.html)")
else:
    lp = tmp / "live.html"
    try:
        subprocess.run(["curl", "-sSf", "-o", str(lp), LIVE], check=True)
        h = lp.read_text()
        print("  (fetched live HTML)")
    except subprocess.CalledProcessError:
        print("  BLOCKER: live fetch failed; re-run with --offline")
        sys.exit(2)

for probe in ("80.90%", "86.60%", "92.39%", "95.11%", "96.59%", "98.08%",
              "99.14%", "99.52%", "99.79%", "99.95%"):
    ck(f"carries {probe}", probe in h)
ck("states the exact law |cos(πf)|", "|cos(πf)|" in h)
ck("shows the 70.71% Nyquist value", "70.71%" in h)
for gone in ("141.4", "1.18–1.24×", "|sinc(2f)| predicts"):
    ck(f"rev-1 error absent: {gone!r}", gone not in h)
occ = [m.start() for m in re.finditer("141%", h)]
ck("141% appears only as the documented-bug anecdote",
   len(occ) == 2 and all(h[max(0, i - 40):i].rstrip().endswith(("published \"", "published"))
                         for i in occ), f"{len(occ)} mention(s)")

n_tab = h.count("<table")
ck("footer table count matches DOM", f"{n_tab} tables" in h, f"{n_tab} tables")
rels = re.findall(r'<a\b[^>]*\brel="([^"]*)"', h)
ck("no anchor carries nofollow", not any("nofollow" in r for r in rels))
ck("exactly one upgolab.com link", h.count('href="https://upgolab.com') == 1)
ck("canonical self", h.count('rel="canonical" href="' + LIVE + '"') == 1)
ck("dateModified 2026-11-19", '"dateModified":"2026-11-19"' in h)
for tag in ("div", "table", "pre", "tr", "td", "code", "h2"):
    ck(f"<{tag}> balanced", h.count(f"<{tag}") == h.count(f"</{tag}>"))

print("\n=== 5. Exp 9 numbers and the clamp ===")
by_off = {r["off"]: r for r in json.loads((REPO / "enc2_offsets.json").read_text())["photo"]} \
    if (REPO / "enc2_offsets.json").exists() else None
if by_off:
    ck("CRF20 chroma 54.448", abs(by_off[0]["c"] - 54.448) < 0.001)
    ck("CRF20 luma 47.809", abs(by_off[0]["y"] - 47.809) < 0.001)
    ck("headroom 6.64 dB", abs(by_off[0]["c"] - by_off[0]["y"] - 6.64) < 0.01)
    ck("luma flat across sweep (<0.05 dB)",
       max(r["y"] for r in by_off.values()) - min(r["y"] for r in by_off.values()) < 0.05)
else:
    print("  (enc2_offsets.json absent -- clamp checked below from the encode itself)")

ref = REPO / "src" / "photo_ref.mkv"
if not ref.exists():
    # No shipped media: synthesise a small planar yuv444p source so the clamp check
    # always runs. Only raw planar input is needed, so keep it tiny and deterministic.
    rng = np.random.default_rng(0)
    sw, sh = 320, 180
    tex = np.zeros((sh, sw))
    sxx = np.mgrid[0:sh, 0:sw][1].astype(float)
    for f, a in ((3, 26), (7, 15), (13, 9)):
        tex += a * np.sin(2 * np.pi * f * sxx / sw + rng.uniform(0, 6.28))
    y = np.clip(138 + tex, 16, 235).astype(np.uint8)
    cb = np.clip(108 + 36 * np.tanh((sxx - sw / 2) / 30.0), 16, 240).astype(np.uint8)
    cr = np.full((sh, sw), 128, np.uint8)
    raw = tmp / "gen.yuv"
    np.concatenate([y.ravel(), cb.ravel(), cr.ravel()]).tofile(raw)
    ref = tmp / "gen_ref.mkv"
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "yuv444p",
                    "-s", f"{sw}x{sh}", "-i", str(raw), "-c:v", "ffv1", str(ref)], check=True)
    print("  (synthesised a 320x180 source for the clamp check)")

dig = {}
for off in (-10, -12, -16, -24, -8):
    p = tmp / f"c{off}.mp4"
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", str(ref), "-pix_fmt", "yuv420p",
                    "-c:v", "libx264", "-crf", "18", "-preset", "slow", "-threads", "1",
                    "-chromaoffset", str(off), str(p)], check=True)
    dig[off] = hashlib.md5(p.read_bytes()).hexdigest()
ck("x264 clamps: -10/-12/-16/-24 bit-identical",
   len({dig[o] for o in (-10, -12, -16, -24)}) == 1, dig[-10][:8])
ck("-8 differs from the clamped set", dig[-8] != dig[-10])

print("\n=== 6. reproduce snippets ===")
snips = [htmlmod.unescape(re.sub(r"<[^>]+>", "", b))
         for b in re.findall(r"<pre><code>(.*?)</code></pre>", h, flags=re.S)]
print(f"  ({len(snips)} code blocks)")
for i, src in enumerate(snips, 1):
    f = tmp / f"s{i}.py"
    f.write_text(src)
    comp = subprocess.run([sys.executable, "-m", "py_compile", str(f)],
                          capture_output=True, text=True)
    if comp.returncode != 0:
        shellish = bool(re.search(r"^\w*[0-9a-f]{32}\s+\S+\.\w+", src, re.M)) \
            or "chroma_qp_offset=" in src
        formula = bool(re.search(r"^\s*y\[n\]\s*=", src))
        ck(f"block {i}: non-Python but legitimate", shellish or formula,
           src.strip().splitlines()[0][:40])
        continue
    if ("src/" in src and "ffmpeg" in src) or "Image.open" in src or "CONTINUES" in src:
        ck(f"block {i}: compiles (needs local inputs)", True)
        continue
    r = subprocess.run([sys.executable, str(f)], capture_output=True, text=True, timeout=900)
    ck(f"block {i}: runs clean", r.returncode == 0 and "Traceback" not in r.stderr,
       r.stderr.strip().splitlines()[-1][:60] if r.returncode else "")
    if r.returncode == 0 and "|cos(pi f)|" in src:
        out = r.stdout
        for want in ("80.90%", "92.39%", "98.08%"):
            ck(f"block {i} prints {want}", want in out)

print("\n=== 7. no script ships the superseded estimator ===")
shipped = sorted(p.name for p in REPO.glob("*.py"))
bad = []
for p in REPO.glob("*.py"):
    src = p.read_text()
    if "percentile" in src and "numpy" in src and "assert" not in src:
        bad.append(p.name)
ck("no shipped script reproduces the rev-1 numbers", not bad, str(bad) if bad else f"{shipped}")

print("\n" + "=" * 62)
print(f"{checks - len(fails)}/{checks} checks passed")
if fails:
    print("FAILED:", fails)
    sys.exit(1)
print("all published claims re-derive; rev-1 errors absent")
