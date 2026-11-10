import argparse, pickle, subprocess
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont

p = argparse.ArgumentParser()
p.add_argument("--run", required=True); p.add_argument("--out", required=True); p.add_argument("--title", default="")
p.add_argument("--fps", type=int, default=15); p.add_argument("--hold", type=float, default=1.0)
a = p.parse_args()
d = pickle.load(open(a.run, "rb")); frames = d["frames"][0]; o = d["out"]
succ = [int(s) for s in np.where(o["matches"][0])[0]]
arr0 = np.array(frames)
ink = (arr0.min(-1) < 240).any(0)
ys, xs = np.where(ink)
pad = int(0.04 * arr0.shape[1])
y0, y1 = max(0, ys.min() - pad), min(arr0.shape[1], ys.max() + pad); x0, x1 = max(0, xs.min() - pad), min(arr0.shape[2], xs.max() + pad)
frames = [f[y0:y1, x0:x1] for f in frames]; del arr0
H, W = frames[0].shape[:2]; H -= H % 2; W -= W % 2
frames = [f[:H, :W] for f in frames]
fnt = lambda s: ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", s)
big, small = fnt(max(18, H // 14)), fnt(max(12, H // 36))
out = []
n = len(frames)
b0 = max(6, H // 60)
for t, f in enumerate(frames):
    im = Image.fromarray(f).convert("RGB"); dr = ImageDraw.Draw(im)
    done = sum(s <= t for s in succ)
    y = H - max(10, H // 40)
    dr.rectangle([0, y + 4, int(W * t / (n - 1)), H], fill=(120, 170, 235))
    for s in succ: dr.rectangle([int(W * s / (n - 1)) - 2, y - 2, int(W * s / (n - 1)) + 2, H], fill=(0, 150, 0))
    hit = t in succ
    dr.text((8, 6), f"{a.title}  step {t}/{n - 1}  successes {done}/{len(succ)}", fill=(0, 0, 0), font=small, stroke_width=2, stroke_fill=(255, 255, 255))
    arr = np.array(im); out += [arr] * (int(a.hold * a.fps) if hit else 1)
Path(a.out).parent.mkdir(parents=True, exist_ok=True)
pr = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(a.fps), "-i", "-",
                       "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", a.out], stdin=subprocess.PIPE)
for arr in out: pr.stdin.write(arr.tobytes())
pr.stdin.close(); pr.wait()
Image.fromarray(out[succ[0]] if succ else out[0]).save(str(Path(a.out).with_suffix(".png")))
print("frames", len(out), "successes", succ, "size", W, H)
