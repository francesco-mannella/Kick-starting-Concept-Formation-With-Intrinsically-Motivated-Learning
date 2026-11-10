import json, numpy as np
from PIL import Image
exec(open('compose_slide.py').read().split("PW, PH =")[0])   # E, WIN, tight, m
M = 1_000_000
COL = {"fov": 5.05 * M, "som": 6.70 * M, "pro": 8.35 * M}; F = 1.0 * M; FY = 2.72 * M; GY, GH, GW = 3.80 * M, 1.30 * M, 1.44 * M
BOX = {"start": (0.30 * M, 0.75 * M, 1.05 * M, 0.83 * M), "end": (1.50 * M, 0.75 * M, 1.05 * M, 0.83 * M), "traj": (2.75 * M, 0.60 * M, 1.65 * M, 1.60 * M),
       "map": (0.78 * M, 2.40 * M, 2.30 * M, 2.62 * M)}
for k, c in COL.items():
    BOX["g_" + k[:3] if False else "g_" + {"fov": "fov", "som": "som", "pro": "pro"}[k]] = (c - GW / 2, GY, GW, GH)
def bbox(fig, kind):
    W_, H_ = fig.size; x0, y0, x1, y1 = [int(v * q) for v, q in zip(WIN[kind], (W_, H_, W_, H_))]
    a = np.array(fig.crop((x0, y0, x1, y1))); ys, xs = np.where(a.min(-1) < 245); pad = 6
    return x0 + max(0, xs.min() - pad), y0 + max(0, ys.min() - pad), x0 + min(a.shape[1], xs.max() + pad), y0 + min(a.shape[0], ys.max() + pad)
plan = {}
for s in E:
    fig = Image.open(m[str(s)]['figure']).convert('RGB'); W_, H_ = fig.size; items = []
    pb = {k: bbox(fig, "p_" + k) for k in ("fov", "som", "pro")}; S = max(max(b[2] - b[0], b[3] - b[1]) for b in pb.values()) + 12
    for oid, kind, tx, ty, w, h in E[s]:
        if kind.startswith("p_"):
            k = kind[2:]; b = pb[k]; cx, cy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2; x0, y0, x1, y1 = cx - S / 2, cy - S / 2, cx + S / 2, cy + S / 2
            side = 0.84 * F; bx, by, bw, bh = COL[k] - side / 2, FY + (F - side) / 2, side, side
        else:
            if kind in ("start", "end"): x0, y0, x1, y1 = [int(v * q) for v, q in zip(WIN[kind], (W_, H_, W_, H_))]
            else: x0, y0, x1, y1 = bbox(fig, kind)
            bx, by, bw, bh = BOX[kind]
        cw, ch = x1 - x0, y1 - y0; k_ = min(bw / cw, bh / ch); dw, dh = cw * k_, ch * k_
        items.append(dict(kind=kind, L=x0 / W_, R=1 - x1 / W_, T=y0 / H_, B=1 - y1 / H_, w=int(dw), h=int(dh), x=int(bx + bw / 2 - dw / 2), y=int(by + bh / 2 - dh / 2)))
    plan[s] = items
    # mock
    r = 1600 / 9144000; mk = Image.new('RGB', (1600, 900), (255, 255, 255))
    for it in items:
        c = fig.crop((int(it['L'] * W_), int(it['T'] * H_), int((1 - it['R']) * W_), int((1 - it['B']) * H_))).resize((max(1, int(it['w'] * r)), max(1, int(it['h'] * r))))
        mk.paste(c, (int(it['x'] * r), int(it['y'] * r)))
    from PIL import ImageDraw
    d = ImageDraw.Draw(mk)
    for k in COL: d.rectangle([(COL[k] - F / 2) * r, FY * r, (COL[k] + F / 2) * r, (FY + F) * r], outline=(230, 200, 0), width=3)
    d.rectangle([4.5 * M * r, 0.55 * M * r, 6.1 * M * r, 2.2 * M * r], outline=(150, 150, 150))
    mk.save(f'layout_mock_s{s}.png')
json.dump(plan, open('plan2.json', 'w'), indent=1); print('ok')
