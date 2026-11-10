import json, numpy as np
from PIL import Image
exec(open('build.py').read().split("manifest = {}")[0])   # E, WIN, tight
m = json.load(open('manifest.json'))
# frame boxes (absolute EMU): tx, ty, w, h  for the three prototype panels per slide, keyed by kind
FR = {45: {"p_fov": (4405988, 2552100, 980400, 970200), "p_pro": (7784963, 2582044, 980400, 970200), "p_som": (6206846, 2577814, 980400, 970200)},
      46: {"p_pro": (7680151, 2434201, 882600, 938100), "p_som": (6048845, 2455779, 882600, 938100), "p_fov": (4382400, 2430476, 882600, 938100)},
      47: {"p_fov": (4398625, 2684175, 885600, 853200), "p_som": (6279250, 2684175, 885600, 853200), "p_pro": (7984900, 2684175, 885600, 853200)}}
PW, PH = 9144000, 5143500; S = 3200 / PW
def to_alpha(img):
    a = np.array(img.convert('RGB')).astype(float); al = 255 - a.min(-1); al = np.clip(al, 0, 255)
    out = np.zeros(a.shape[:2] + (4,)); k = np.maximum(al, 1)[..., None]
    out[..., :3] = np.clip((a - (255 - al)[..., None]) * 255 / k, 0, 255); out[..., 3] = al
    return Image.fromarray(out.astype('uint8'), 'RGBA')
for s, (run, wd, pat) in SRC.items():
    fig = Image.open(m[str(s)]['figure']).convert('RGB')
    canvas = Image.new('RGBA', (3200, 1800), (0, 0, 0, 0))
    endbox = [(w, h) for _, kd, _, _, w, h in E[s] if kd == 'end'][0]
    for oid, kind, tx, ty, w, h in E[s]:
        c = tight(fig, WIN[kind]) if kind not in ('start', 'end') else fig.crop(tuple(int(v * q) for v, q in zip(WIN[kind], (2400, 3200, 2400, 3200)))).convert('RGB')
        if kind in FR[s]:
            ftx, fty, fw, fh = FR[s][kind]; bx, by, bw, bh = ftx + 0.08 * fw, fty + 0.08 * fh, 0.84 * fw, 0.84 * fh
        else: bx, by, bw, bh = tx, ty, w, h
        if kind in ('start', 'end'):
            cx, cy = tx + w / 2, ty + h / 2; bw, bh = endbox; bx, by = cx - bw / 2, cy - bh / 2
        k = min(bw * S / c.width, bh * S / c.height); cc = c.resize((max(1, int(c.width * k)), max(1, int(c.height * k))), Image.LANCZOS)
        px = int((bx + bw / 2) * S - cc.width / 2); py = int((by + bh / 2) * S - cc.height / 2)
        canvas.alpha_composite(to_alpha(cc), (px, py))
    canvas.save(f'comp_s{s}.png')
    bg = Image.new('RGBA', canvas.size, (255, 255, 255, 255)); bg.alpha_composite(canvas); bg.convert('RGB').resize((1600, 900)).save(f'comp_preview_s{s}.png')
print('ok')
