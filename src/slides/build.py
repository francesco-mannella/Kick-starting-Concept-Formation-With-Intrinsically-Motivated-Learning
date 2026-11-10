import json, glob, os
from pathlib import Path
from PIL import Image
H = Path(os.path.expanduser('~/tmp/kss/fig567_fric')); OUT = H / 'slides45_47'
W = H / 'demos/work'
SRC = {45: ('seed_4/parasite_004', 'best_blue-cube', 'final_e06_g04_r0_*.png'),
       46: ('seed_6/parasite_006', 'best_red-triangle', 'final_e04_g00_r0_*.png'),
       47: ('seed_6/parasite_006', 'best_green-stick', 'final_e11_g02_r0_*.png')}
# element: (objectId, kind, crop L,R,T,B of the old figure is only used to classify; box = (tx,ty,w,h) in EMU)
E = {
 45: [("h39bceb20c3d3c807_0_1","start",273138,502390,1554476,1362354),("h39bceb20c3d3c807_0_18","end",1741728,502388,882600,1362354),
      ("h39bceb20c3d3c807_0_2","traj",3345210,954550,1468328,1293090),("h39bceb20c3d3c807_0_3","map",574729,2387436,2735416,2617900),
      ("h39bceb20c3d3c807_0_4","g_fov",4045526,3543301,1711100,1600200),("h39bceb20c3d3c807_0_8","g_pro",7639118,3538632,1381300,1600200),
      ("h39bceb20c3d3c807_0_9","g_som",5938334,3543301,1554476,1600200),
      ("h39bceb20c3d3c807_0_5","p_fov",4736623-320963,2247550+164675,980498,1131000),("h39bceb20c3d3c807_0_12","p_pro",7935829,2306988+96275,938758,1131000),
      ("h39bceb20c3d3c807_0_15","p_som",6159083+17075,2317828+89688,1072950,1110000)],
 46: [("h39bceb20c3d3c807_0_26","start",0,442767,1567500,1366400),("h39bceb20c3d3c807_0_25","end",1629800,449597,1103700,1366400),
      ("h39bceb20c3d3c807_0_33","traj",3538250,583996,1567500,1268700),("h39bceb20c3d3c807_0_27","map",382401,2156464,3157026,2987100),
      ("h39bceb20c3d3c807_0_28","g_pro",7370049,3235078,1567500,1770200),("h39bceb20c3d3c807_0_30","g_som",5678038,3267250,1810112,1738000),
      ("h39bceb20c3d3c807_0_31","g_fov",3773686,3267250,1880376,1738000),
      ("h39bceb20c3d3c807_0_37","p_pro",7008281+666950,2077196+231313,957150,1189400),("h39bceb20c3d3c807_0_40","p_som",6248533-215050,2362944-54438,957150,1189400),
      ("h39bceb20c3d3c807_0_43","p_fov",5107722-799850,2083093+231326,957150,1177500)],
 47: [("h39bceb20c3d3c807_0_51","start",0,453491,1673350,1362400),("h39bceb20c3d3c807_0_65","end",1673352,453491,1417300,1362400),
      ("h39bceb20c3d3c807_0_52","traj",2969650,1209298,1545728,1362400),("h39bceb20c3d3c807_0_55","map",129056,2450727,2840584,2686800),
      ("h39bceb20c3d3c807_0_50","g_som",5717401,3537338,1673352,1600200),("h39bceb20c3d3c807_0_53","g_fov",3980050,3537325,1737360,1600200),
      ("h39bceb20c3d3c807_0_60","g_pro",7726688,3537325,1417320,1600200),
      ("h39bceb20c3d3c807_0_49","p_fov",4428568,2571751,885600,1066000),("h39bceb20c3d3c807_0_54","p_pro",8056246,2571750,758216,1066000),
      ("h39bceb20c3d3c807_0_57","p_som",6090938,2568125,1087238,1073200)],
}
# coarse windows (fractions of the 2400x3200 figure) around each panel; tight bbox of non-white inside
WIN = {"start": (0.045, 0.045, 0.13, 0.095), "end": (0.18, 0.045, 0.265, 0.095), "traj": (0.37, 0.05, 0.98, 0.43),
       "map": (0.02, 0.23, 0.33, 0.49), "p_pro": (0.05, 0.54, 0.29, 0.73), "p_som": (0.37, 0.55, 0.64, 0.70), "p_fov": (0.69, 0.52, 0.99, 0.75),
       "g_pro": (0.05, 0.80, 0.30, 1.0), "g_som": (0.37, 0.80, 0.64, 1.0), "g_fov": (0.71, 0.80, 0.98, 1.0)}
def tight(im, box, pad=6):
    W_, H_ = im.size; x0, y0, x1, y1 = [int(v * s) for v, s in zip(box, (W_, H_, W_, H_))]
    c = im.crop((x0, y0, x1, y1)).convert("RGB"); import numpy as np
    a = np.array(c); ink = (a.min(-1) < 245); ys, xs = np.where(ink)
    if len(ys) == 0: return c
    return c.crop((max(0, xs.min() - pad), max(0, ys.min() - pad), min(c.width, xs.max() + pad), min(c.height, ys.max() + pad)))
manifest = {}
for s, (run, wd, pat) in SRC.items():
    f = glob.glob(str(W / run / 'simulations' / run.split('/')[1] / 'storage' / wd / 'final' / 'demos' / pat)); assert len(f) == 1, (s, f)
    fig = Image.open(f[0]); manifest[s] = dict(figure=f[0], elements=[])
    for oid, kind, tx, ty, w, h in E[s]:
        c = tight(fig, WIN[kind]) if kind not in ('start', 'end') else fig.crop(tuple(int(v * q) for v, q in zip(WIN[kind], (2400, 3200, 2400, 3200)))).convert('RGB'); scale = min(w / c.width, h / c.height)
        px = 3  # output pixels per ~ 1/3 of source scale; keep the source resolution up to box size
        outw, outh = 1200, int(1200 * h / w)
        canvas = Image.new("RGB", (outw, outh), (255, 255, 255))
        k = min(outw / c.width, outh / c.height); cc = c.resize((max(1, int(c.width * k)), max(1, int(c.height * k))), Image.LANCZOS)
        canvas.paste(cc, ((outw - cc.width) // 2, (outh - cc.height) // 2))
        name = f"s{s}_{kind}.png"; canvas.save(OUT / name)
        manifest[s]["elements"].append(dict(old=oid, kind=kind, file=name, translateX=tx, translateY=ty, width=w, height=h))
json.dump(manifest, open(OUT / 'manifest.json', 'w'), indent=1)
# mock composites at 1600x900 (EMU 9144000 x 5143500)
for s in E:
    m = Image.new("RGB", (1600, 900), (255, 255, 255)); r = 1600 / 9144000
    for e in manifest[s]["elements"]:
        im = Image.open(OUT / e["file"]).resize((max(1, int(e["width"] * r)), max(1, int(e["height"] * r))))
        m.paste(im, (int(e["translateX"] * r), int(e["translateY"] * r)))
    m.save(OUT / f"mock_s{s}.png")
print("ok")
