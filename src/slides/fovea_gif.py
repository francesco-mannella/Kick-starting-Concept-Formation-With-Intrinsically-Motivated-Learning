import sys, pickle, io
import numpy as np
import matplotlib; matplotlib.use("agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image
rec_path, ep, out = sys.argv[1], int(sys.argv[2]), sys.argv[3]
R = pickle.load(open(rec_path, "rb"))[ep::18]
fig, (a1, a2) = plt.subplots(1, 2, figsize=(5.6, 2.9), gridspec_kw=dict(width_ratios=[1, 1], wspace=0.35))
a1.set_title("Visual field", fontsize=11); a2.set_title("Fovea", fontsize=11)
im1 = a1.imshow(R[0]["field"], extent=[0, 60, 0, 60], origin="upper", interpolation="nearest")
box = Rectangle((0, 0), 4, 4, fill=False, ec="black", lw=1.4); a1.add_patch(box)
im2 = a2.imshow(R[0]["fovea"], extent=[0, 10, 10, 0], interpolation="nearest")
for a in (a1, a2):
    a.tick_params(labelsize=8)
a1.set_xticks([0, 20, 40, 60]); a1.set_yticks([0, 20, 40, 60]); a2.set_xticks([0, 5, 10]); a2.set_yticks([0, 5, 10])
frames = []
for x in R:
    im1.set_data(x["field"]); im2.set_data(x["fovea"])
    box.set_xy((x["eye"][0] + 5 - 2, x["eye"][1] + 10 - 2))
    fig.canvas.draw()
    frames.append(Image.fromarray(np.asarray(fig.canvas.buffer_rgba())[..., :3]).convert("P", palette=Image.ADAPTIVE, colors=32))
frames[0].save(out, save_all=True, append_images=frames[1:], duration=83, loop=0, optimize=True, disposal=1)
print(len(frames), frames[0].size)
