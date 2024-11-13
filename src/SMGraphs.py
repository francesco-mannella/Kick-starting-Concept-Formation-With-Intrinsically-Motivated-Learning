import glob
import os
import params
from shutil import copyfile,rmtree
import pathlib
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from mkvideo import vidManager


from matplotlib.colors import LinearSegmentedColormap

c = [0, 0.5, 1]
colors = np.vstack([x.ravel() for x in np.meshgrid(c, c, c)]).T
colors = colors[:-1]
c = np.reshape(colors[1:], (5, 5, 3))
c = np.transpose(c, (1, 0, 2))
c = np.reshape(c, (25, 3))
colors[1:, :] = c
full_palette = LinearSegmentedColormap.from_list("basic", colors)

palette = matplotlib.colors.LinearSegmentedColormap.from_list(
    name="custom", colors=[[0, 1, 0], [1, 0, 0], [0, 0, 1]]
)
internal_side = int(np.sqrt(params.internal_size))
visual_side = int(np.sqrt(params.visual_size / 3))

storage_dir = "storage"
site_dir = "www"
os.makedirs(storage_dir, exist_ok=True)
os.makedirs(site_dir, exist_ok=True)

def remove_figs(epoch=0):
    if epoch > 0:
        epoch_dir = f"{storage_dir}/{epoch:06d}"
        os.makedirs(storage_dir, exist_ok=True)
        os.makedirs(epoch_dir, exist_ok=True)

        try:
            copyfile(f"{site_dir}/visual_map.png", f"{epoch_dir}/visual_map.png")
            copyfile(f"{site_dir}/comp_map.png", f"{epoch_dir}/comp_map.png")
            copyfile(f"{site_dir}/log.png", f"{epoch_dir}/log.png")
            copyfile(f"{site_dir}/trajectories.png", f"{epoch_dir}/trajectories.png")
        except OSError:
            pass
    else:
        print("Starting simulation ...")
        if not os.path.exists(f"{site_dir}/blank.gif"):
            blank_video()
        os.makedirs("storage", exist_ok=True)
        for f in glob.glob("storage/*"):
            if os.path.isdir(f):
                rmtree(f)
            else:
                os.remove(f)
        copyfile(f"{site_dir}/blank.gif", f"{site_dir}/tv.gif")
        copyfile(f"{pathlib.Path(__file__).parent.resolve()}/arms.html", f"{site_dir}/arms.html")

        figs = glob.glob(f"{site_dir}/episode*.gif") + glob.glob(f"{site_dir}/*.png")
        for f in figs:
            if os.path.isdir(f):
                rmtree(f)
            else:
                os.remove(f)
        for k in range(params.tests):
            copyfile(f"{site_dir}/blank.gif", f"{site_dir}/episode%d.gif" % k)

        copyfile(f"{site_dir}/blank.gif", f"{site_dir}/visual_map.png")
        copyfile(f"{site_dir}/blank.gif", f"{site_dir}/comp_map.png")
        copyfile(f"{site_dir}/blank.gif", f"{site_dir}/log.png")
        copyfile(f"{site_dir}/blank.gif", f"{site_dir}/trajectories.png")


def trajectories_map(wfile=None):
    if wfile is None: 
        wfile = f"{site_dir}/trajectories.npy"
    data = np.load(wfile, allow_pickle=True)
    cells, stime, _ = data.shape
    side = int(np.sqrt(cells))
    fig = plt.figure(figsize=(8, 8))
    colors = palette(np.linspace(0, 1, stime))
    for cell in range(cells):
        ax = fig.add_subplot(side, side, cell + 1, aspect="equal")

        ax.add_collection(
            LineCollection(
                segments=np.hstack(
                    [
                        data[cell].reshape(-1, 1, 2)[:-1],
                        data[cell].reshape(-1, 1, 2)[1:],
                    ]
                ),
                colors=colors,
            )
        )
        ax.scatter(*data[cell].T, c=palette(np.linspace(0, 1, stime)), alpha=0.1)

        ax.set_xlim([-0.1, np.pi / 2 + 0.1])
        ax.set_ylim([-0.1, np.pi / 2 + 0.1])
        ax.set_xticks([])
        ax.set_yticks([])

    fig.tight_layout(pad=0.0)
    fig.savefig(f"{site_dir}/trajectories.png")


def visual_map(wfile=None):
    if wfile is None: wfile = f"{site_dir}/visual_weights.npy"
    # visual map
    data_v = np.load(wfile, allow_pickle=True)
    data_v = data_v.reshape(visual_side, visual_side, 3, internal_side, internal_side)
    data_v = data_v.transpose(3, 0, 4, 1, 2)
    data_v = data_v.reshape(visual_side * internal_side, visual_side * internal_side, 3)

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, aspect="equal")
    ax.imshow((data_v - data_v.min()) / (data_v.max() - data_v.min()))
    ax.set_axis_off()
    fig.tight_layout(pad=0.0)
    fig.savefig(f"{site_dir}/visual_map.png")
    plt.close("all")


def somatosensory_map(wfile=None):
    if wfile is None: wfile = f"{site_dir}/ssensory_weights.npy"
    # visual map
    data_v = np.load(wfile, allow_pickle=True)
    data_v = data_v.reshape(4, internal_side, internal_side)
    data_v = data_v.transpose(1, 2, 0)
    data_v = data_v.reshape(internal_side, internal_side * 4)

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, aspect="equal")
    ax.imshow((data_v - data_v.min()) / (data_v.max() - data_v.min()))
    ax.set_axis_off()
    fig.tight_layout(pad=0.0)
    fig.savefig(f"{site_dir}/ssensory_map.png")
    plt.close("all")


def comp_map(wfile=None):
    if wfile is None: wfile = f"{site_dir}/comp_grid.npy"
    # comp map
    data_c = np.load(wfile, allow_pickle=True)
    data_c = data_c.reshape(internal_side, internal_side)

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, aspect="equal")
    ax.imshow(data_c, vmin=0, vmax=1)
    ax.set_axis_off()
    fig.tight_layout(pad=0.0)
    fig.savefig(f"{site_dir}/comp_map.png")
    plt.close("all")


def representations_movements(v_r, ss_r, p_r, a_r, name):

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, aspect="equal")
    vm = vidManager(fig, "irep", "irep", duration=50)

    x = np.arange(internal_side)
    grid = np.stack(np.meshgrid(x, x)).reshape(2, -1)

    pv = ax.scatter(*grid, c="green", s=np.ones(params.internal_size))
    pss = ax.scatter(*grid, c="red", s=100 * np.ones(params.internal_size))
    pp = ax.scatter(*grid, c="blue", s=100 * np.ones(params.internal_size))
    pa = ax.scatter(*grid, c="black", s=100 * np.ones(params.internal_size))

    for i, (v, ss, p, a) in enumerate(zip(v_r, ss_r, p_r, a_r)):
        pv.set_sizes(700 * v)
        pss.set_sizes(700 * ss)
        pp.set_sizes(700 * p)
        pa.set_sizes(700 * a)
        ax.set_title("%d" % i)
        vm.save_frame()

    vm.mk_video(name=name, dirname=".")
    plt.close("all")


def blank_video():
    name = "blank"
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, aspect="equal")
    vm = vidManager(fig, "blank", f"{site_dir}/blank", duration=50)

    x = np.arange(internal_side)
    grid = np.stack(np.meshgrid(x, x)).reshape(2, -1)

    ax.set_visible(False)

    for t in range(5):
        vm.save_frame()

    vm.mk_video(name=name, dirname=f"{site_dir}")
    plt.close("all")


def log(wfile=None):
    if wfile is None: wfile = f"{site_dir}/log.npy"
    log = np.load(wfile, allow_pickle=True)
    fig = plt.figure(figsize=(4, 2))
    ax = fig.add_subplot(111)
    stime = len(log)
    ax.fill_between(np.arange(stime), log[:, 0], log[:, 2], fc="red", alpha=0.3)
    ax.plot(np.arange(stime), log[:, 1], c=[0.5, 0, 0])
    ax.set_xlim([-stime * 0.1, stime * 1.1])
    m = log.max()
    ax.set_ylim([-m * 0.1, m * 1.1])
    fig.savefig(f"{site_dir}/log.png")
    plt.close("all")
