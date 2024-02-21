# %%
import os, glob
import sys
import regex as re
from shutil import copyfile
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.collections import LineCollection
from matplotlib.colors import (
    LinearSegmentedColormap,
    ListedColormap,
    )
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN, KMeans
from shapely.geometry import LineString
import pandas as pd
import seaborn as sns
from PIL import Image

#matplotlib.use("QtAgg")

rng = np.random.RandomState(1)

# Add parent directory to Python module path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import SMGraphs
import params
from SMMain import Main

# %%


internal_side = int(np.sqrt(params.internal_size))
visual_side = int(np.sqrt(params.visual_size / 3))


# %%
def manage_fonts():

    SMALL_SIZE = 6
    MEDIUM_SIZE = 11
    BIGGER_SIZE = 12

    plt.rc('font', size=MEDIUM_SIZE)          # controls default text sizes
    plt.rc('axes', titlesize=MEDIUM_SIZE)     # fontsize of the axes title
    plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('legend', fontsize=MEDIUM_SIZE)    # legend fontsize
    plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

# %%
def generic_map(data, ax):
    side = internal_side
    size = params.internal_size
    max_explained_var_ratio = 0.9
    pca = PCA(whiten=True, n_components=3, random_state=rng)

    data = data - np.mean(data, 0) / np.std(data, 0)

    pcanalysis = pca.fit(data.T)
    comps = pcanalysis.transform(data.T)
    comps = (comps - comps.min()) / (comps.max() - comps.min())
    ax.imshow(comps.reshape(side, side, -1), aspect="equal")
    ax.set_axis_off()
    return comps


# %%
def map_groups(data, ax):
    side = internal_side
    size = params.internal_size
    max_explained_var_ratio = 0.9
    pca = PCA(whiten=True, n_components=3, random_state=rng)

    data = data - np.mean(data, 0) / np.std(data, 0)
    pcanalysis = pca.fit(data.T)
    comps = pcanalysis.transform(data.T)
    comps = (comps - comps.min()) / (comps.max() - comps.min())
    clustering = KMeans(n_clusters=3).fit(comps)
    vals = clustering.labels_
    vals = vals.reshape(10, 10)

    c = [0, 0.5, 1]
    colors = np.vstack([x.ravel() for x in np.meshgrid(c, c, c)]).T
    colors = colors[:-1]
    c = np.reshape(colors[1:], (5, 5, 3))
    c = np.transpose(c, (1, 0, 2))
    c = np.reshape(c, (25, 3))
    colors[1:, :] = c

    full_palette = LinearSegmentedColormap.from_list("basic", colors)
    ax.imshow(vals, cmap=full_palette, aspect="auto")
    ax.set_axis_off()
    return comps, vals
# %%

# %%
def visual_map(data_v, ax):

    # visual map
    data_v = data_v.reshape(
        visual_side,
        visual_side,
        3,
        internal_side,
        internal_side,
    )
    data_v = data_v.transpose(3, 0, 4, 1, 2)
    data_v = data_v.reshape(
        visual_side * internal_side,
        visual_side * internal_side,
        3,
    )

    ax.imshow((data_v - data_v.min()) / (data_v.max() - data_v.min()))
    ax.set_axis_off()

# %%

def ssensory(data, ax, idx=None):
    if idx is None: 
        ax.clear()
        X, Y = 0, 0 
    else:
        X, Y = idx

    n_sensors = data.shape[0]
    sep = 0.08
    seg = n_sensors//8
    ss = np.array([
            [-0.3660254, 1.3660254],
            [-0.8660254, 0.5      ],
            [-0.       , 0.       ], 
            [ 0.       , 0.       ],
            [ 0.8660254, 0.5      ],
            [ 0.3660254, 1.3660254],
        ]) * 0.45

    supper = ss*(1 - sep*2) - [0, 5*sep/16] 
    slower = ss*(1 + sep*2) + [0, 5*sep/16] 

    supperl = LineString(supper)
    lng = supperl.length
    supperl = np.vstack(
        [np.hstack(supperl.interpolate(x).xy) for x in np.linspace(0, lng, n_sensors//2)]
    )
    
    slowerl = LineString(slower)
    lng = slowerl.length
    slowerl = np.vstack(
        [np.hstack(slowerl.interpolate(x).xy) for x in np.linspace(0, lng, n_sensors//2)]
    )

    lower = np.hstack([
            data[(1*seg):(2*seg)][::-1],            
            data[(0*seg):(1*seg)][::-1],            
            data[(7*seg):(8*seg)][::-1],            
            data[(6*seg):(7*seg)][::-1],            
            ])
    upper = np.hstack([
            data[(2*seg):(3*seg)],            
            data[(3*seg):(4*seg)],            
            data[(4*seg):(5*seg)],            
            data[(5*seg):(6*seg)],            
            ])


    base = np.array([X, Y]) + [0.5, 0.2]
    
    ax.plot(*(ss + base).T, c="#b77", 
            lw=160*sep if idx is None else 5*sep, 
            zorder=10) 
    ax.plot(*(ss + base).T, c="#bbbbbb", 
            lw=560*sep if idx is None else 5*sep, 
            zorder=0) 
    
    ax.scatter(*(supperl + [base + [0, sep]]).T,
            c="black",
            s=300 * upper if idx is None else 25*upper,
            zorder=30,
    )
    ax.scatter(*(slowerl + [base - [0, sep]]).T,
            c="black",
            s=300 * lower if idx is None else 25*lower,
            zorder=30,
    )

    if idx is None:
        ax.set_axis_off()

        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])


# ax=plt.subplot(111, aspect="auto");ssensory(np.random.rand(40), ax ); plt.show()

# %%
def ssensory_map(data, ax, n_sensors=40):
    data = (data - data.min(0).reshape(1, -1)) / (data.max() - data.min())

    # visual map
    data = (data - data.min(0).reshape(1, -1)) / (data.max() - data.min())
    data = data.reshape(n_sensors, internal_side, internal_side)
    data = data.transpose(0, 2, 1)
    data = data.reshape(n_sensors, -1)


    idx = 0
    for x in range(internal_side):
        for y in range(internal_side - 1, -1, -1):
            ssensory(data[:, idx], ax, [x, y])
            idx += 1

    ax.set_xlim([0, internal_side])
    ax.set_ylim([0, internal_side])
    ax.set_axis_off()


# %%
def proprio(data, ax, idx=None):

    if idx is None:
        ax.clear()
    data = data[3:]
    data = np.array(data)
    data *= -1 
    data[:-2] = np.maximum(-np.pi * 0.5, np.minimum(np.pi * 0.5, data[:-2]))
    data[-1] = np.maximum(0, np.minimum(2 * data[-2], data[-1]))
    data[-2:] = -np.maximum(0, np.minimum(np.pi * 0.5, data[-2:]))
    data *= [1, -1]

    if idx is None:
        x, y = 0, 0
    else:
        x, y = idx

    l = 0.24
    sss = []
    points = np.zeros([3, 2])
    for i, a in enumerate(data):
        ca = np.sum(data[:i]) + a #- np.pi / 2
        point = points[i] + l * np.array([np.cos(ca), np.sin(ca)])
        points[i + 1] = point
    sss.append(points)

    points = np.zeros([3, 2])
    data = -data
    for i, a in enumerate(data):
        ca = np.sum(data[:i]) + a #+ np.pi / 2
        point = points[i] + l * np.array([np.cos(ca), np.sin(ca)])
        points[i + 1] = point
    
    sss.append(points)

    base = np.array([0.3, 0.5]) + [x, y]

    for ss in sss:
        ax.plot(
            *(ss * [0.9, 0.9] + base).T,
            c="black",
            lw=15 if idx is None else 1,
            zorder=0,
        )
        ax.scatter(
            *(ss * [0.9, 0.9] + base).T,
            c="#888",
            s=400 if idx is None else 5,
            zorder=10,
        )

    if idx is None:
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])
        ax.set_axis_off()


# ax=plt.subplot(111, aspect="equal");proprio([0,0,0, -0*np.pi/2, -0*np.pi/4], ax ); plt.show()


# %%
def proprio_map(data, ax):

    # visual map
    data = data.reshape(5, internal_side, internal_side)
    data = data.transpose(0, 2, 1)
    data = data.reshape(5, -1).T

    idx = 0
    for x in range(internal_side):
        for y in range(internal_side - 1, -1, -1):
            proprio(data[idx, :], ax, [x, y])
            idx += 1

    ax.set_xlim([0, internal_side])
    ax.set_ylim([0, internal_side])
    ax.set_axis_off()


# %%
def topological_variance(data, ax, pc=6, k=3):
    side = internal_side
    size = params.internal_size
    max_explained_var_ratio = 0.9
    pca = None
    if pc is not None:
        pca = PCA(whiten=True, n_components=pc, random_state=rng)

    data = data - np.mean(data, 0) / np.std(data, 0)
    if pca is not None:
        pcanalysis = pca.fit(data.T)
        comps = pcanalysis.transform(data.T)
    else:
        comps = data.T
    clustering = KMeans(init="k-means++",n_clusters=k, random_state=rng).fit(comps)
    vals = clustering.labels_
    vals = vals.reshape(10, 10)
    ax.imshow(
        vals,
        cmap=plt.cm.Paired,
    )
    ax.set_axis_off()

    return vals


# %%
def figure_topological_alignement(files, epochs, n_items=6, every=2):

    files = files[: n_items * every : every]
    epochs = epochs[: n_items * every : every]

    data = []
    fig = plt.figure(figsize=(5, 4))
    gs = gridspec.GridSpec(12, n_items)

    for k, f in enumerate(files):
        store = np.load(f, allow_pickle=True)[0]
        col = []
        for h, tmap in enumerate(["visual", "ssensory", "proprio", "policy"]):
            ax = fig.add_subplot(
                gs[(h * 3) : ((h * 3) + 2), k],
                aspect="auto",
            )
            ax.set_axis_off()
            d = generic_map(store[tmap], ax)
            col.append(d)
        data.append(col)
    data = np.array(data).reshape(n_items, 4, -1)

    changes = np.linalg.norm(np.diff(data, axis=0), axis=2)
    max_change = changes.max()
    for h, tmap in enumerate(["visual", "ssensory", "proprio", "policy"]):
        ax = fig.add_subplot(
            gs[((h * 3) + 2), :],
            aspect="auto",
        )
        ax.plot(np.arange(n_items), np.hstack([np.nan, changes[:, h]]), c="black")
        # plt.fill_between(
        #     np.arange(n_items),
        #     np.hstack([np.nan, changes[:, h]]) * 0,
        #     np.hstack([np.nan, changes[:, h]]),
        #     fc="#888888",
        #     ec="#000000",
        # )
        plt.scatter(
            np.arange(n_items),
            np.hstack([np.nan, changes[:, h]]),
            c="black",
        )


        ax.set_xlim(0, n_items)
        ax.set_ylim(-max_change * 0.1, max_change * 1.2)
        ax.set_yticks([0])

        if h < 4 - 1:
            ax.set_xticks([])
        else:
            ax.set_xticks(arange(n_items) + 0.5)
            ax.set_xticklabels(epochs)
            ax.set_xlabel("Epochs")

    fig.tight_layout(pad=0.1)
    fig.savefig("Fig_alignement.png")
    fig.savefig("Fig_alignement.svg")


# %%
def figure_variance(wfile):
    fig = plt.figure(figsize=(8, 2))
    gs = gridspec.GridSpec(1, 4)
    ax1 = fig.add_subplot(gs[:, 0])
    _ = generic_map(wfile, ax1)
    ax2 = fig.add_subplot(gs[:, 1])
    comps, vals = map_groups(wfile, ax2)

    df = np.hstack([comps, vals.reshape(-1, 1)])
    df = pd.DataFrame(df, columns=["p1", "p2", "p3", "cluster"])

    fig.add_subplot(gs[2:])
    ms = pd.melt(df, id_vars=["cluster"])
    sns.boxplot(x="cluster", y="value", hue="variable", data=ms)
    fig2 = plt.figure()
    fig2.add_subplot(111)
    ms = df.groupby(["cluster"], as_index=False).agg(["mean"]).reset_index()
    mse = df.groupby(["cluster"], as_index=False).agg(["std"]).reset_index()
    ms.plot(
        x="cluster",
        y=["p1", "p2", "p3"],
        yerr=mse,
        kind="bar",
        capsize=4,
        rot=0,
    )


# %%
def figure_reprs(rdata, ax):

    palette = sns.color_palette("bright", 4)

    palette_f = np.hstack([palette, 0.3 * np.ones([4, 1])])
    palette_e = np.hstack([palette, np.ones([4, 1])])

    ps = []
    cp = rdata.query(f' modality=="policy"')
   
    radii = [30, 0, 5, 15]

    for i, modality in enumerate(["visual", "ssensory", "proprio", "policy"]):
        cdf = rdata.query(f' modality=="{modality}"')
        ax.plot(
            [cp["y"], cdf["y"]],
            [9 - cp["x"], 9 - cdf["x"]],
            c="black",
            lw=0.3,
            zorder=0,
        )
        p = ax.scatter(
            cdf["y"],
            9 - cdf["x"],
            s=30 + radii[i],
            c=[palette[i]],
            marker=(3+i, i//2), 
            zorder=30 - radii[i],
        )
        ps.append(p)
    
    psi = cdf["psi"].to_numpy()[0]
    
    ax.fill_between(
            x=[11, 12],
            y1=[2,  2],
            y2=2 + 6*np.ones(2)*psi,
            ec=None,
            fc="#a5a8",
            )
    
    ax.plot(
            [11, 12, 12, 11, 11 ], 
            [ 2,  2,  8,  8,  2 ], 
            c="black", lw=0.5)

    ax.set_xlim([-1, 13])
    ax.set_ylim([-1, 10])
    _ = ax.set_xticks(np.arange(10))
    _ = ax.set_yticks(np.arange(10))
    _ = ax.set_yticklabels(np.arange(10)[::-1])

    return ps


# %%
def figure_convergence(rootname, tss, ax):

    data_file = f"{rootname}_data.npy"
    data = np.load(data_file, allow_pickle=True)[0]
    df = aggregate_demo_data(data)


def aggregate_demo_data(data):

    stime = params.stime

    m = data["match_value"]
    mi = data["match_increment"]
    ss = data["ss"].mean(1)

    d = np.vstack(
        [
            np.hstack(
                [
                    m.reshape(-1, 1),
                    mi.reshape(-1, 1),
                    ss.reshape(-1, 1),
                    np.arange(params.stime).reshape(-1, 1),
                    i * np.ones(params.stime).reshape(-1, 1),
                    data[x][:, 0].reshape(-1, 1),
                    data[x][:, 1].reshape(-1, 1),
                ]
            )
            for i, x in enumerate(["v_p", "ss_p", "p_p", "a_p"])
        ]
    )

    df = pd.DataFrame(
        d, columns=["match", "match_increment", "ss", "ts", "modality", "x", "y"]
    )

    df["modality"] = pd.Categorical(df["modality"])
    df["modality"] = df["modality"].cat.rename_categories(
        {
            0: "visual",
            1: "ssensory",
            2: "proprio",
            3: "policy",
        }
    )
    mi = df["match_increment"].to_numpy()
    m = df["match"].to_numpy()
    df["psi"] = (mi / mi.max() > 0.05) * m

    return df


def figure_paths(data_dict, ax):

    stime = params.stime

    df = aggregate_demo_data(data_dict)

    palette = np.array([[0, 0, 0.5], [0, 1, 0], [0.5, 0.5, 0], [0.5, 0, 0]])
    palette_f = np.hstack([palette, 0 * np.ones([4, 1])])
    palette_e = np.hstack([palette, np.ones([4, 1])])

    ps = []
    for c, modality in enumerate(
        [
            "visual",
            "ssensory",
            "proprio",
            "policy",
        ]
    ):
        cdf = df.query(f' modality=="{modality}"')
        p = plt.scatter(
            cdf["y"],
            9 - cdf["x"],
            s=10,
            fc=[palette_f[c]],
            ec=[palette_e[c]],
            lw=2,
        )
        ps.append(p)

    plt.legend(ps, ["visual", "ssensory", "proprio", "policy"])

    ax.set_xlim([-1, 14])
    ax.set_ylim([-1, 10])
    _ = ax.set_xticks(np.arange(10))
    _ = ax.set_yticks(np.arange(10))


# %%
if __name__ == "__main__":

    # %%
    data_files = sorted(glob.glob("storage/*/weig*"))
    main_files = sorted(glob.glob("storage/*/main*npy"))
    data_epochs = [int(re.sub(".*e/(.*)/w.*", r"\1", f)) for f in data_files]
    store = np.load(data_files[-1], allow_pickle=True)[0]

    # %%
    logs = np.load(main_files[-1], allow_pickle=True)[0].logs

    # %%

    rng = np.random.RandomState(1)

    fig = plt.figure(figsize=(6, 3))
    ax = fig.add_subplot(111)
    ax.fill_between(np.arange(800), logs[:800,0], logs[:800,2], fc="#ff8888")
    ax.plot(logs[:800, 1], c="black")
    ax.set_yticks(linspace(0,1,5))
    ax.set_xlabel("Epochs")
    ax.set_ylabel("Competence prediction")
    fig.tight_layout(pad=1)
    fig.savefig("competence_plot.png", dpi=400)
    

    # %%
    data = store["visual"]
    fig = plt.figure(figsize=(16, 8))
    ax1 = fig.add_subplot(121, aspect="equal")
    visual_map(data, ax1)
    ax2 = fig.add_subplot(122, aspect="equal")
    vvals = topological_variance(data, ax2, 3, 3)
    fig.savefig("visual_variance.svg")
    plt.show()

    # %%
    rng = np.random.RandomState(1)
    data = store["ssensory"]
    data = data/data.max()
    fig = plt.figure(figsize=(16, 8))
    ax1 = fig.add_subplot(121, aspect="equal")
    ssensory_map(data, ax1)
    ax2 = fig.add_subplot(122, aspect="equal")
    svals = topological_variance(data, ax2, 20, 6)
    fig.savefig("ssensory_variance.svg")

    # %%
    data = store["proprio"]
    fig = plt.figure(figsize=(16, 8))
    ax1 = fig.add_subplot(121, aspect="equal")
    proprio_map(data, ax1)
    ax2 = fig.add_subplot(122, aspect="equal")
    pvals = topological_variance(data, ax2, 3, 5)
    fig.savefig("proprio_variance.svg")

    # %%
    data = store["policy"]

    fig = plt.figure(figsize=(16, 8))
    ax1 = fig.add_subplot(121, aspect="equal")
    _ = generic_map(data, ax1)
    ax2 = fig.add_subplot(122, aspect="equal")
    avals = topological_variance(data, ax2, 3, 5)
    fig.savefig("policy_variance.svg")

    

    # %%


    fig = plt.figure(figsize=(12, 4))
    gs = GridSpec(2, 6) 
    
    data = store["visual"]
    ax1 = fig.add_subplot(gs[0,0], aspect="equal")
    visual_map(data, ax1)   
    ax5 = fig.add_subplot(gs[1,0], aspect="equal")
    vvals = topological_variance(data, ax5, 5, 5)

    data = store["ssensory"]
    ax2 = fig.add_subplot(gs[0,1], aspect="equal")
    ssensory_map(data, ax2)
    ax6 = fig.add_subplot(gs[1,1], aspect="equal")
    svals = topological_variance(data, ax6, 20, 6)
    
    data = store["proprio"]
    ax3 = fig.add_subplot(gs[0,2], aspect="equal")
    proprio_map(data, ax3)
    ax7 = fig.add_subplot(gs[1,2], aspect="equal")
    pvals = topological_variance(data, ax7, 2, 4)
    
    data = store["policy"]
    ax4 = fig.add_subplot(gs[0,3], aspect="equal")
    _ = generic_map(data, ax4)
    ax8 = fig.add_subplot(gs[1,3], aspect="equal")
    avals = topological_variance(data, ax8, 5, 4)
    


    vals = vvals + 10*svals +100*pvals +1000*avals
    v = np.ones_like(vals)
    idcs = np.arange(len(np.unique(vals)))
    np.random.shuffle(idcs)
    for i, x in enumerate(np.unique(vals)): 
        v[vals==x] = idcs[i]  
    ax10 = fig.add_subplot(gs[:,4:], aspect="equal")
    ax10.imshow(v.reshape(10, 10), cmap= plt.cm.tab20)
    ax10.set_axis_off()
    fig.tight_layout(pad=1)
    fig.savefig("cats.png", dpi=400)
    plt.show()

  # %%
    figure_topological_alignement(data_files, data_epochs, n_items=11, every=7)

    # %%
    # demo fig

    plt.close("all")
    manage_fonts()

    rootname = f"www/demo_{40:04d}"
    data_file = f"{rootname}_data.npy"
    data = np.load(data_file, allow_pickle=True)[0]
    df = aggregate_demo_data(data)

    v_file = f"{rootname}_goal.png"
    v_png = Image.open(v_file)
    t_file = f"{rootname}_touch.png"
    t_png = Image.open(t_file)
    p_file = f"{rootname}_proprio.png"
    p_png = Image.open(p_file)

    gif_file = f"{rootname}.gif"
    gif = Image.open(gif_file)
    w, h = gif.size

    frames = np.arange(0, 10, 2).astype("int") + 31
    n = len(frames)
    print(n)
    fig = plt.figure(figsize=(7, 4))
    gs = gridspec.GridSpec(12, n * 4)

    g_axes = [
        fig.add_axes([0.09, 0.5, 0.15, 0.45]),
        fig.add_axes([0.28, 0.5, 0.15, 0.45]),
        fig.add_axes([0.47, 0.5, 0.15, 0.45]),
        fig.add_axes([0.72, 0.58, 0.22, 0.45], aspect="equal") 
    ]
    env_axes = [
        fig.add_subplot(gs[4:8, (i * 4) : (i * 4) + 4], aspect="equal")
        for i in range(n)
    ]
    repr_axes = [
        fig.add_subplot(gs[8:12, (i * 4) : (i * 4) + 4], aspect="equal")
        for i in range(n)
    ]
    fig.tight_layout(pad=0.2)

    for i, frame in enumerate(frames):
        rax = repr_axes[i]
        eax = env_axes[i]

        eax.clear()
        eax.set_axis_off()
        gif.seek(frame)
        g = gif.crop((100, 160, 160, 220))
        eax.imshow(g)

        rax.clear()
        ps = figure_reprs(df.query(f"ts=={frame}"), rax)
        rax.set_title(f"ts: {frame}")

    for i in range(4): 
        gax = g_axes[i]
        gax.clear()
        gax.set_axis_off()
        if i == 0:
            gax.set_title("Target\nvisual")
            gax.set_aspect("auto")
            gax.imshow(v_png)
        if i == 1:
            gax.set_title("Target\ntouch")
            gax.set_aspect("auto")
            gax.imshow(t_png)
        if i == 2:
            gax.set_title("Target\nproprioception")
            gax.set_aspect("auto")
            gax.imshow(p_png)

    leg = g_axes[-1].legend(
        ps,
        ["Visual", "Touch", "Proprio", "Policy"],
        title="Representations in\nthe internal space",
    )
    leg.get_frame().set_linewidth(0.0)

    fig.savefig("Figure_path.png", dpi=400)
    fig.savefig("Figure_path.svg")

# %%  
    figs = 3, 80, 56
    frames = 80, 33, 50 
    
    fig = plt.figure(figsize=(7,7))

    for i,(frame, fidx) in enumerate(zip(frames, figs)):
        print(frame)
        rootname = f"www/demo_{fidx:04d}"
        data_file = f"{rootname}_data.npy"
        data = np.load(data_file, allow_pickle=True)[0]

        ax1 = fig.add_subplot(3,3,i*3 + 1, aspect="equal")
        gif_file = f"{rootname}.gif"
        gif = Image.open(gif_file)
        w, h = gif.size
        gif.seek(frame)
        g = gif.crop((45, 110, 165, 265))
        ax1.imshow(g)
        ax1.set_ylim(160, 40)
        ax1.set_xticks([])
        ax1.set_yticks([])


        ax2 = fig.add_subplot(3,3,i*3+2, aspect="equal")
        visual = data["v"][frame]
        ax2.imshow(visual.reshape(10, 10, 3))
        ax2.set_xticks([])
        ax2.set_yticks([])

        ax3 = fig.add_subplot(3,3,i*3+3, aspect="equal")
        visual = data["v_g"][frame].numpy()
        ax3.imshow(visual.reshape(10, 10, 3))
        ax3.set_xticks([])
        ax3.set_yticks([])

        fig.tight_layout(pad=1)
        fig.savefig("gen", dpi=400)
