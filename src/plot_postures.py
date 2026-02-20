"""
Visualization module for trajectory animation with proprioceptive,
sensory, and visual weight maps.
"""

import argparse

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Rectangle
from PIL import Image
from scipy.interpolate import splev, splprep

from params import Parameters
from SMGraphs import GraphManager
from SMMain import build_episode_dataset


matplotlib.use("qtagg")


def generate_offset_points(points, distance=0.05):
    points = np.asarray(points, dtype=float)
    n = len(points)

    if n == 0:
        return np.empty((0, 2), dtype=float)
    if n == 1:
        return points.copy()

    p_prev = np.empty_like(points)
    p_next = np.empty_like(points)
    p_prev[0] = points[0]
    p_prev[1:] = points[:-1]
    p_next[-1] = points[-1]
    p_next[:-1] = points[1:]

    d = p_next - p_prev
    lengths = np.hypot(d[:, 0], d[:, 1])

    nonzero = lengths > 0
    normals = np.zeros_like(d)
    normals[nonzero, 0] = -d[nonzero, 1] / lengths[nonzero]
    normals[nonzero, 1] = d[nonzero, 0] / lengths[nonzero]

    return points + normals * distance


def interp(points, n=10):
    tck, u = splprep(points.T, s=0)
    u_new = np.linspace(0, 1, n)
    return np.vstack(splev(u_new, tck)).T


def plot_polyline(angles, lengths):
    angles = np.array(angles)

    angles[0] += 90
    angles[-2:] *= [-1, 1]

    angle_sum = np.cumsum(np.radians(angles))
    x1, y1 = np.zeros(len(angles) + 1), np.zeros(len(angles) + 1)
    x1[1:], y1[1:] = lengths * np.cos(angle_sum), lengths * np.sin(angle_sum)
    arm_coords = np.vstack((np.cumsum(x1), np.cumsum(y1))).T

    angles[-2:] *= -1
    angle_sum = np.cumsum(np.radians(angles))
    x2, y2 = np.zeros(len(angles) + 1), np.zeros(len(angles) + 1)
    x2[1:], y2[1:] = lengths * np.cos(angle_sum), lengths * np.sin(angle_sum)
    x2y2 = np.vstack((np.cumsum(x2), np.cumsum(y2))).T[-3:]

    grip_coords = np.vstack([arm_coords[-2:][::-1], x2y2])
    arm_coords = arm_coords[:-2]

    return arm_coords, grip_coords


def get_sensors_coords(grip_coords, n=20):
    n_2 = n // 2

    pts1 = interp(generate_offset_points(grip_coords, distance=-0.1), n)
    pts2 = interp(generate_offset_points(grip_coords, distance=0.1), n)
    pts1 = pts1[::-1]

    points = np.vstack([pts1[n_2:], pts2, pts1[:n_2]])
    return points


def plot_somatosensory(axes, weights, px, py, sensor_points):
    sensors = weights["ssensory"][px, py]
    ax = axes["ssensory"]
    ax.clear()
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.5, 1.5)

    idcs = np.where(sensors > 1e-5)
    ax.scatter(*sensor_points[idcs].T, c="black", s=0.5 + 60 * sensors[idcs])

    ax.scatter(*sensor_points.T, c="black", s=0.1)
    ax.set_axis_off()


def plot_proprioceptive(axes, weights, px, py, g):
    angles = weights["proprio"][px, py]
    grip = g.generate_gripper(angles)
    ax = axes["proprio"]
    ax.clear()
    ax.set_xlim(-0.2, 0.8)
    ax.set_ylim(-0.8, 0.8)

    ax.scatter(*grip.T, c="black")
    ax.plot(*grip.T, c="black")
    ax.set_axis_off()


def plot_retina(axes, weights, px, py):
    retina = weights["visual"][px, py]

    retina = (retina - retina.min()) / np.ptp(retina)
    ax = axes["visual"]
    ax.clear()
    ax.imshow(retina, aspect="auto")
    ax.set_axis_off()


def load_and_process_data(trajectory_file="trajectories.csv", weight_file="weights.npy"):
    df = pd.read_csv(trajectory_file)
    has_sensors = "s0" in df.columns

    weights = np.load(weight_file, allow_pickle=True)[0]

    dim, _ = weights["proprio"].shape
    weights["proprio"] = weights["proprio"].reshape(dim, 10, 10)
    weights["proprio"] = weights["proprio"].transpose(1, 2, 0)
    weights["proprio"] = weights["proprio"][:, :, -2:]

    dim, _ = weights["ssensory"].shape
    weights["ssensory"] = weights["ssensory"].reshape(dim, 10, 10)
    weights["ssensory"] = weights["ssensory"].transpose(1, 2, 0)

    weights["visual"] = weights["visual"].reshape(10, 10, 3, 10, 10)
    weights["visual"] = weights["visual"].transpose(3, 4, 0, 1, 2)

    return df, has_sensors, weights


def create_figure_layout(g):
    xlims = np.array([-2, 4])
    ylims = np.array([-3, 3])

    gridsize = (4, 3)

    ratio = gridsize[0] / gridsize[1]
    dim = 8

    fig = plt.figure(figsize=(dim, dim * ratio))

    pmap_ax = plt.subplot2grid(gridsize, [3, 0], 1, 1, fig=fig)
    smap_ax = plt.subplot2grid(gridsize, [3, 1], 1, 1, fig=fig)
    vmap_ax = plt.subplot2grid(gridsize, [3, 2], 1, 1, fig=fig)
    pmap_ax.set_axis_off()
    smap_ax.set_axis_off()
    vmap_ax.set_axis_off()

    proprio_ax = plt.subplot2grid(gridsize, [2, 0], 1, 1, fig=fig)
    ssensory_ax = plt.subplot2grid(gridsize, [2, 1], 1, 1, fig=fig)
    visual_ax = plt.subplot2grid(gridsize, [2, 2], 1, 1, fig=fig)
    proprio_ax.set_title("Proprioception")
    ssensory_ax.set_title("Somatosensory")
    visual_ax.set_title("Foveal vision")

    sensor_points = g.generate_sensor_points(40)

    video_ax = plt.subplot2grid(gridsize, [0, 1], 2, 2, fig=fig, aspect="equal")
    label_ax = plt.subplot2grid(gridsize, [0, 0], 1, 1, fig=fig, aspect="equal")
    label_ax.set_axis_off()
    traces_ax = plt.subplot2grid(gridsize, [1, 0], 1, 1, fig=fig, aspect="equal")
    traces_ax.set_xlim(-0.5, 9.5)
    traces_ax.set_ylim(-0.5, 9.5)
    traces_ax.set_xticks(np.arange(10), [])
    traces_ax.set_yticks(np.arange(10), [])
    traces_ax.grid()

    fig.tight_layout(pad=0.1)

    axes = {
        "pmap": pmap_ax,
        "smap": smap_ax,
        "vmap": vmap_ax,
        "traces": traces_ax,
        "proprio": proprio_ax,
        "ssensory": ssensory_ax,
        "visual": visual_ax,
        "video": video_ax,
        "label": label_ax,
    }
    return fig, axes, xlims, ylims, sensor_points


def setup_ax(axes, key, xlims=None, ylims=None):
    ax = axes[key]
    ax.clear()
    if xlims is not None:
        ax.set_xlim(xlims)
    if ylims is not None:
        ax.set_ylim(ylims)
    ax.set_axis_off()


def add_marker(ax, point, width, height):
    rp = Rectangle(point - 0.5, width=width, height=height, fc="#fff0", ec="red")
    ax.add_patch(rp)


def update_maps(g, wfile, font_size, axes, px, py):
    setup_ax(axes, "pmap")
    g.proprio_map(ax=axes["pmap"], wfile=wfile)
    add_marker(axes["pmap"], np.array([py, px]) + [0.5, 0.5], 1, 1)

    setup_ax(axes, "vmap")
    g.visual_map(ax=axes["vmap"], wfile=wfile)
    add_marker(axes["vmap"], np.array([py, 10 - px - 1]) * 10 + 0.5, 10, 10)

    setup_ax(axes, "smap")
    g.somatosensory_map(ax=axes["smap"], wfile=wfile)
    add_marker(axes["smap"], np.array([py, px]) + 0.5, 1, 1)


class TrajectoryAnimator:
    def __init__(
        self, params, font_size, axes, fig, has_sensors, xlims, ylims, episode_id, rep
    ):
        self.axes = axes
        self.fig = fig
        self.has_sensors = has_sensors
        self.xlims = xlims
        self.ylims = ylims
        self.font_size = font_size

        self.lines1 = []
        self.lines2 = []
        self.scatters = []
        self.reps = {}
        self.traces = {}
        self._initialized = False

        self.episode_id = episode_id
        self.rep = rep
        self.episode_df, self.conditions_df = build_episode_dataset(params)

        self._anim = None
        self.params = params

    def _init_artists(self, n, episode_id, trajectory):
        self.goal_color = "#cc4"
        self.touch_color = "#c44"
        self.proprio_color = "#44c"

        video_ax = self.axes["video"]
        traces_ax = self.axes["traces"]
        label_ax = self.axes["label"]

        for i in range(n):
            (line1,) = video_ax.plot([], [], c="black", marker="o", zorder=-100 + n)
            (line2,) = video_ax.plot([], [], c="black", marker="o", zorder=-100 + n)
            self.lines1.append(line1)
            self.lines2.append(line2)

            if self.has_sensors:
                scatter = video_ax.scatter([], [], c="red", zorder=-100 + n - 1)
                self.scatters.append(scatter)

            self.traces = {
                "g": traces_ax.plot([999, 999], [999, 999], c=self.goal_color)[0],
                "ss": traces_ax.plot([999, 999], [999, 999], c=self.touch_color)[0],
                "p": traces_ax.plot([999, 999], [999, 999], c=self.proprio_color)[0],
            }

            self.reps = {
                "g": traces_ax.scatter(
                    999,
                    999,
                    marker="h",
                    fc=self.goal_color,
                    ec="#000",
                    lw=0.5,
                    s=300,
                    label=None if i < n - 1 else "goal",
                ),
                "ss": traces_ax.scatter(
                    999,
                    999,
                    marker="*",
                    fc=self.touch_color,
                    ec="#000",
                    lw=0.5,
                    s=300,
                    label=None if i < n - 1 else "somatosen",
                ),
                "p": traces_ax.scatter(
                    999,
                    999,
                    marker="*",
                    fc=self.proprio_color,
                    ec="#000",
                    lw=0.5,
                    s=300,
                    label=None if i < n - 1 else "proprio",
                ),
            }

        traces_ax.legend(loc="center left", bbox_to_anchor=(1, 0.7), title="Reps")

        xlims = [0, 10]
        ylims = [0, 10]
        label_ax.set_xlim(xlims)
        label_ax.set_ylim(ylims)
        ylims = label_ax.get_ylim()
        print(xlims, ylims)

        extent = [0, 6, 4, 10]
        ts0, tsl = trajectory.ets.iloc[[0, -1]]

        gif = Image.open(f"episode_{self.episode_id}_{self.rep+1}.gif")
        gif.seek(0)
        gif.seek(
            self.params.drop_first_n_steps + self.params.policy_selection_steps + tsl
        )
        self.framel = np.array(gif)[150:250, 50:150]

        self.episode_template = label_ax.imshow(self.framel, extent=extent, zorder=900)

        objs = ["blue cube", "red triangle", "green cube"]
        obj = objs[self.episode_df.query(f"index=={episode_id}").context.iloc[0] - 1]
        stretch = self.episode_df.query(f"index=={episode_id}").stretch.iloc[0]
        rot = self.episode_df.query(f"index=={episode_id}").rotation.iloc[0]

        self.episode_label = label_ax.text(
            0,
            0,
            f" Object: {obj}\nStretch: {stretch}\n"
            f"rotation: {np.degrees(rot).round(0)}°\n",
            fontdict={"size": self.font_size},
            zorder=800,
            verticalalignment="bottom",
        )

        self.axes["proprio"].set_title("Proprioception")
        self.axes["ssensory"].set_title("Somatosensory")
        self.axes["visual"].set_title("Foveal vision")

        self.axes["proprio"].text(
            -1.2,
            0.7,
            "Current prototypes",
            fontdict={"size": self.font_size},
            rotation=90,
        )
        self.axes["pmap"].text(
            -2.5,
            1.5,
            "Representation grids",
            fontdict={"size": self.font_size},
            rotation=90,
        )
        self._initialized = True

    def clear(self):
        for line in self.lines1:
            line.set_data([], [])
        for line in self.lines2:
            line.set_data([], [])
        for scatter in self.scatters:
            scatter.set_offsets(np.empty((0, 2)))
            scatter.set_sizes([])

    def animate(self, trajectory):
        n = trajectory.shape[0]
        ts_vals = trajectory.ts.to_numpy()
        data = trajectory.iloc[:, 1:6].to_numpy()

        sensor_data = trajectory.iloc[:, 6:46].to_numpy() if self.has_sensors else None

        ss_data = trajectory.loc[:, ["touch_x", "touch_y"]].to_numpy()[:, ::-1]
        p_data = trajectory.loc[:, ["proprio_x", "proprio_y"]].to_numpy()[:, ::-1]
        g_data = trajectory.loc[:, ["prototype_x", "prototype_y"]].to_numpy()[:, ::-1]

        if not self._initialized or len(self.lines1) != n:
            for line in self.lines1:
                line.remove()
            for line in self.lines2:
                line.remove()
            for scatter in self.scatters:
                scatter.remove()
            self.lines1.clear()
            self.lines2.clear()
            self.scatters.clear()
            self.traces.clear()
            self.reps.clear()
            self._init_artists(n, self.episode_id, trajectory)

        indices = ts_vals.astype(int)
        all_angles = np.degrees(data[indices])
        self.polylines = [plot_polyline(ang, [1, 1, 1, 0.5, 0.5]) for ang in all_angles]
        alphas = 0.02 + 0.98 * np.exp(-np.linspace(-5, 0, n) ** 2)

        offsets_pts = []
        sizes_arr = []
        if self.has_sensors:
            all_sensors = sensor_data[indices]
            for i, (arm_coords, grip_coords) in enumerate(self.polylines):
                pts = get_sensors_coords(grip_coords)
                offsets_pts.append(pts)
                ssensors = all_sensors[i]
                sizes_arr.append(100 * ssensors)

        traces_ax = self.axes["traces"]

        def update(frame_idx):
            artists = []
            for i in range(frame_idx + 1):
                arm_coords, grip_coords = self.polylines[i]
                alpha = alphas[i]

                self.lines1[i].set_data(arm_coords[:, 0], arm_coords[:, 1])
                self.lines1[i].set_alpha(alpha)
                self.lines2[i].set_data(grip_coords[:, 0], grip_coords[:, 1])
                self.lines2[i].set_alpha(alpha)

                p = traces_ax.plot(
                    *ss_data[: i + 1].T, c=self.touch_color, lw=2, zorder=-3
                )
                p.extend(
                    traces_ax.plot(
                        *p_data[: i + 1].T,
                        c=self.proprio_color,
                        lw=2,
                        zorder=-3,
                    )
                )

                self.reps["ss"].set_offsets(ss_data[i])
                self.reps["p"].set_offsets(p_data[i])
                self.reps["g"].set_offsets([g_data[i]])

                artists.extend(
                    [
                        self.lines1[i],
                        self.lines2[i],
                        *self.reps.values(),
                    ]
                )

                if self.has_sensors:
                    self.scatters[i].set_offsets(offsets_pts[i])
                    self.scatters[i].set_sizes(sizes_arr[i])
                    self.scatters[i].set_alpha(0.2 + 0.8 * alpha)
                    artists.append(self.scatters[i])
                artists.extend(p)
                artists.append(self.episode_template)
                artists.append(self.episode_label)
            return artists

        self.anim = FuncAnimation(
            self.fig, update, frames=n, interval=50, blit=True, repeat=False
        )
        return self.anim


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process episode and goal identifiers.")
    parser.add_argument(
        "-e",
        "--episode_id",
        type=int,
        help="Unique identifier for the episode",
    )
    parser.add_argument(
        "-r",
        "--rep",
        type=int,
        help="Number of episode repetition",
    )
    parser.add_argument(
        "-g",
        "--goal_id",
        type=int,
        help="Unique identifier for the goal within the episode",
    )
    parser.add_argument(
        "-o",
        "--online",
        action="store_true",
        help="iRender online",
    )
    args = parser.parse_args()

    params = Parameters()
    sm = None
    g = GraphManager(sm, params)
    font_size = 11

    df, has_sensors, weights = load_and_process_data(trajectory_file="trajectory_df.csv")
    wfile = "weights.npy"

    df["ets"] = df.groupby(["episode_id", "e_seed"]).cumcount()

    trajectory = df[(df.episode_id == args.episode_id) & (df.goal_id == args.goal_id)]

    if "e_seed" in trajectory.columns:
        seeds = trajectory.e_seed.unique()
        cur_seed = seeds[args.rep]
        trajectory = trajectory.query(f"e_seed=={cur_seed}")

    fig, axes, xlims, ylims, sensor_points = create_figure_layout(g)
    setup_ax(axes, "video", xlims, ylims)
    px = int(trajectory.prototype_x.iat[0])
    py = int(trajectory.prototype_y.iat[0])

    update_maps(g, wfile, font_size, axes, px, py)
    plot_proprioceptive(axes, weights, px, py, g)
    plot_somatosensory(axes, weights, px, py, sensor_points)
    plot_retina(axes, weights, px, py)

    animator = TrajectoryAnimator(
        params,
        font_size,
        axes,
        fig,
        has_sensors,
        xlims,
        ylims,
        args.episode_id,
        args.rep,
    )
    anim = animator.animate(trajectory)

    if args.online:
        plt.show()
    else:
        anim.save(
            filename=f"postures_e{args.episode_id}_g{args.goal_id}_{args.rep}" ".gif",
            writer="pillow",
        )
