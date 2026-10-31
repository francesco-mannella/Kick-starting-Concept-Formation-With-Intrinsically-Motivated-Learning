#!/usr/bin/env python
"""Visualization module for trajectory animation with weight maps."""

import argparse
import sys
import warnings
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Rectangle
from PIL import Image
from scipy.interpolate import splev, splprep

from params import Parameters
from posture_geometry import arm_and_gripper
from SMGraphs import GraphManager
from SMMain import build_episode_dataset


warnings.filterwarnings("error")

RENDER_CROP = (slice(99, 280), slice(52, 233))


def generate_offset_points(points, distance=0.05):
    """Generate points offset by perpendicular normals."""
    points = np.asarray(points, dtype=float)
    n = len(points)
    if n < 2:
        return points.copy() if n == 1 else np.empty((0, 2), dtype=float)

    p_prev = np.empty_like(points)
    p_next = np.empty_like(points)
    p_prev[0], p_prev[1:] = points[0], points[:-1]
    p_next[-1], p_next[:-1] = points[-1], points[1:]

    d = p_next - p_prev
    lengths = np.hypot(d[:, 0], d[:, 1])
    nonzero = lengths > 0

    normals = np.zeros_like(d)
    normals[nonzero, 0] = -d[nonzero, 1] / lengths[nonzero]
    normals[nonzero, 1] = d[nonzero, 0] / lengths[nonzero]

    return points + normals * distance


def interp(points, n=10):
    """Interpolate points using B-spline."""
    tck, _ = splprep(points.T, s=0)
    return np.vstack(splev(np.linspace(0, 1, n), tck)).T


def plot_polyline(angles, lengths=(1, 1, 1, 0.5, 0.5)):
    """Arm and gripper coordinates from the 8 joint angles in degrees (see posture_geometry)."""
    return arm_and_gripper(angles, lengths[:3], lengths[3:5])


def get_sensors_coords(grip_coords, n=20):
    """Get sensor coordinates around gripper."""
    n_2 = n // 2
    pts1 = interp(generate_offset_points(grip_coords, -0.1), n)[::-1]
    pts2 = interp(generate_offset_points(grip_coords, 0.1), n)
    return np.vstack([pts1[n_2:], pts2, pts1[:n_2]])


class ProprioceptionPlot:
    """Handles proprioceptive weight visualization."""

    def __init__(self, axes, weights, g):
        self.weights = weights
        self.g = g
        if isinstance(axes, dict):
            self.ax = axes["proprio"]
        else:
            self.ax = axes
        self._init_plot()

    def _init_plot(self):
        self.ax.clear()
        self.ax.set_xlim(-1.1, 1.1)
        self.ax.set_ylim(-1.1, 1.1)
        self.ax.set_aspect("equal", adjustable="box")
        self.ax.set_axis_off()
        self.scatter = self.ax.scatter([], [], c="black")
        (self.line,) = self.ax.plot([], [], c="black")

    def update(self, px, py):
        angles = self.weights["proprio"][px, py]
        grip = self.g.generate_gripper(angles)
        self.scatter.set_offsets(grip)
        self.line.set_data(grip[:, 0], grip[:, 1])


class SomatosensoryPlot:
    """Handles somatosensory weight visualization."""

    def __init__(self, axes, weights, sensor_points):
        self.weights = weights
        self.sensor_points = sensor_points
        if isinstance(axes, dict):
            self.ax = axes["ssensory"]
        else:
            self.ax = axes
        self._init_plot()

    def _init_plot(self):
        self.ax.clear()
        self.ax.set_xlim(0, 1)
        self.ax.set_ylim(-0.5, 1.5)
        self.ax.set_axis_off()
        self.scatter_active = self.ax.scatter([], [], c="black")
        self.scatter_base = self.ax.scatter(*self.sensor_points.T, c="black", s=0.1)

    def update(self, px, py):
        sensors = self.weights["ssensory"][px, py]
        idcs = np.where(sensors > 1e-5)[0]
        self.scatter_active.set_offsets(self.sensor_points[idcs])
        self.scatter_active.set_sizes(0.5 + 60 * sensors[idcs])


class RetinaPlot:
    """Handles visual/retina weight visualization."""

    def __init__(self, axes, weights):
        self.weights = weights
        self.ax = axes["visual"]
        self._init_plot()

    def _init_plot(self):
        self.ax.clear()
        self.ax.set_axis_off()
        dummy = np.zeros((10, 10))
        self.image = self.ax.imshow(dummy, aspect="auto")

    def update(self, px, py):
        retina = self.weights["visual"][py, px]
        retina = (retina - retina.min()) / np.ptp(retina)
        self.image.set_data(retina)


class MapPlot:
    """Handles weight map visualization with markers."""

    def __init__(self, ax, g, wfile, map_type, font_size):
        self.ax = ax
        self.g = g
        self.wfile = wfile
        self.map_type = map_type
        self.font_size = font_size
        self.markers = []
        self._init_plot()

    def _init_plot(self):
        self.ax.clear()
        self.ax.set_axis_off()
        if self.map_type == "visual":
            self.g.visual_map(ax=self.ax, wfile=self.wfile)
        elif self.map_type == "proprio":
            self.g.proprio_map(ax=self.ax, wfile=self.wfile)
        elif self.map_type == "somatosensory":
            self.g.somatosensory_map(ax=self.ax, wfile=self.wfile)

        self.ax.set_aspect("equal")
        xlims = self.ax.get_xlim()
        ylims = self.ax.get_ylim()
        xrng = np.ptp(xlims)
        yrng = np.ptp(ylims)

        if self.map_type != "visual":
            self.ax.set_xlim(-xrng * 0.05, xrng * 1.05)
            self.ax.set_ylim(-yrng * 0.05, yrng * 1.05)
        else:
            self.ax.set_xlim(-xrng * 0.06, xrng * 1.05)
            self.ax.set_ylim(yrng * 0.96, -yrng * 0.16)

        self._create_markers()

    def _create_markers(self):
        for m in self.markers:
            for p in m:
                p.remove()
        self.markers = []

        colors = ["yellow", "red", "blue"]
        for _ in range(3):
            bg_rect = Rectangle((0, 0), 1, 1, lw=2, fc="#0000", ec="black", visible=False)
            fg_rect = Rectangle((0, 0), 1, 1, lw=1, fc="#fff0", ec="black", visible=False)
            self.ax.add_patch(bg_rect)
            self.ax.add_patch(fg_rect)
            self.markers.append((bg_rect, fg_rect))

        for i, (bg, fg) in enumerate(self.markers):
            fg.set_edgecolor(colors[i])

    def _update_marker(self, idx, point, width, height, lw=None):
        bg_rect, fg_rect = self.markers[idx]

        w = width + width * 0.1 * (lw or 0)
        h = height + height * 0.1 * (lw or 0)
        pt = point - width * 0.05 * (lw or 0)

        bg_rect.set_xy(pt - 0.5)
        bg_rect.set_width(w)
        bg_rect.set_height(h)
        bg_rect.set_linewidth(2 + (lw or 0))
        bg_rect.set_visible(True)

        fg_rect.set_xy(pt - 0.5)
        fg_rect.set_width(w)
        fg_rect.set_height(h)
        fg_rect.set_linewidth(1 if lw is None else lw)
        fg_rect.set_visible(True)

    def update(self, px, py, tx, ty, ax, ay):
        if self.map_type == "visual":
            self._update_marker(0, np.array([py, 10 - 1 - px]) * 10 + 0.5, 10, 10, lw=2)
            self._update_marker(1, np.array([ty, 10 - 1 - tx]) * 10 + 0.5, 10, 10)
            self._update_marker(2, np.array([ay, 10 - 1 - ax]) * 10 + 0.5, 10, 10)
        else:
            self._update_marker(0, np.array([py, px]) + 0.5, 1, 1, lw=2)
            self._update_marker(1, np.array([ty, tx]) + 0.5, 1, 1)
            self._update_marker(2, np.array([ay, ax]) + 0.5, 1, 1)


def load_and_process_data(trajectory_file="trajectories.csv", weight_file="weights.npy"):
    """Load trajectory and weight data, reshape weights appropriately."""
    df = pd.read_csv(trajectory_file)
    has_sensors = "s0" in df.columns
    weights = np.load(weight_file, allow_pickle=True)[0]

    dim = weights["proprio"].shape[0]
    weights["proprio"] = weights["proprio"].reshape(dim, 10, 10)
    weights["proprio"] = weights["proprio"].transpose(1, 2, 0)

    dim = weights["ssensory"].shape[0]
    weights["ssensory"] = weights["ssensory"].reshape(dim, 10, 10)
    weights["ssensory"] = weights["ssensory"].transpose(1, 2, 0)

    weights["visual"] = weights["visual"].reshape(10, 10, 3, 10, 10)
    weights["visual"] = weights["visual"].transpose(4, 3, 0, 1, 2)

    return df, has_sensors, weights


def arm_points(df):
    angles = np.degrees(df[[f"d{x}" for x in range(1, 9)]].to_numpy())
    return np.vstack(
        [np.vstack(plot_polyline(a.copy(), [1, 1, 1, 0.5, 0.5])) for a in angles]
    )


def arm_limits(trajectory, margin=0.15):
    """Square limits fitted to the arm and gripper over all frames of the displayed trajectory."""
    points = arm_points(trajectory)
    lo, hi = points.min(0), points.max(0)
    half = (hi - lo).max() / 2 + margin
    cx, cy = (lo + hi) / 2
    return np.array([cx - half, cx + half]), np.array([cy - half, cy + half])


def create_figure_layout(g, xlims=None, ylims=None):
    """Create figure with subplot grid layout."""
    if xlims is None:
        xlims, ylims = np.array([-2, 4]), np.array([-3, 3])
    gridsize = (4, 3)
    ratio = gridsize[0] / gridsize[1]
    dim = 8

    fig = plt.figure(figsize=(dim, dim * ratio))

    axes = {}
    for name, pos in [("pmap", [3, 0]), ("smap", [3, 1]), ("vmap", [3, 2])]:
        axes[name] = plt.subplot2grid(gridsize, pos, 1, 1, fig=fig)
        axes[name].set_axis_off()

    titles = {
        "proprio": "Proprioception",
        "ssensory": "Somatosensory",
        "visual": "Foveal vision",
    }
    for i, (name, title) in enumerate(titles.items()):
        axes[name] = plt.subplot2grid(gridsize, [2, i], 1, 1, fig=fig)
        axes[name].set_title(title)

    sensor_points = g.generate_sensor_points(40)

    axes["video"] = plt.subplot2grid(gridsize, [0, 1], 2, 2, fig=fig, aspect="equal")
    axes["label"] = plt.subplot2grid(gridsize, [0, 0], 1, 1, fig=fig, aspect="equal")
    axes["label"].set_axis_off()

    axes["traces"] = plt.subplot2grid(gridsize, [1, 0], 1, 1, fig=fig, aspect="equal")
    axes["traces"].set_xlim(-0.5, 9.5)
    axes["traces"].set_ylim(-0.5, 9.5)
    axes["traces"].set_xticks(np.arange(10), [])
    axes["traces"].set_yticks(np.arange(10), [])
    axes["traces"].grid()

    fig.tight_layout(pad=0.1)
    return fig, axes, xlims, ylims, sensor_points


def setup_ax(axes, key, xlims=None, ylims=None):
    """Clear and configure axis limits."""
    ax = axes[key]
    ax.clear()
    if xlims is not None:
        ax.set_xlim(xlims)
    if ylims is not None:
        ax.set_ylim(ylims)
    ax.set_axis_off()


class BaseTrajectoryAnimator:
    """Base class for trajectory animation with common functionality."""

    COLORS = {"goal": "#cc4", "touch": "#c44", "proprio": "#44c"}
    GIF_NAME_TEMPLATE = "e{episode_id:02d}_g{goal_id:02d}_r{rep}_{px}{py}"

    def __init__(self, fig, ax, episode_id, rep, xlims=None, ylims=None, end_steps=40):
        self.fig = fig
        self.ax = ax
        self.xlims = xlims
        self.ylims = ylims
        self.episode_id = episode_id
        self.rep = rep
        self.end_steps = end_steps
        self.lines1, self.lines2 = [], []
        self.traces = {}
        self._initialized = False
        self.anim = None

        if xlims is not None:
            self.ax.set_xlim(xlims)
        if ylims is not None:
            self.ax.set_ylim(ylims)

    def _create_arm_lines(self, ax, n):
        """Create arm and gripper line artists."""
        for _ in range(n):
            (line1,) = ax.plot([], [], c="black", marker="o", zorder=-100 + n)
            (line2,) = ax.plot([], [], c="black", marker="o", zorder=-100 + n)
            self.lines1.append(line1)
            self.lines2.append(line2)

    def _remove_lines(self):
        """Remove arm and gripper lines."""
        for line in self.lines1 + self.lines2:
            line.remove()
        self.lines1, self.lines2 = [], []

    def clear(self):
        """Clear all animation artists."""
        for line in self.lines1 + self.lines2:
            line.set_data([], [])

    def _prepare_animation_data(self, trajectory):
        """Prepare common animation data from trajectory."""
        n = trajectory.shape[0]
        ts_vals = trajectory.ts.to_numpy()
        data = trajectory.iloc[:, 1:9].to_numpy()
        indices = ts_vals.astype(int)
        all_angles = np.degrees(data[indices])
        polylines = [plot_polyline(ang, [1, 1, 1, 0.5, 0.5]) for ang in all_angles]
        alphas = 0.01 + 0.99 * np.exp(-np.linspace(-2.3, 0, n) ** 2)
        return n, polylines, alphas

    def get_output_name(self, goal_id, px, py):
        """Generate output filename for animation."""
        return self.GIF_NAME_TEMPLATE.format(
            episode_id=self.episode_id, goal_id=goal_id, rep=self.rep, px=px, py=py
        )

    def save_animation(self, goal_id, px, py, fps=20):
        """Save animation as GIF and PNG."""
        name = self.get_output_name(goal_id, px, py)
        writer = matplotlib.animation.PillowWriter(fps=fps, metadata=["loop", "1"])

        demos = Path("demos")
        self.anim.save(filename=demos / f"{name}.gif", writer=writer)
        self.fig.savefig(demos / f"final_{name}.png", dpi=300)

    def show(self):
        """Display animation interactively."""
        plt.show()


class SimpleTrajectoryAnimator(BaseTrajectoryAnimator):
    """Handles simplified trajectory animation showing only arm traces."""

    def _init_artists(self, n):
        """Initialize plot artists for animation."""
        self._create_arm_lines(self.ax, n)
        for line in self.lines1 + self.lines2:
            line.set_markersize(line.get_markersize() / 5)
        self._initialized = True

    def _clear_artists(self):
        """Remove all existing artists."""
        self._remove_lines()
        self._initialized = False

    def animate(self, trajectory):
        """Create and return animation for trajectory."""
        n, polylines, alphas = self._prepare_animation_data(trajectory)
        alphas = 0.01 + 0.99 * np.exp(-np.linspace(-2, 0, n) ** 2)

        if not self._initialized or len(self.lines1) != n:
            self._clear_artists()
            self._init_artists(n)

        total_frames = n + self.end_steps

        def update(frame_idx):
            data_idx = min(frame_idx, n - 1)
            artists = []
            for i in range(data_idx + 1):
                arm_coords, grip_coords = polylines[i]
                self.lines1[i].set_data(arm_coords[:, 0], arm_coords[:, 1])
                self.lines1[i].set_alpha(alphas[i])
                self.lines2[i].set_data(grip_coords[:, 0], grip_coords[:, 1])
                self.lines2[i].set_alpha(alphas[i])
                artists.extend([self.lines1[i], self.lines2[i]])
            return artists

        self.anim = FuncAnimation(
            self.fig, update, frames=total_frames, interval=50, blit=True, repeat=False
        )
        return self.anim


class TrajectoryAnimator(BaseTrajectoryAnimator):
    """Handles trajectory animation with visual feedback."""

    def __init__(
        self,
        params,
        font_size,
        axes,
        fig,
        has_sensors,
        xlims,
        ylims,
        episode_id,
        rep,
        trajectories,
        weights,
        sensor_points,
        g,
        wfile,
        fast=True,
        end_steps=40,
        online=False,
    ):
        """Initialize the trajectory animator."""
        super().__init__(fig, axes.get("video"), episode_id, rep, xlims, ylims, end_steps)
        self.axes = axes
        self.has_sensors = has_sensors
        self.font_size = font_size
        self.fast = fast
        self.params = params
        self.trajectories = trajectories
        self.weights = weights
        self.sensor_points = sensor_points
        self.g = g
        self.wfile = wfile
        self.online = online

        self.scatters = []
        self.reps = {}
        self.episode_df, self.conditions_df = build_episode_dataset(params)

        self.proprio_plot = ProprioceptionPlot(axes, weights, g)
        self.somato_plot = SomatosensoryPlot(axes, weights, sensor_points)
        self.retina_plot = RetinaPlot(axes, weights)

        self.vmap_plot = MapPlot(axes["vmap"], g, wfile, "visual", font_size)
        self.pmap_plot = MapPlot(axes["pmap"], g, wfile, "proprio", font_size)
        self.smap_plot = MapPlot(axes["smap"], g, wfile, "somatosensory", font_size)

    def _init_artists(self, n, episode_id, trajectory):
        """Initialize plot artists for animation."""
        video_ax = self.axes["video"]
        traces_ax = self.axes["traces"]

        self._create_arm_lines(video_ax, n)

        if self.has_sensors:
            for _ in range(n):
                scatter = video_ax.scatter([], [], c="red", zorder=-100 + n - 1)
                self.scatters.append(scatter)

        self._init_traces(traces_ax, n)
        self._init_reps(traces_ax)
        traces_ax.legend(
            loc="lower center", bbox_to_anchor=(0.5, 1.01), ncol=3, frameon=False,
            fontsize=8, handletextpad=0.2, columnspacing=0.8,
        )

        self._init_labels(self.axes["label"], trajectory, episode_id)
        self._initialized = True

    def _init_traces(self, ax, n):
        """Initialize trace lines."""
        trace_keys = [("g", "goal"), ("ss", "touch"), ("p", "proprio")]
        if self.fast:
            self.traces = {
                k: ax.plot([999], [999], lw=0.5, c=self.COLORS[c])[0]
                for k, c in trace_keys
            }
        else:
            self.traces = {
                k: [ax.plot([999], [999], c=self.COLORS[c])[0] for _ in range(n - 1)]
                for k, c in trace_keys
            }

    def _init_reps(self, ax):
        """Initialize representation scatter plots."""
        configs = [
            ("g", "h", "goal", "goal"),
            ("ss", "*", "touch", "somato"),
            ("p", "*", "proprio", "proprio"),
        ]
        for key, marker, color_key, label in configs:
            self.reps[key] = ax.scatter(
                999,
                999,
                marker=marker,
                fc=self.COLORS[color_key],
                ec="#000",
                lw=0.5,
                s=300,
                label=label,
            )

    def _get_frame_from_trajectory(self, episode_id, trajectory, save=False):
        """Extract frame from GIF based on trajectory data."""
        tss = trajectory.ets.iloc[0]
        tsl = trajectory.ets.iloc[-1]
        t_tot_frames = self.trajectories.shape[0]

        episode_dir = Path("episodes")
        episode_file = episode_dir / f"episode_{episode_id}_{self.rep}.gif"

        frames_dir = Path("goal_frames")
        frames_dir.mkdir(exist_ok=True)
        print(episode_file)

        with Image.open(episode_file) as gif:
            start = self.params.drop_first_n_steps + self.params.policy_selection_steps

            print("Saving first frame:")
            print(f"     timestep: {tss}/{t_tot_frames}")
            gif.seek(start + tss)
            frames = np.array(gif)[RENDER_CROP]

            print("Saving last frame:")
            print(f"     timestep: {tsl}/{t_tot_frames}")
            gif.seek(start + tsl)
            framel = np.array(gif)[RENDER_CROP]

            if save:
                goal_id = int(trajectory.goal_id.iat[0])
                px = int(trajectory.prototype_x.iat[0])
                py = int(trajectory.prototype_y.iat[0])
                frames_file = (
                    frames_dir / f"sf_e{episode_id:02d}_g{goal_id:02d}"
                    f"_r{self.rep}_{px}{py}.png"
                )
                plt.imsave(frames_file, frames)
                framel_file = (
                    frames_dir / f"lf_e{episode_id:02d}_g{goal_id:02d}"
                    f"_r{self.rep}_{px}{py}.png"
                )
                print(framel_file)
                plt.imsave(framel_file, framel)
            return frames, framel

    def _init_labels(self, ax, trajectory, episode_id):
        """Initialize episode labels and image."""
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 10)

        self.frames, self.framel = self._get_frame_from_trajectory(episode_id, trajectory)
        self.episode_templates = []
        self.episode_templates.append(
            ax.imshow(self.frames, extent=[0, 5, 5, 10], zorder=900)
        )
        self.episode_templates.append(
            ax.imshow(self.framel, extent=[5, 10, 5, 10], zorder=900)
        )

        query = self.episode_df.query(f"index=={episode_id}")
        objs = ["blue cube", "red triangle", "green cube"]
        obj = objs[query.context.iloc[0] - 1]
        stretch = query.stretch.iloc[0]
        rot = np.degrees(query.rotation.iloc[0]).round(0)

        self.episode_label = ax.text(
            0,
            0,
            f" Object: {obj}\nStretch: {stretch}\nrotation: {rot}°\n",
            fontdict={"size": self.font_size},
            zorder=800,
            verticalalignment="bottom",
        )

        self.axes["proprio"].set_title("Proprioception")
        self.axes["ssensory"].set_title("Somatosensory")
        self.axes["visual"].set_title("Foveal vision")
        self.axes["proprio"].text(
            -0.3,
            -0.5,
            "Current prototypes",
            fontdict={"size": self.font_size},
            rotation=90,
        )
        self.axes["pmap"].text(
            -1.5,
            1.5,
            "Representation grids",
            fontdict={"size": self.font_size},
            rotation=90,
        )

    def clear(self):
        """Clear all animation artists."""
        super().clear()
        for scatter in self.scatters:
            scatter.set_offsets(np.empty((0, 2)))
            scatter.set_sizes([])

    def animate(self, trajectory):
        """Create and return animation for trajectory."""
        n, polylines, alphas = self._prepare_animation_data(trajectory)
        sensor_data = trajectory.iloc[:, 9:49].to_numpy() if self.has_sensors else None

        ss_data = trajectory.loc[:, ["touch_x", "touch_y"]].to_numpy()[:, ::-1]
        p_data = trajectory.loc[:, ["proprio_x", "proprio_y"]].to_numpy()[:, ::-1]
        g_data = trajectory.loc[:, ["prototype_x", "prototype_y"]].to_numpy()[:, ::-1]

        ss_pp = p_pp = None
        if not self.fast:
            ss_pp = np.stack([ss_data[:-1], ss_data[1:]], axis=1)
            p_pp = np.stack([p_data[:-1], p_data[1:]], axis=1)

        if not self._initialized or len(self.lines1) != n:
            self._clear_artists()
            self._init_artists(n, self.episode_id, trajectory)

        offsets_pts, sizes_arr = self._prepare_sensor_data(
            n, polylines, sensor_data, trajectory
        )

        px = int(trajectory.prototype_x.iat[0])
        py = int(trajectory.prototype_y.iat[0])

        total_frames = n + self.end_steps
        self.proprio_plot.update(px, py)
        self.somato_plot.update(px, py)
        self.retina_plot.update(px, py)
        if self.online:
            tx = int(trajectory.touch_x.iloc[-1])
            ty = int(trajectory.touch_y.iloc[-1])
            ax = int(trajectory.proprio_x.iloc[-1])
            ay = int(trajectory.proprio_y.iloc[-1])
            self.vmap_plot.update(px, py, tx, ty, ax, ay)
            self.pmap_plot.update(px, py, tx, ty, ax, ay)
            self.smap_plot.update(px, py, tx, ty, ax, ay)

        def update(frame_idx):
            data_idx = min(frame_idx, n - 1)
            print(f"frame: {frame_idx:4d}")
            artists = []

            tx = int(trajectory.touch_x.iloc[data_idx])
            ty = int(trajectory.touch_y.iloc[data_idx])
            ax = int(trajectory.proprio_x.iloc[data_idx])
            ay = int(trajectory.proprio_y.iloc[data_idx])

            if not self.online:
                self.vmap_plot.update(px, py, tx, ty, ax, ay)
                self.pmap_plot.update(px, py, tx, ty, ax, ay)
                self.smap_plot.update(px, py, tx, ty, ax, ay)

            for i in range(data_idx + 1):
                arm_coords, grip_coords = polylines[i]
                loc_alphas = 0.01 + 0.99 * np.exp(-np.linspace(-14, 0, i + 1) ** 2)

                age = data_idx - i
                alpha = 1.0 if age == 0 else 0.03 + 0.97 * np.exp(-((age / 12) ** 2))
                width = 3.0 if age == 0 else 1.2
                for line, coords in ((self.lines1[i], arm_coords), (self.lines2[i], grip_coords)):
                    line.set_data(coords[:, 0], coords[:, 1])
                    line.set_alpha(alpha)
                    line.set_linewidth(width)
                    line.set_zorder(-100 + n + i / n)

                self._update_traces(i, ss_data, p_data, ss_pp, p_pp, loc_alphas)
                self._update_reps(i, ss_data, p_data, g_data, loc_alphas, n)

                artists.extend([self.lines1[i], self.lines2[i], *self.reps.values()])
                artists.extend(self._get_trace_artists())

                if self.has_sensors:
                    self.scatters[i].set_offsets(offsets_pts[i])
                    self.scatters[i].set_sizes(sizes_arr[i])
                    self.scatters[i].set_alpha(0.2 + 0.8 * alpha)
                    artists.append(self.scatters[i])

                artists.extend(self.episode_templates + [self.episode_label])

            self.proprio_plot.update(px, py)
            self.somato_plot.update(px, py)
            self.retina_plot.update(px, py)

            return artists

        self.anim = FuncAnimation(
            self.fig, update, frames=total_frames, interval=50, blit=True, repeat=False
        )
        return self.anim

    def _prepare_sensor_data(self, n, polylines, sensor_data, trajectory):
        """Prepare sensor offset and size data for animation."""
        offsets_pts, sizes_arr = [], []
        if self.has_sensors:
            ts_vals = trajectory.ts.to_numpy()
            indices = ts_vals.astype(int)
            all_sensors = sensor_data[indices]
            for i, (_, grip_coords) in enumerate(polylines):
                offsets_pts.append(get_sensors_coords(grip_coords))
                sizes_arr.append(100 * all_sensors[i])
        return offsets_pts, sizes_arr

    def _clear_artists(self):
        """Remove all existing artists."""
        self._remove_lines()
        for scatter in self.scatters:
            scatter.remove()
        self.scatters.clear()
        self.traces.clear()
        self.reps.clear()
        self._initialized = False

    def _update_traces(self, i, ss_data, p_data, ss_pp, p_pp, loc_alphas):
        """Update trace line data."""
        if self.fast:
            self.traces["ss"].set_data(*ss_data[: i + 1].T)
            self.traces["p"].set_data(*p_data[: i + 1].T)
        else:
            trace_configs = [("ss", ss_pp, 0.2), ("p", p_pp, 0.8)]
            for key, pp_data, base_lw in trace_configs:
                for k, tr in enumerate(self.traces[key][: i + 1]):
                    tr.set_data(*pp_data[k].T)
                    tr.set_linewidth(base_lw + 4 * loc_alphas[k])
                    tr.set_alpha(loc_alphas[k])

    def _update_reps(self, i, ss_data, p_data, g_data, loc_alphas, n):
        """Update representation scatter plots."""
        rep_configs = [("ss", ss_data, 50, 250), ("p", p_data, 0, 300)]
        for key, data, base_size, scale in rep_configs:
            self.reps[key].set_offsets(data[: i + 1])
            sizes = base_size + scale * loc_alphas[: i + 1]
            self.reps[key].set_sizes(sizes)

            fc = np.tile(self.reps[key].get_facecolor(), [n, 1])
            ec = np.tile(self.reps[key].get_edgecolor(), [n, 1])
            fc[: i + 1, 3] = loc_alphas[: i + 1]
            ec[: i + 1, 3] = loc_alphas[: i + 1]
            self.reps[key].set_facecolor(fc[: i + 1])
            self.reps[key].set_edgecolor(ec[: i + 1])

        self.reps["g"].set_offsets([g_data[i]])

    def _get_trace_artists(self):
        """Get all trace artists as flat list."""
        if self.fast:
            return list(self.traces.values())
        return [seg for trace in self.traces.values() for seg in trace]


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
        help="Unique identifier for the goal",
    )
    parser.add_argument(
        "-l",
        "--list",
        action="store_true",
        help="Only list the table of episodes and goals",
    )
    parser.add_argument(
        "-m",
        "--episode_matches",
        action="store_true",
        help="Only list the table of episode matches",
    )
    parser.add_argument(
        "-o",
        "--online",
        action="store_true",
        help="Render online",
    )
    parser.add_argument(
        "-f",
        "--fast",
        action="store_true",
        help="Fast rendering of representations",
    )
    parser.add_argument(
        "-d",
        "--last_frame",
        action="store_true",
        help="Only saves last frame image",
    )
    parser.add_argument(
        "-s",
        "--simplified",
        action="store_true",
        help="Simplified animation, trajectories only",
    )
    args = parser.parse_args()

    params = Parameters()
    g = GraphManager(None, params)
    font_size = 11

    df, has_sensors, weights = load_and_process_data(trajectory_file="trajectory_df.csv")
    seeds = np.sort(df.e_seed.unique())
    wfile = "weights.npy"

    gdf = (
        df.groupby(["e_seed", "episode_id", "goal_id"])
        .first()
        .reset_index()[["e_seed", "episode_id", "goal_id"]]
    )

    if args.episode_matches:
        mdf = df
        print(mdf)
        sys.exit()

    if args.list:
        eseed_idcs = gdf.e_seed.unique()
        eseed_idcs.sort()
        eseed_map = {v: i for i, v in enumerate(eseed_idcs)}
        gdf.loc[:, "rep"] = gdf.e_seed.map(eseed_map)
        del gdf["e_seed"]

        gdf = gdf.groupby(["rep", "episode_id"]).count().reset_index()
        gdf = gdf.pivot(index="episode_id", columns="rep").reset_index()
        print(gdf.columns)
        gdf.columns = [
            "_".join(map(str, col)).strip("_") if isinstance(col, tuple) else col
            for col in gdf.columns
        ]
        gdf["type"] = [["B", "R", "G"][i] for i in np.arange(18) % 3]
        print(gdf)
        sys.exit()

    curr_episode_gdf = gdf.query(
        f"episode_id == {args.episode_id} and e_seed == {int(seeds[args.rep])}"
    )

    curr_episode_gdf = curr_episode_gdf.copy()
    curr_episode_gdf.loc[:, "curr_goal"] = ""

    mask = curr_episode_gdf["goal_id"] == args.goal_id
    curr_episode_gdf.loc[mask, "curr_goal"] = "*"

    print()
    print(curr_episode_gdf)
    print()

    if args.goal_id not in curr_episode_gdf["goal_id"].values:
        raise ValueError("The current episode do not reach the this goal number")

    df.loc[:, "ets"] = df.groupby(["e_seed", "episode_id"]).cumcount()
    mask = df.e_seed == seeds[args.rep]
    mask &= df.episode_id == args.episode_id
    trajectories = df.loc[mask].copy()
    mask &= df.goal_id == args.goal_id
    trajectory = df.loc[mask].copy()

    if args.online:
        matplotlib.use("qtagg")
    else:
        matplotlib.use("agg")

    px = int(trajectory.prototype_x.iat[0])
    py = int(trajectory.prototype_y.iat[0])

    if args.simplified:
        xlims, ylims = np.array([-2, 4]), np.array([-3, 3])
        fig, ax = plt.subplots(1, 1, figsize=(1, 1 * (np.ptp(xlims) / np.ptp(ylims))))
        ax.set_axis_off()
        ax.set_axpect = "equal"
        animator = SimpleTrajectoryAnimator(
            fig,
            ax,
            args.episode_id,
            args.rep,
            xlims,
            ylims,
        )

    else:
        xlims, ylims = arm_limits(trajectory)
        fig, axes, xlims, ylims, sensor_points = create_figure_layout(g, xlims, ylims)

        setup_ax(axes, "video", xlims, ylims)

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
            fast=args.fast,
            trajectories=trajectories,
            weights=weights,
            sensor_points=sensor_points,
            g=g,
            wfile=wfile,
            online=args.online,
        )

        if args.last_frame:
            animator._get_frame_from_trajectory(args.episode_id, trajectory, save=True)
            sys.exit()

    anim = animator.animate(trajectory)

    if args.online:
        print(f"x: {px:< 2d} y: {py:< 2d}")
        animator.show()
    else:
        animator.save_animation(args.goal_id, px, py)
