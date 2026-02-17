"""
Visualization module for trajectory animation with proprioceptive,
sensory, and visual weight maps.
"""

import argparse
import glob
import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Rectangle
from scipy.interpolate import splev, splprep

from params import Parameters
from SMGraphs import GraphManager
from SMMain import build_episode_dataset


matplotlib.use("qtagg")


def generate_offset_points(points, distance=0.05):
    """
    Generate points offset perpendicular to a polyline.

    Computes normal vectors at each point along the polyline and
    shifts points by the specified distance in the normal direction.

    Args:
        points: Array of shape (N, 2) defining the polyline vertices.
        distance: Offset distance. Positive values offset to the left
            of the polyline direction, negative to the right.

    Returns:
        np.ndarray: Array of shape (N, 2) containing offset points.
    """
    points = np.asarray(points, dtype=float)
    n = len(points)

    # Handle edge cases for empty or single-point input
    if n == 0:
        return np.empty((0, 2), dtype=float)
    if n == 1:
        return points.copy()

    # Build arrays for tangent computation using neighboring points
    p_prev = np.empty_like(points)
    p_next = np.empty_like(points)
    p_prev[0] = points[0]
    p_prev[1:] = points[:-1]
    p_next[-1] = points[-1]
    p_next[:-1] = points[1:]

    # Compute tangent vectors and their magnitudes
    d = p_next - p_prev
    lengths = np.hypot(d[:, 0], d[:, 1])

    # Compute unit normals perpendicular to tangents
    nonzero = lengths > 0
    normals = np.zeros_like(d)
    normals[nonzero, 0] = -d[nonzero, 1] / lengths[nonzero]
    normals[nonzero, 1] = d[nonzero, 0] / lengths[nonzero]

    return points + normals * distance


def interp(points, n=10):
    """
    Interpolate points using B-spline to create a smooth curve.

    Fits a B-spline through all input points and samples it uniformly.

    Args:
        points: Array of shape (M, 2) containing control points.
        n: Number of output points to generate.

    Returns:
        np.ndarray: Array of shape (n, 2) with interpolated coordinates.
    """
    # s=0 forces the spline to pass through all control points
    tck, u = splprep(points.T, s=0)
    u_new = np.linspace(0, 1, n)
    return np.vstack(splev(u_new, tck)).T


def plot_polyline(angles, lengths):
    """
    Compute polyline coordinates for a multi-segment arm with gripper.

    Converts joint angles and segment lengths into Cartesian coordinates
    for both the arm and the symmetric gripper fingers.

    Args:
        angles: Array of joint angles in degrees.
        lengths: Array of segment lengths corresponding to each joint.

    Returns:
        tuple: (arm_coords, grip_coords) where each is an array of
            shape (N, 2) containing the segment endpoint coordinates.
    """
    angles = np.array(angles)

    # Adjust angles for the coordinate system convention
    angles[1] += 90
    angles[-2:] *= [-1, 1]

    # Forward kinematics: compute cumulative angles and positions
    angle_sum = np.cumsum(np.radians(angles))[1:]
    x1, y1 = np.zeros(len(angles)), np.zeros(len(angles))
    x1[1:], y1[1:] = lengths * np.cos(angle_sum), lengths * np.sin(angle_sum)
    arm_coords = np.vstack((np.cumsum(x1), np.cumsum(y1))).T

    # Compute mirrored gripper finger on opposite side
    angles[-2:] *= -1
    angle_sum = np.cumsum(np.radians(angles))[1:]
    x2, y2 = np.zeros(len(angles)), np.zeros(len(angles))
    x2[1:], y2[1:] = lengths * np.cos(angle_sum), lengths * np.sin(angle_sum)
    x2y2 = np.vstack((np.cumsum(x2), np.cumsum(y2))).T[-3:]

    # Combine both gripper fingers into single coordinate array
    grip_coords = np.vstack([arm_coords[-2:][::-1], x2y2])
    arm_coords = arm_coords[:3]

    return arm_coords, grip_coords


def get_sensors_coords(grip_coords, n=20):
    """
    Generate sensor point coordinates along the gripper surface.

    Creates two offset curves on either side of the gripper and
    combines them into a continuous sensor array.

    Args:
        grip_coords: Array of gripper coordinates from plot_polyline.
        n: Total number of interpolation points per side.

    Returns:
        np.ndarray: Array of shape (n*2, 2) with sensor positions.
    """
    n_2 = n // 2

    # Generate offset curves on both sides of gripper
    pts1 = interp(generate_offset_points(grip_coords, distance=-0.1), n)
    pts2 = interp(generate_offset_points(grip_coords, distance=0.1), n)
    pts1 = pts1[::-1]

    # Combine into continuous sensor array
    points = np.vstack([pts1[n_2:], pts2, pts1[:n_2]])
    return points


def plot_somatosensory(ax, weights, px, py, sensor_points):
    """
    Display somatosensory activation pattern on the given axis.

    Renders sensor activations as scatter points with sizes
    proportional to activation strength.

    Args:
        ax: Matplotlib axis for plotting.
        weights: Dictionary containing 'ssensory' weight data.
        px: X index into the weight grid.
        py: Y index into the weight grid.
        sensor_points: Array of sensor point coordinates.
    """
    sensors = weights["ssensory"][px, py]
    ax.clear()
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.5, 1.5)

    # Plot active sensors with size proportional to activation
    idcs = np.where(sensors > 1e-5)
    ax.scatter(*sensor_points[idcs].T, c="black", s=0.5 + 60 * sensors[idcs])

    # Plot all sensor positions as reference points
    ax.scatter(*sensor_points.T, c="black", s=0.1)
    ax.set_axis_off()


def plot_proprioceptive(ax, weights, px, py):
    """
    Display proprioceptive gripper pose visualization.

    Renders the gripper configuration corresponding to the
    proprioceptive weights at the specified grid position.

    Args:
        ax: Matplotlib axis for plotting.
        weights: Dictionary containing 'proprio' weight data.
        px: X index into the weight grid.
        py: Y index into the weight grid.
    """
    angles = weights["proprio"][px, py]
    grips = g.generate_gripper(angles)
    ax.clear()
    ax.set_xlim(-1.1, 0.1)
    ax.set_ylim(0.7, 1.3)

    # Draw each gripper segment
    for grip in grips:
        ax.scatter(*grip.T, c="black")
        ax.plot(*grip.T, c="black")
    ax.set_axis_off()


def plot_retina(ax, weights, px, py):
    """
    Display decoded visual retina image.

    Normalizes and renders the visual weight data as an image.

    Args:
        ax: Matplotlib axis for plotting.
        weights: Dictionary containing 'visual' weight data.
        px: X index into the weight grid.
        py: Y index into the weight grid.
    """
    retina = weights["visual"][px, py]

    # Normalize to [0, 1] range for display
    retina = (retina - retina.min()) / np.ptp(retina)
    ax.clear()
    ax.imshow(retina, aspect="auto")
    ax.set_axis_off()


def load_and_process_data(trajectory_file="trajectories.csv", weight_file="weights.npy"):
    """
    Load trajectory data and reshape weight matrices for visualization.

    Reads CSV trajectory data and numpy weight arrays, then reshapes
    the weight matrices into the required grid format.

    Args:
        trajectory_file: Path to CSV file containing trajectory data.
        weight_file: Path to numpy file containing weight matrices.

    Returns:
        tuple: (df, has_sensors, weights) where df is the trajectory
            DataFrame, has_sensors indicates if sensor data exists,
            and weights is the reshaped weight dictionary.
    """
    df = pd.read_csv(trajectory_file)
    has_sensors = "s0" in df.columns

    weights = np.load(weight_file, allow_pickle=True)[0]

    # Reshape proprioceptive weights to (10, 10, 2) grid
    dim, _ = weights["proprio"].shape
    weights["proprio"] = weights["proprio"].reshape(dim, 10, 10)
    weights["proprio"] = weights["proprio"].transpose(1, 2, 0)
    weights["proprio"] = weights["proprio"][:, :, -2:]

    # Reshape somatosensory weights to (10, 10, N) grid
    dim, _ = weights["ssensory"].shape
    weights["ssensory"] = weights["ssensory"].reshape(dim, 10, 10)
    weights["ssensory"] = weights["ssensory"].transpose(1, 2, 0)

    # Reshape visual weights to (10, 10, 3, 10, 10) grid
    weights["visual"] = weights["visual"].reshape(10, 10, 3, 10, 10)
    weights["visual"] = weights["visual"].transpose(3, 4, 0, 1, 2)

    return df, has_sensors, weights


def create_figure_layout():
    """
    Create figure with grid layout for visualization panels.

    Sets up a multi-panel figure with axes for weight maps,
    decoded representations, trajectory video, and trace plots.

    Returns:
        tuple: (fig, axes, xlims, ylims, sensor_points) where axes
            is a dictionary mapping panel names to axis objects.
    """
    xlims = np.array([-0.5, 3])
    ylims = np.array([-2.5, 2])

    gridsize = (3, 5)
    fig = plt.figure(figsize=(8 * 1.666, 8))

    # Create weight map axes (left column)
    pmap_ax = plt.subplot2grid(gridsize, [0, 0], 1, 1, fig=fig)
    smap_ax = plt.subplot2grid(gridsize, [2, 0], 1, 1, fig=fig)
    vmap_ax = plt.subplot2grid(gridsize, [1, 0], 1, 1, fig=fig)
    pmap_ax.set_axis_off()
    smap_ax.set_axis_off()
    vmap_ax.set_axis_off()

    pmap_ax.set_title("Proprioception")
    smap_ax.set_title("Somatosensory")
    vmap_ax.set_title("Foveal vision")
    # Create decoded representation axes (second column)
    proprio_ax = plt.subplot2grid(gridsize, [0, 1], 1, 1, fig=fig)
    ssensory_ax = plt.subplot2grid(gridsize, [2, 1], 1, 1, fig=fig)
    visual_ax = plt.subplot2grid(gridsize, [1, 1], 1, 1, fig=fig)

    sensor_points = g.generate_sensor_points(40)

    # Create main video and trace axes (right side)
    video_ax = plt.subplot2grid(gridsize, [0, 2], 3, 3, fig=fig, aspect="equal")
    traces_ax = plt.subplot2grid(gridsize, [2, 2], 1, 1, fig=fig, aspect="equal")
    traces_ax.set_xlim(-0.5, 9.5)
    traces_ax.set_ylim(-0.5, 9.5)
    traces_ax.set_xticks(np.arange(10), [])
    traces_ax.set_yticks(np.arange(10), [])
    traces_ax.grid()

    fig.tight_layout(pad=0.3)

    axes = {
        "pmap": pmap_ax,
        "smap": smap_ax,
        "vmap": vmap_ax,
        "traces": traces_ax,
        "proprio": proprio_ax,
        "ssensory": ssensory_ax,
        "visual": visual_ax,
        "video": video_ax,
    }
    return fig, axes, xlims, ylims, sensor_points


def setup_ax(ax, xlims=None, ylims=None):
    """
    Clear and configure axis with optional limits.

    Args:
        ax: Matplotlib axis to configure.
        xlims: Optional tuple of (xmin, xmax) limits.
        ylims: Optional tuple of (ymin, ymax) limits.
    """
    ax.clear()
    if xlims is not None:
        ax.set_xlim(xlims)
    if ylims is not None:
        ax.set_ylim(ylims)
    ax.set_axis_off()


def add_marker(ax, point, width, height):
    """
    Add a rectangular marker to highlight a grid position.

    Args:
        ax: Matplotlib axis to add marker to.
        point: Bottom-left corner coordinates of the rectangle.
        width: Rectangle width.
        height: Rectangle height.
    """
    rp = Rectangle(point - 0.5, width=width, height=height, fc="#fff0", ec="red")
    ax.add_patch(rp)


def update_maps(g, wfile, pmap_ax, vmap_ax, smap_ax, px, py):
    """
    Update all weight map visualizations with current position marker.

    Args:
        g: GraphManager instance for generating map visualizations.
        wfile: Path to weights file.
        pmap_ax: Axis for proprioceptive map.
        vmap_ax: Axis for visual map.
        smap_ax: Axis for somatosensory map.
        px: X index of current position in weight grid.
        py: Y index of current position in weight grid.
    """
    # Update proprioceptive map with marker
    setup_ax(pmap_ax)
    g.proprio_map(ax=pmap_ax, wfile=wfile)
    add_marker(pmap_ax, np.array([py, px]) + [-0.5, 1], 1, 1)

    # Update visual map with marker (note coordinate transformation)
    setup_ax(vmap_ax)
    g.visual_map(ax=vmap_ax, wfile=wfile)
    add_marker(vmap_ax, np.array([py, 10 - px - 1]) * 10 + 0.5, 10, 10)

    # Update somatosensory map with marker
    setup_ax(smap_ax)
    g.somatosensory_map(ax=smap_ax, wfile=wfile)
    add_marker(smap_ax, np.array([py, px]) + 0.5, 1, 1)

    pmap_ax.set_title("Proprioception")
    smap_ax.set_title("Somatosensory")
    vmap_ax.set_title("Foveal vision")


class TrajectoryAnimator:
    """
    Handles animation of arm trajectories with sensor visualization.

    Manages the creation and updating of matplotlib artists for
    animating arm movements, sensor activations, and trajectory traces.

    Attributes:
        video_ax: Main axis for arm animation.
        traces_ax: Axis for trajectory trace plots.
        fig: Parent matplotlib figure.
        has_sensors: Whether sensor data is available.
        xlims: X-axis limits for video axis.
        ylims: Y-axis limits for video axis.
    """

    def __init__(self, params, video_ax, traces_ax, fig, has_sensors, xlims, ylims):
        """
        Initialize the trajectory animator.

        Args:
            video_ax: Matplotlib axis for main animation.
            traces_ax: Matplotlib axis for trajectory traces.
            fig: Parent matplotlib figure.
            has_sensors: Boolean indicating sensor data availability.
            xlims: Tuple of (xmin, xmax) for video axis.
            ylims: Tuple of (ymin, ymax) for video axis.
        """
        self.video_ax = video_ax
        self.traces_ax = traces_ax
        self.fig = fig
        self.has_sensors = has_sensors
        self.xlims = xlims
        self.ylims = ylims

        # Artist containers
        self.lines1 = []
        self.lines2 = []
        self.scatters = []
        self.reps = {}
        self.traces = {}
        self._initialized = False

        # Load episode image paths
        script_dir = os.path.dirname(os.path.abspath(__file__))
        self.episodes = glob.glob(f"{script_dir}/data/e_*")
        self.episode_df, self.conditions_df = build_episode_dataset(params)
        self.episodes.sort()

        self._anim = None

    def _init_artists(self, n):
        """
        Initialize matplotlib artists for n trajectory frames.

        Creates line and scatter objects for arm segments, gripper,
        sensors, and trajectory traces.

        Args:
            n: Number of frames in the trajectory.
        """
        goal_color = "#ff2"
        touch_color = "#f22"
        proprio_color = "#22f"

        for _ in range(n):
            # Create arm and gripper line artists
            (line1,) = self.video_ax.plot([], [], c="black", marker="o")
            (line2,) = self.video_ax.plot([], [], c="black", marker="o")
            self.lines1.append(line1)
            self.lines2.append(line2)

            # Create sensor scatter artist if needed
            if self.has_sensors:
                scatter = self.video_ax.scatter([], [], c="red")
                self.scatters.append(scatter)

            # Create trace lines for goal, touch, and proprioception
            self.traces = {
                "g": self.traces_ax.plot([999, 999], [999, 999], c=goal_color)[0],
                "ss": self.traces_ax.plot([999, 999], [999, 999], c=touch_color)[0],
                "p": self.traces_ax.plot([999, 999], [999, 999], c=proprio_color)[0],
            }

            # Create marker scatter artists for current positions
            self.reps = {
                "g": self.traces_ax.scatter(
                    999, 999, marker="h", fc=goal_color, ec="#000", lw=0.5, s=300
                ),
                "ss": self.traces_ax.scatter(
                    999, 999, marker="*", fc=touch_color, ec="#000", lw=0.5, s=300
                ),
                "p": self.traces_ax.scatter(
                    999, 999, marker="*", fc=proprio_color, ec="#000", lw=0.5, s=300
                ),
            }

        self._initialized = True

    def clear(self):
        """Clear all artist data for reuse."""
        for line in self.lines1:
            line.set_data([], [])
        for line in self.lines2:
            line.set_data([], [])
        for scatter in self.scatters:
            scatter.set_offsets(np.empty((0, 2)))
            scatter.set_sizes([])

    def render_episode(self, n):
        """
        Render episode image overlay in corner of video axis.

        Args:
            n: Episode index to render.
        """
        episode = self.episodes[n]
        img = plt.imread(episode)

        # Calculate image placement in bottom-right corner
        xlim = self.video_ax.get_xlim()
        ylim = self.video_ax.get_ylim()

        scale = 0.2
        ax_width = xlim[1] - xlim[0]
        ax_height = ylim[1] - ylim[0]

        extent = [
            xlim[1] - scale * ax_width,
            xlim[1],
            ylim[0],
            ylim[0] + scale * ax_height,
        ]

        self.video_ax.imshow(img, extent=extent, aspect="auto", zorder=10)

        objs = ["blue cube", "red triangle", "green cube"]
        # stretches = ["no stretch", "double"]
        # rots = ["no rotation", "45°"]
        obj = objs[self.episode_df.query(f"index=={n}").context.iloc[0] - 1]
        stretch = self.episode_df.query(f"index=={n}").stretch.iloc[0]
        rot = self.episode_df.query(f"index=={n}").rotation.iloc[0]
        self.video_ax.text(
            xlim[1] - 2 * scale * ax_width,
            ylim[0] + scale * ax_height,
            f" Object: {obj}\nStretch: {stretch}\nrotation: {np.degrees(rot)}\n",
        )

    def animate(self, trajectory):
        """
        Create and display animation for the given trajectory.

        Processes trajectory data, initializes artists, and runs
        the matplotlib animation loop.

        Args:
            trajectory: DataFrame containing trajectory data with
                columns for joint angles, sensor values, and positions.
        """
        n = trajectory.shape[0]
        ts_vals = trajectory.ts.to_numpy()
        data = trajectory.iloc[:, 1:6].to_numpy()

        # Extract sensor data if available
        sensor_data = trajectory.iloc[:, 6:46].to_numpy() if self.has_sensors else None

        # Extract position traces (note axis swap)
        ss_data = trajectory.loc[:, ["touch_x", "touch_y"]].to_numpy()[:, ::-1]
        p_data = trajectory.loc[:, ["proprio_x", "proprio_y"]].to_numpy()[:, ::-1]
        g_data = trajectory.loc[:, ["prototype_x", "prototype_y"]].to_numpy()[:, ::-1]

        # Reinitialize artists if frame count changed
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
            self._init_artists(n)

        # Compute alpha values for fade effect (recent frames more opaque)
        exp_coeff = -((n / 100) ** -2)
        n_minus_1 = n - 1
        indices = ts_vals.astype(int)
        all_angles = np.degrees(data[indices])
        self.polylines = [plot_polyline(ang, [1, 1, 0.5, 0.5]) for ang in all_angles]
        alphas = 0.2 + 0.8 * np.exp(exp_coeff * (np.arange(n) / n_minus_1 - 1) ** 2)

        # Precompute sensor visualization data
        offsets_pts = []
        sizes_arr = []
        if self.has_sensors:
            all_sensors = sensor_data[indices]
            for i, (arm_coords, grip_coords) in enumerate(self.polylines):
                pts = get_sensors_coords(grip_coords)
                offsets_pts.append(pts)
                ssensors = all_sensors[i]
                sizes_arr.append(100 * ssensors)

        self.render_episode(trajectory.episode_id.iloc[0])

        def update(frame_idx):
            """Update function called for each animation frame."""
            artists = []
            for i in range(frame_idx + 1):
                arm_coords, grip_coords = self.polylines[i]
                alpha = alphas[i]

                # Update arm segment lines
                self.lines1[i].set_data(arm_coords[:, 0], arm_coords[:, 1])
                self.lines1[i].set_alpha(alpha)
                self.lines2[i].set_data(grip_coords[:, 0], grip_coords[:, 1])
                self.lines2[i].set_alpha(alpha)

                # Draw trajectory traces
                p = self.traces_ax.plot(*ss_data[:i].T, c="#f22", zorder=-3)
                p.extend(self.traces_ax.plot(*p_data[:i].T, c="#22f", zorder=-3))

                # Update current position markers
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

                # Update sensor visualization if available
                if self.has_sensors:
                    self.scatters[i].set_offsets(offsets_pts[i])
                    self.scatters[i].set_sizes(sizes_arr[i])
                    self.scatters[i].set_alpha(alpha)
                    artists.append(self.scatters[i])
                artists.extend(p)
            return artists

        self.anim = FuncAnimation(
            self.fig, update, frames=n, interval=500, blit=True, repeat=False
        )
        return self.anim


if __name__ == "__main__":
    # Parse command line arguments for episode selection
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
    args = parser.parse_args()

    # Initialize parameters and graph manager
    params = Parameters()
    sm = None
    g = GraphManager(sm, params)

    # Load data and extract requested trajectory
    df, has_sensors, weights = load_and_process_data(trajectory_file="trajectory_df.csv")
    wfile = "weights.npy"

    trajectory = df[(df.episode_id == args.episode_id) & (df.goal_id == args.goal_id)]

    # Handle multiple repetitions if present
    if "e_seed" in trajectory.columns:
        seeds = trajectory.e_seed.unique()
        cur_seed = seeds[args.rep]
        trajectory = trajectory.query(f"e_seed=={cur_seed}")

    # # Open pre-rendered GIF if available
    # gif_path = f"rendered_episodes/episode_{args.episode_id}_{args.rep+1}.gif"
    # print(gif_path)
    # if Path(gif_path).exists():
    #     subprocess.Popen(
    #         ["imv-x11", gif_path],
    #         start_new_session=True,
    #         close_fds=True,
    #     )

    # Set up visualization layout
    fig, axes, xlims, ylims, sensor_points = create_figure_layout()
    setup_ax(axes["video"], xlims, ylims)
    px = int(trajectory.prototype_x.iat[0])
    py = int(trajectory.prototype_y.iat[0])

    # Render weight maps and decoded representations
    update_maps(g, wfile, axes["pmap"], axes["vmap"], axes["smap"], px, py)
    plot_proprioceptive(axes["proprio"], weights, px, py)
    plot_somatosensory(axes["ssensory"], weights, px, py, sensor_points)
    plot_retina(axes["visual"], weights, px, py)

    # Create and run animation
    animator = TrajectoryAnimator(
        params, axes["video"], axes["traces"], fig, has_sensors, xlims, ylims
    )
    anim = animator.animate(trajectory)

    anim.save(filename=f"postures_e{args.episode_id}_g{args.goal_id}_{args.rep}.gif", writer="pillow")
