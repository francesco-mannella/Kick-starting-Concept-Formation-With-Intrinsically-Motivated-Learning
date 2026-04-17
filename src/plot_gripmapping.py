import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from plot_postures import plot_polyline, plot_somatosensory
from SMGraphs import GraphManager
from SMMain import Main


# Ensure Main module is loaded (side effect import)
_ = Main

# Use Qt backend for interactive plotting
matplotlib.use("qtagg")


# =============================================================================
# Data Loading and Configuration
# =============================================================================


def load_main_data(filepath="main.dump.npy"):
    """Load serialized main object from numpy file."""
    return np.load(filepath, allow_pickle=True)[0]


def extract_weight_mapping(weights, touch_size, posture_size, subdomain_shape):
    """
    Extract and reshape weight matrices for each sensory modality.

    Returns dict with keys: 'ssensory', 'proprio', 'pos'
    Each value is reshaped to (subdomain_side, subdomain_side, input_dim).
    """
    subdomain_side = int(np.sqrt(subdomain_shape))
    posture_start = touch_size
    posture_end = touch_size + posture_size

    mapping = {
        "ssensory": weights[:touch_size, :subdomain_shape],
        "proprio": np.degrees(
            weights[posture_start:posture_end, subdomain_shape : 2 * subdomain_shape]
        ),
        "pos": weights[posture_end:, 2 * subdomain_shape :],
    }

    # Reshape each mapping to grid format
    for key, val in mapping.items():
        val = val.reshape(-1, subdomain_side, subdomain_side)
        mapping[key] = val.transpose(1, 2, 0)

    return mapping, subdomain_side


# =============================================================================
# Figure Setup Utilities
# =============================================================================


def create_figure_grid(subdomain_side, figsize=(4, 4)):
    """Create a figure with subdomain_side x subdomain_side subplots."""
    return plt.subplots(subdomain_side, subdomain_side, figsize=figsize)


def configure_somatosensory_axes(axes):
    """Configure axes for somatosensory visualization (no axis lines)."""
    for ax in axes.flatten():
        ax.set_axis_off()


def configure_posture_axes(axes):
    """Configure axes for arm posture visualization."""
    for ax in axes.flatten():
        ax.set_axis_off()
        ax.set_aspect("equal")
        ax.set_xlim(-1, 4)
        ax.set_ylim(-1, 4)


def configure_position_axes(axes):
    """Configure axes for position visualization with grid overlay."""
    for ax in axes.flatten():
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlim(-5, 30)
        ax.set_ylim(-5, 30)
        _draw_position_grid(ax)


def _draw_position_grid(ax):
    """Draw grid lines and border for position subplot."""
    # Inner grid lines
    for i in range(0, 30, 5):
        ax.plot([-5, 30], [i, i], c="black", lw=0.5)
        ax.plot([i, i], [-5, 30], c="black", lw=0.5)
    # Border lines
    for i in [-5, 30]:
        ax.plot([-5, 30], [i, i], c="black", lw=3)
        ax.plot([i, i], [-5, 30], c="black", lw=3)


# =============================================================================
# Plotting Functions
# =============================================================================


def plot_arm_posture(ax, angles, arm_lengths):
    """Plot arm and gripper polyline on given axis."""
    arm_coords, grip_coords = plot_polyline(angles, arm_lengths)
    ax.plot(*arm_coords.T, "o-", c="black", markersize=4)
    ax.plot(*grip_coords.T, "o-", c="black", markersize=4)


def plot_position_marker(ax, position):
    """Plot concentric circles marking a position."""
    ax.scatter(*position, c="#faa", s=700)  # Outer halo
    ax.scatter(*position, c="#e44", s=250)  # Middle ring
    ax.scatter(*position, c="black", s=40)  # Center dot


def render_all_subplots(
    axes1, axes2, axes3, mapping, subdomain_side, sensor_points, arm_lengths
):
    """Render visualizations across all subplot grids."""
    for i in range(subdomain_side):
        for j in range(subdomain_side):
            # Somatosensory map
            plot_somatosensory(axes1[i, j], mapping, i, j, sensor_points)

            # Arm posture
            angles = mapping["proprio"][i, j]
            plot_arm_posture(axes2[i, j], angles, arm_lengths)

            # Position marker
            plot_position_marker(axes3[i, j], mapping["pos"][i, j])


# =============================================================================
# Main Execution
# =============================================================================

# Load data
main = load_main_data()
params = main.params

# Generate sensor points for visualization
graph_manager = GraphManager(None, params)
sensor_points = graph_manager.generate_sensor_points(40)

# Extract network weights and compute dimensions
weights = main.agent.grip_agent.grip.map
inner_domain_shape = weights.shape[1]
subdomain_shape = inner_domain_shape // 3

# Build weight mappings for each modality
mapping, subdomain_side = extract_weight_mapping(
    weights,
    touch_size=params.somatosensory_size,
    posture_size=5,
    subdomain_shape=subdomain_shape,
)

# Create and configure figure grids
fig1, axes1 = create_figure_grid(subdomain_side)  # Somatosensory
fig2, axes2 = create_figure_grid(subdomain_side)  # Posture
fig3, axes3 = create_figure_grid(subdomain_side)  # Position

configure_somatosensory_axes(axes1)
configure_posture_axes(axes2)
configure_position_axes(axes3)

fig1.tight_layout(pad=0)
fig2.tight_layout(pad=0)
fig3.tight_layout(pad=0)

# Render all visualizations
arm_lengths = [1, 1, 1, 0.5, 0.5]
render_all_subplots(
    axes1, axes2, axes3, mapping, subdomain_side, sensor_points, arm_lengths
)

fig1.savefig("somatosensory.svg")
fig2.savefig("posture.svg")
fig3.savefig("position.svg")
