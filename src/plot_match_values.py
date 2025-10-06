import sys
import numpy as np
import matplotlib.pyplot as plt


def make_plots(simulation_data_file):
    # movable: 1, 0, 0
    # controllable: 0, 1, 0
    # still: 0, 0, 1
    # unreachable: 1, 1, 0

    # modalities: visual, touch, proprioception, action, goal

    data = np.load(simulation_data_file, allow_pickle=True)

    match_inc_per_mod = data[0]["match_increment_per_mod"].reshape((-1, 200, 5, 5))
    match_inc_visual = match_inc_per_mod[:, :, 0, -1]
    match_inc_touch = match_inc_per_mod[:, :, 1, -1]
    match_inc_proprio = match_inc_per_mod[:, :, 2, -1]
    match_inc_action = match_inc_per_mod[:, :, 3, -1]

    match_values_per_mod = data[0]["match_value_per_mod"].reshape((-1, 200, 5, 5))
    match_values_visual = match_values_per_mod[:, :, 0, -1]
    match_values_touch = match_values_per_mod[:, :, 1, -1]
    match_values_proprio = match_values_per_mod[:, :, 2, -1]
    match_values_action = match_values_per_mod[:, :, 3, -1]

    match_values = data[0]["match_value"].reshape((-1, 200))
    match_inc = data[0]["match_increment"].reshape((-1, 200))

    visual_som_activations = data[0]["v_r"].reshape((-1, 200, 10, 10))
    visual_som_winner = visual_som_activations.reshape(-1, 100).argmax(axis=1)

    visual_input = data[0]["v"].reshape((-1, 200, 100, 3))
    visual_input0 = visual_input[:, 0]

    movable_mask = (visual_input0[:, :, 0].round(0).min(axis=1) == 0)
    controllable_mask = (visual_input0[:, :, 1].round(0).min(axis=1) == 0)
    still_mask = (visual_input0[:, :, 2].round(0).min(axis=1) == 0)

    match_movable = match_values[movable_mask, :]
    match_controllable = match_values[controllable_mask, :]
    match_still = match_values[still_mask, :]

    plt.plot(match_movable.mean(axis=0), color="r", label="movable")
    plt.plot(match_controllable.mean(axis=0), color="g", label="controllable")
    plt.plot(match_still.mean(axis=0), color="b", label="still")
    plt.legend()
    plt.savefig("match_value_plot.png")
    plt.close()

    plt.plot(match_values_visual[movable_mask].mean(axis=0), label="visual")
    plt.plot(match_values_touch[movable_mask].mean(axis=0), label="touch")
    plt.plot(match_values_proprio[movable_mask].mean(axis=0), label="proprio")
    plt.plot(match_values_action[movable_mask].mean(axis=0), label="action")
    plt.legend()
    plt.savefig("match_per_mod_movable_plot.png")
    plt.close()

    plt.plot(match_values_visual[controllable_mask].mean(axis=0), label="visual")
    plt.plot(match_values_touch[controllable_mask].mean(axis=0), label="touch")
    plt.plot(match_values_proprio[controllable_mask].mean(axis=0), label="proprio")
    plt.plot(match_values_action[controllable_mask].mean(axis=0), label="action")
    plt.legend()
    plt.savefig("match_per_mod_controllable_plot.png")
    plt.close()

    plt.plot(match_values_visual[still_mask].mean(axis=0), label="visual")
    plt.plot(match_values_touch[still_mask].mean(axis=0), label="touch")
    plt.plot(match_values_proprio[still_mask].mean(axis=0), label="proprio")
    plt.plot(match_values_action[still_mask].mean(axis=0), label="action")
    plt.legend()
    plt.savefig("match_per_mod_still_plot.png")
    plt.close()

    plt.plot(match_inc[movable_mask].mean(axis=0), color="r", label="movable")
    plt.plot(match_inc[controllable_mask].mean(axis=0), color="g", label="controllable")
    plt.plot(match_inc[still_mask].mean(axis=0), color="b", label="still")
    plt.legend()
    plt.savefig("match_inc_plot.png")
    plt.close()

    plt.plot(match_inc_visual[movable_mask].mean(axis=0), label="visual")
    plt.plot(match_inc_touch[movable_mask].mean(axis=0), label="touch")
    plt.plot(match_inc_proprio[movable_mask].mean(axis=0), label="proprio")
    plt.plot(match_inc_action[movable_mask].mean(axis=0), label="action")
    plt.legend()
    plt.savefig("match_inc_per_mod_movable_plot.png")
    plt.close()

    plt.plot(match_inc_visual[controllable_mask].mean(axis=0), label="visual")
    plt.plot(match_inc_touch[controllable_mask].mean(axis=0), label="touch")
    plt.plot(match_inc_proprio[controllable_mask].mean(axis=0), label="proprio")
    plt.plot(match_inc_action[controllable_mask].mean(axis=0), label="action")
    plt.legend()
    plt.savefig("match_inc_per_mod_controllable_plot.png")
    plt.close()

    plt.plot(match_inc_visual[still_mask].mean(axis=0), label="visual")
    plt.plot(match_inc_touch[still_mask].mean(axis=0), label="touch")
    plt.plot(match_inc_proprio[still_mask].mean(axis=0), label="proprio")
    plt.plot(match_inc_action[still_mask].mean(axis=0), label="action")
    plt.legend()
    plt.savefig("match_inc_per_mod_still_plot.png")
    plt.close()


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} filename\n")
        sys.exit(1)

    make_plots(sys.argv[1])
