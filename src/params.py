import numpy as np

from parameter_manager import ParameterManager


class Parameters(ParameterManager):
    def __init__(
        self,
        task_space=None,  # {"xlim": [-10, 50], "ylim": [-10, 50]}
        stime=200,
        drop_first_n_steps=25,
        policy_selection_steps=25,
        env_reset_freq=2,
        esn_tau=5.0,
        esn_alpha=0.2,
        esn_epsilon=1.0e-30,
        arm_input=2,
        arm_hidden=100,
        arm_output=3,
        grip_input=47,
        grip_hidden=75,
        grip_output=5,
        grip_map_name="data/StoredGripActuatorMap",
        grip_encoding="linear",
        grip_input_norm="none",
        grip_out_fn="clip",
        grip_norm_name="data/StoredGripActuatorNorm",
        grip_touch_gain=1.0,
        grip_basis_name=None,
        grip_policy_dim=None,
        policy_winner="argmin",
        map_winner="argmin",
        map_update="adam",
        map_ema_eta=0.05,
        policy_update="adam",
        policy_ema_eta=0.05,
        internal_size=100,
        visual_size=300,
        somatosensory_size=40,
        proprioception_size=5,
        num_objects=4,
        v_eradial_prop=0.1,
        ss_eradial_prop=0.1,
        p_eradial_prop=0.1,
        a_eradial_prop=0.1,
        policy_params_amplitude=8.0,
        decay=5.0,
        local_decay=5.0,
        base_match_sigma=1.5,
        match_sigma=1.5,
        base_internal_sigma=0.1,
        internal_sigma=10.0,
        base_lr=0.005,
        max_lr=1.0,
        stm_lr=0.1,
        v_lr=1.0,
        ss_lr=1.0,
        p_lr=1.0,
        a_lr=1.0,
        policy_base=0.8,
        policy_base_arm=[x * np.pi for x in [0.2, -0.3, -0.4]],
        base_policy_noise=0.01,
        max_policy_noise=20.0,
        policy_weights_sigma=2.0,
        motor_noise=1.0,
        representation_sigma=2.0,
        modalities_weights=(0.0, 1.0, 1.0, 0.0),  # use only touch and proprioception
        match_mode="arithmetic",
        touch_gate=False,
        touch_incr_th=0.0,
        policy_update_min_match=0.0,
        policy_update_min_mod_match=0.0,
        policy_noise_subspace_k=0,
        policy_noise_floor=0.1,
        policy_output_space=False,
        policy_noise_reach_k=0,
        policy_update_baseline=0.0,
        match_incr_th=0.001,
        cum_match_stop_th=1.0,
        predict_lr=0.05,
        epochs=1000,
        batch_size=18,
        tests=18,
        evaluation_episodes=360,
        demo_episodes_max_single_goal=5,
        demo_episodes_max_trials=100,
        epochs_to_test=100,
        load_weights=False,
        shuffle_weights=False,
        action_steps=5,
        obj_fix_prob=2.0,
        obj_stretch_conditions=[1, 2],
        obj_rotation_conditions=[0, 0.785398, 1.570796],
        obj_x=2,
        obj_y=0.5,
        maximum_goal_activation=575.0,
    ):

        self.task_space = (
            {"xlim": [-10, 50], "ylim": [-10, 50]} if task_space is None else task_space
        )
        self.stime = stime
        self.drop_first_n_steps = drop_first_n_steps
        self.policy_selection_steps = policy_selection_steps
        self.env_reset_freq = env_reset_freq
        self.esn_tau = esn_tau
        self.esn_alpha = esn_alpha
        self.esn_epsilon = esn_epsilon
        self.arm_input = arm_input
        self.arm_hidden = arm_hidden
        self.arm_output = arm_output
        self.grip_input = grip_input
        self.grip_hidden = grip_hidden
        self.grip_output = grip_output
        self.grip_map_name = grip_map_name
        self.grip_encoding = grip_encoding
        self.grip_input_norm = grip_input_norm
        self.grip_out_fn = grip_out_fn
        self.grip_norm_name = grip_norm_name
        self.grip_touch_gain = grip_touch_gain
        self.grip_basis_name = grip_basis_name
        self.grip_policy_dim = grip_policy_dim
        self.policy_winner = policy_winner
        self.map_winner = map_winner
        self.map_update = map_update
        self.map_ema_eta = map_ema_eta
        self.policy_update = policy_update
        self.policy_ema_eta = policy_ema_eta
        self.internal_size = internal_size
        self.visual_size = visual_size
        self.somatosensory_size = somatosensory_size
        self.proprioception_size = proprioception_size
        self.policy_size = (grip_policy_dim or grip_hidden) * grip_output
        self.num_objects = num_objects
        self.v_eradial_prop = v_eradial_prop
        self.ss_eradial_prop = ss_eradial_prop
        self.p_eradial_prop = p_eradial_prop
        self.a_eradial_prop = a_eradial_prop
        self.policy_params_amplitude = policy_params_amplitude
        self.decay = decay
        self.local_decay = local_decay
        self.base_match_sigma = base_match_sigma
        self.match_sigma = match_sigma
        self.base_internal_sigma = base_internal_sigma
        self.internal_sigma = internal_sigma
        self.base_lr = base_lr
        self.max_lr = max_lr
        self.stm_lr = stm_lr
        self.v_lr = v_lr
        self.ss_lr = ss_lr
        self.p_lr = p_lr
        self.a_lr = a_lr
        self.policy_base_arm = policy_base_arm
        self.policy_base = policy_base
        self.base_policy_noise = base_policy_noise
        self.max_policy_noise = max_policy_noise
        self.policy_weights_sigma = policy_weights_sigma
        self.motor_noise = motor_noise
        self.representation_sigma = representation_sigma
        self.modalities_weights = modalities_weights
        self.match_mode = match_mode
        self.touch_gate = touch_gate
        self.touch_incr_th = touch_incr_th
        self.policy_update_min_match = policy_update_min_match
        self.policy_update_min_mod_match = policy_update_min_mod_match
        self.policy_noise_subspace_k = policy_noise_subspace_k
        self.policy_noise_floor = policy_noise_floor
        self.policy_output_space = policy_output_space
        self.policy_noise_reach_k = policy_noise_reach_k
        self.policy_update_baseline = policy_update_baseline
        self.match_incr_th = match_incr_th
        self.cum_match_stop_th = cum_match_stop_th
        self.predict_lr = predict_lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.tests = tests
        self.epochs_to_test = epochs_to_test
        self.load_weights = load_weights
        self.shuffle_weights = shuffle_weights
        self.action_steps = action_steps
        self.obj_fix_prob = obj_fix_prob
        self.obj_stretch_conditions = obj_stretch_conditions
        self.obj_rotation_conditions = obj_rotation_conditions
        self.obj_x = obj_x
        self.obj_y = obj_y
        self.maximum_goal_activation = maximum_goal_activation
        self.evaluation_episodes = evaluation_episodes
        self.demo_episodes_max_single_goal = demo_episodes_max_single_goal
        self.demo_episodes_max_trials = demo_episodes_max_trials

        super(Parameters, self).__init__()


if __name__ == "__main__":
    # Use case 1: Initialize, update, and save parameters to a file.
    param_string = "epochs=2;predict_lr=0.2"
    param_file = "tmp_file"
    p1 = Parameters()
    p1.update(param_string)
    p1.save(param_file)
    p1.save(param_file + ".json", mode="json")

    # Use case 2: Load parameters from a file.
    p2 = Parameters()
    p2.load(param_file)
    p2.load(param_file + ".json", mode="json")

    # Use case 3: Update parameters using a JSON string.
    p1.update(repr(p2), mode="json")
