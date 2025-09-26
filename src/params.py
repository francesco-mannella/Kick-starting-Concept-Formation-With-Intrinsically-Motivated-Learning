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
        grip_input=44,
        grip_hidden=100,
        grip_output=5,
        internal_size=100,
        visual_size=300,
        somatosensory_size=40,
        proprioception_size=5,
        policy_size=100 * 5,
        num_objects=4,
        v_eradial_prop=0.1,
        ss_eradial_prop=0.1,
        p_eradial_prop=0.1,
        a_eradial_prop=0.1,
        policy_params_amplitude=8.0,
        decay=3.0,
        local_decay=1.0,
        base_match_sigma=5,
        match_sigma=5,
        base_internal_sigma=0.5 * np.sqrt(2),
        internal_sigma=40.0,
        base_lr=0.005,
        max_lr=2.0,
        stm_lr=0.1,
        policy_base_arm=np.pi * 0.01,
        policy_base=np.pi * 0.25,
        reach_grip_prop=0.3,
        base_policy_noise=0.02,
        max_policy_noise=0.6,
        policy_weights_sigma=2,
        motor_noise=1.0,
        representation_sigma=2,
        modalities_weights=None,  # [1.0, 1.0, 1.0, 1.0]
        match_incr_th=0.001,
        cum_match_stop_th=3.0,
        predict_lr=0.05,
        epochs=1000,
        batch_size=24,
        tests=12,
        evaluation_episodes=100,
        demo_episodes_max_single_goal=5,
        demo_episodes_max_trials=10000,
        epochs_to_test=100,
        load_weights=False,
        shuffle_weights=False,
        action_steps=5,
        obj_fix_prob=2.0,
        obj_stretch_conditions=[1, 1.5],
        obj_rotation_conditions=[0, 0.785398, 1.570796],
        obj_x=2,
        obj_y=0.25,
        maximum_goal_activation=575.0,
    ):

        self.task_space = (
            {"xlim": [-10, 50], "ylim": [-10, 50]}
            if task_space is None
            else task_space
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
        self.internal_size = internal_size
        self.visual_size = visual_size
        self.somatosensory_size = somatosensory_size
        self.proprioception_size = proprioception_size
        self.policy_size = policy_size
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
        self.policy_base_arm = policy_base_arm
        self.policy_base = policy_base
        self.base_policy_noise = base_policy_noise
        self.max_policy_noise = max_policy_noise
        self.policy_weights_sigma = policy_weights_sigma
        self.motor_noise = motor_noise
        self.representation_sigma = representation_sigma
        self.modalities_weights = (
            [1.0, 1.0, 1.0, 1.0]
            if modalities_weights is None
            else modalities_weights
        )
        self.match_incr_th = match_incr_th
        self.cum_match_stop_th = cum_match_stop_th
        self.predict_lr = predict_lr
        self.reach_grip_prop = reach_grip_prop
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
