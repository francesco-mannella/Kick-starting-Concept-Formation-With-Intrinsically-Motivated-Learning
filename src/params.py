import json
import sys

import numpy as np


class Parameters:
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
        explore_sigma=8.0,
        decay=3.0,
        local_decay=1.0,
        base_match_sigma=5,
        match_sigma=5,
        base_internal_sigma=0.5 * np.sqrt(2),
        internal_sigma=40.0,
        base_lr=0.005,
        max_lr=2.0,
        stm_lr=0.1,
        policy_base=np.pi * 0.25,
        base_policy_noise=0.02,
        max_policy_noise=0.6,
        policy_weights_sigma=2,
        motor_noise=1.0,
        representation_sigma=2,
        modalities_weights=None,  # [1.0, 1.0, 1.0, 1.0]
        match_incr_th=0.02,
        cum_match_stop_th=10.0,
        predict_lr=0.1,
        reach_grip_prop=0.1,
        predict_ampl=2,
        predict_base_ampl=2,
        predict_ampl_prop=0.95,
        epochs=400,
        batch_size=24,
        tests=12,
        epochs_to_test=100,
        load_weights=False,
        shuffle_weights=False,
        action_steps=5,
        obj_fix_prob=0.2,
        obj_var_prob=1.6,
        obj_rot_var=3.1415922653,
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
        self.explore_sigma = explore_sigma
        self.decay = decay
        self.local_decay = local_decay
        self.base_match_sigma = base_match_sigma
        self.match_sigma = match_sigma
        self.base_internal_sigma = base_internal_sigma
        self.internal_sigma = internal_sigma
        self.base_lr = base_lr
        self.max_lr = max_lr
        self.stm_lr = stm_lr
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
        self.predict_ampl = predict_ampl
        self.predict_base_ampl = predict_base_ampl
        self.predict_ampl_prop = predict_ampl_prop
        self.epochs = epochs
        self.batch_size = batch_size
        self.tests = tests
        self.epochs_to_test = epochs_to_test
        self.load_weights = load_weights
        self.shuffle_weights = shuffle_weights
        self.action_steps = action_steps
        self.obj_fix_prob = obj_fix_prob
        self.obj_var_prob = obj_var_prob
        self.obj_rot_var = obj_rot_var
        self.param_types = {
            k: type(v)
            for k, v in self.__dict__.items()
            if not k.startswith("__") and not callable(v)
        }

    def _string_to_json(self, param_list):
        if not param_list:
            return {}
        return dict(
            item.split("=", 1) for item in param_list.split(";") if "=" in item
        )

    def _json_to_params(self, param_dict):
        if not param_dict:
            return
        for key, value in param_dict.items():
            if key in dir(self):
                converter = self.param_types[key]
                try:
                    if isinstance(converter, type):
                        setattr(self, key, converter(value))
                    elif isinstance(converter, list):
                        setattr(
                            self,
                            key,
                            [
                                type(converter[0])(v)
                                for v in value.strip("[]()").split(",")
                            ],
                        )
                    elif isinstance(converter, tuple):
                        setattr(
                            self,
                            key,
                            tuple(
                                type(converter[0])(v)
                                for v in value.strip("[]()").split(",")
                            ),
                        )
                    elif isinstance(converter, np.ndarray):
                        setattr(
                            self,
                            key,
                            np.array(
                                [
                                    type(converter.item(0))(v)
                                    for v in value.strip("[]()").split(",")
                                ]
                            ),
                        )
                    elif isinstance(converter, dict):
                        setattr(self, key, json.loads(value))
                    elif converter is bool:
                        setattr(self, key, value == "True")
                    else:
                        setattr(self, key, converter(value))
                except Exception as e:
                    print(f"Error converting parameter {key}: {e}")
            else:
                print(f"There's no parameter named {key}")
                sys.exit(1)

    def update(self, param_string):
        self.

    def save(self, filepath):
        params = {
            key: getattr(self, key)
            for key in self.__dict__
            if key != "param_types"
        }
        with open(filepath, "w") as file:
            json.dump(params, file, indent=4)

    def load(self, filepath):
        with open(filepath, "r") as file:
            param_list = "".join([line.strip() + ";" for line in file])
        self.string_to_params(param_list)

    def __hash__(self):
        # Using a tuple comprehension to collect all non-callable and
        # non-private attributes (those not starting with "_") into a tuple
        attr_values = tuple(
            (attr, self._make_hashable(getattr(self, attr)))
            for attr in dir(self)
            if not callable(getattr(self, attr)) and not attr.startswith("_")
        )
        hashid = hash(attr_values)
        # Create a unique string from the tuple and return its hash
        return hashid

    def _make_hashable(self, value):
        if isinstance(value, dict):
            # Convert dictionary to a frozenset of its items (key-value pairs)
            return frozenset(
                (key, self._make_hashable(v)) for key, v in value.items()
            )
        elif isinstance(value, list):
            # Convert list to a tuple of its elements
            return tuple(self._make_hashable(v) for v in value)
        elif isinstance(value, set):
            # Convert set to a frozenset of its elements
            return frozenset(self._make_hashable(v) for v in value)
        # Add other types like list, set, etc., if needed
        return value
