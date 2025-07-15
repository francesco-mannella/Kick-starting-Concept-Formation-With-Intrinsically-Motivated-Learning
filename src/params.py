import sys
import json

import numpy as np


class ParameterManager:

    def __init__(self):
        self._set_param_types()

    def _set_param_types(self):
        self.param_types = {
            k: type(v)
            for k, v in self.__dict__.items()
            if not k.startswith("__") and not callable(v)
        }

    def check_json_format(self, json_string):
        json_string = (
            json_string.replace("'", '"')
            .replace(" ", "")
            .replace("True", "true")
            .replace("False", "false")
        )
        return json_string

    def _string_to_json(self, param_string, mode="user"):
        """Converts a semicolon-separated string to a JSON dictionary.

        Args:
            param_string (str): String representing the params.
            mode (str): Format of the string; "user" for
              semicolon-separated key-value pairs, "json" for JSON string.

        Returns:
            dict: Dictionary representing the JSON object.

        """
        if not param_string:
            return {}

        if mode == "user":
            # Split the string into key-value pairs
            param_string = param_string.replace(" ", "")
            params = dict(
                s.split("=", 1) for s in param_string.split(";") if "=" in s
            )
            # Format the key-value pairs into a JSON string
            params = ",".join(f'"{k.strip()}":{v}' for k, v in params.items())
            params = "{" + params + "}"
        else:
            params = param_string

        params = self.check_json_format(params)

        try:
            param_dict = json.loads(params)
        except ValueError as e:
            print(f"Error decoding JSON: {e}")
            sys.exit(1)

        return param_dict

    def _json_to_params(self, param_dict):
        """Set attributes from a dictionary.

        Args:
            param_dict (dict): Dictionary of parameters.
        """
        # Iterate over the key-value pairs in the input dictionary.
        for key, value in param_dict.items():
            # For each pair, set an attribute of the instance.
            setattr(self, key, value)

    def _params_to_dict(self):
        params = {
            key: value
            for key, value in self.__dict__.items()
            if key != "param_types" and not callable(value)
        }
        return params

    def __repr__(self):
        params = self._params_to_dict()
        return json.dumps(params)

    def update(self, param_string, mode="user"):
        """Update parameters based on the input string.

        Args:
            param_string (str): Input string containing parameters.
            mode (str, optional): "user" to convert the input
                string to JSON. "json" if the input string is a
                json format. Defaults to "user".
        """
        if mode == "user":
            # Convert user-provided string to JSON format
            param_dict = self._string_to_json(param_string)
        elif mode == "json":
            param_dict = self._string_to_json(param_string, mode="json")

        # Update internal parameters using the JSON dictionary
        self._json_to_params(param_dict)

    def save(self, filepath, mode="user"):
        """Saves parameters to a file.

        Args:
            filepath (str): The path to the file.
            mode (str, optional): Specifies the saving mode.
              "user": Saves parameters in a human-readable format.
              Each parameter is written as 'key = value' on a new line.
              "json": Saves parameters in JSON format.
              Defaults to "user".
        """
        with open(filepath, "w") as file:
            if mode == "user":
                for key, value in self._params_to_dict().items():
                    file.write(f"{key} = {value}\n")
            elif mode == "json":
                params = self._params_to_dict()
                json.dump(params, file, indent=4)

    def load(self, filepath, mode="user"):
        """Loads parameters from a file.

        Args:
            filepath (str): The path to the file.
            mode (str, optional): Specifies the loading mode.
              "user": Loads parameters from a human-readable format.
              Expects each parameter to be in the format 'key = value'.
              "json": Loads parameters from JSON format.
              Defaults to "user".
        """
        with open(filepath, "r") as file:
            if mode == "user":
                param_list = "".join([line.strip() + ";" for line in file])
                self.update(param_list)
            elif mode == "json":
                params = json.load(file)
                self.__dict__.update(params)

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
        obj_x=4,
        obj_y=2,
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
        self.obj_x = obj_x
        self.obj_y = obj_y

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
