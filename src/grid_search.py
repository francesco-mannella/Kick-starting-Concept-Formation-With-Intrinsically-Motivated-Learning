import os
import re
import subprocess
from itertools import product

import numpy as np
import slugify


def get_combinations(data):
    """
    Generates all possible combinations of list elements from a dictionary.

    Args:
       data: A dictionary where values are lists.

    Yields:
       A dictionary representing a single combination of elements.
    """
    combinations = product(*[value for value in data.values()])
    for combination in combinations:
        yield dict(zip(data.keys(), combination))


def optimize_option_key(options_str):
    """
    Generates an optimized option key from a string of options.

    Args:
        - options_str: A string containing options

    Returns:
        A slugified string representing the option key.
    """
    cleaned_str = options_str.replace("-o", "-").replace(" ", "")
    cleaned_str = re.sub(r"epochs=\d+", "", cleaned_str)
    return slugify.slugify(cleaned_str)


params = {
    "epochs": [1000],
    "decay": [5.0, 5.5],
    "local_decay": [2.0, 2.5, 3.0],
    "obj_stretch_conditions": [[1, 2]],
    "reach_grip_prop": [0.3],
    "policy_base_arm": [0.0314],
    "max_policy_noise": [100.0],
    "internal_sigma": [8],
    "obj_x": [2.0],
    "obj_y": [0.5],
}
seeds = np.arange(3)

base_name = "battery"


processes = []
MAX_PROCESSES = 4

orig_path = os.path.dirname(os.path.realpath(__file__))

for i, p in enumerate(get_combinations(params)):
    for seed in seeds:
        # If MAX_PROCESSES reached, wait until all of them finish.
        if len(processes) == MAX_PROCESSES:
            for process in processes:
                process.wait()
            processes = []
        #
        options_str = ""
        for k, v in p.items():
            options_str += f" -o '{k}={v}'"
        option_key = optimize_option_key(options_str)

        base_cmd_str = (
            f"nohup python {orig_path}/SMMain.py "
            f"-n {base_name}_{option_key}_{seed:06d} "
            f"-s {seed} -t 55000 -x -g -w "
            "--wdb_project grasp-simulation "
            "--wdb_entity francesco-mannella"
        )
        cmd_str = base_cmd_str + options_str

        print(f"Running: {cmd_str}")
        processes.append(subprocess.Popen(cmd_str, shell=True))
