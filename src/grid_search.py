import os
import re
import subprocess
from itertools import product

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


# params = {
#     "obj_fix_prob": [0.2, 0.4, 0.6, 0.8, 1.2],
#     "obj_var_prob": [1.2, 1.6],
#     "obj_x": [2],
#     "obj_y": [2],
#     "epochs": [1000],
# }
#
# base_name = "obj_params"


params = {
    "cum_match_stop_th": [3.0],
    "representation_sigma": [2],
    "base_match_sigma": [8],
    "match_sigma": [8],
    "predict_rl": [0.02],
    "obj_fix_prob": [0.8],
    "obj_var_prob": [0.7, 1.2],
    "obj_x": [2],
    "obj_y": [2],
    "epochs": [400],
}

base_name = "incr_rescaled_comp"


processes = []
MAX_PROCESSES = 4

orig_path = os.path.dirname(os.path.realpath(__file__))

for i, p in enumerate(get_combinations(params)):
    #
    # If MAX_PROCESSES reached, wait until all of them finish.
    if len(processes) == MAX_PROCESSES:
        for process in processes:
            process.wait()
        processes = []
    #
    options_str = ""
    for k, v in p.items():
        options_str += f" -o {k}={v}"
    option_key = optimize_option_key(options_str)

    base_cmd_str = (
        f"nohup python -u {orig_path}/SMMain.py "
        f"-n {base_name}_{option_key} -s 1000 -t 55000 -x -g -w "
        f"> {base_name}_{option_key}.log 2>&1"

    )
    cmd_str = base_cmd_str + options_str
    print(cmd_str)

    print(f"Running: {cmd_str}")
    processes.append(subprocess.Popen(cmd_str, shell=True))
