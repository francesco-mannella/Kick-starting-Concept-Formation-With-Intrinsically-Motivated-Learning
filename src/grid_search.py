import os
from itertools import product

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

params = {
    "match_th": [0.3, 0.4, 0.5],
    "match_incr_th": [0.05, 0.1],
    "predict_lr": [0.1, 0.05],
    "base_lr": [0.01, 0.02, 0.05],
    "stm_lr": [0.5, 0.2, 0.1],
    "cum_match_stop_th": [20, 30],
    "modalities_weights": [[1., 1., 1., 1.], [1., 2., 1., 1.], [1., 2., 2., 1.]]
}

params = {
    "match_th": [0.3, 0.4],
    "stm_lr": [0.5, 0.1],
}

base_name = "grid_search"


for i, p in enumerate(get_combinations(params)):
    base_cmd_str = f"nohup python SMMain.py -n {base_name}_{i} -s 1000 -t 55000 -x -g"
    options_str = ""
    for k, v in p.items():
        options_str += f" -o {k}={v}"
    cmd_str = base_cmd_str + options_str
    print(f"Running: {cmd_str}")
    os.system(cmd_str)

