import subprocess
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
    "cum_match_stop_th": [10, 20, 30, 40],
}

base_name = "gs_cum_match_stop_th"

processes = []
MAX_PROCESSES = 4

for i, p in enumerate(get_combinations(params)):

    # If MAX_PROCESSES reached, wait until all of them finish.
    if len(processes) == MAX_PROCESSES:
        for process in processes:
            process.wait()
        processes = []

    base_cmd_str = f"nohup python SMMain.py -n {base_name}_{i} -s 1000 -t 55000 -x -g -w"
    options_str = ""
    for k, v in p.items():
        options_str += f" -o {k}={v}"
    cmd_str = base_cmd_str + options_str
    print(f"Running: {cmd_str}")
    processes.append(subprocess.Popen(cmd_str, shell=True))
