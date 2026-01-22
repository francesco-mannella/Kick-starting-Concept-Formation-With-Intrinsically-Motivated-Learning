import argparse
import collections
import os
import re
import subprocess
import sys
from itertools import product

import numpy as np
import slugify


def parse_arguments():
    parser = argparse.ArgumentParser(description="Process some integers.")
    parser.add_argument("--wandb", action="store_true", help="Enable WANDB")
    parser.add_argument("--n_seeds", type=int, default=5, help="Number of seeds")
    parser.add_argument("--max_processes", type=int, default=2, help="Max processes")
    parser.add_argument("--base_name", type=str, default="testnoise", help="Base name")
    parser.add_argument("--seeds", nargs="+", type=int, default=[93581], help="Seeds")
    return parser.parse_args()


args = parse_arguments()

params = dict(
    decay=[5.5, 6],
    base_match_sigma=2,
    match_sigma=2,
    base_internal_sigma=0.1,
    cum_match_stop_th=1.0,
)


def get_combinations(data):
    for k, v in data.items():
        if not isinstance(v, collections.abc.Iterable):
            data[k] = [v]
    combinations = product(*[value for value in data.values()])
    for combination in combinations:
        yield dict(zip(data.keys(), combination))


def optimize_option_key(options_str):
    cleaned_str = options_str.replace("-o", "-").replace(" ", "")
    cleaned_str = re.sub(r"epochs=\d+", "", cleaned_str)
    return slugify.slugify(cleaned_str)


seeds = args.seeds or np.random.randint(0, 1e5, args.n_seeds)


processes = []
orig_path = os.path.dirname(os.path.realpath(__file__))

for i, p in enumerate(get_combinations(params)):
    for seed in seeds:
        if len(processes) == args.max_processes:
            for process in processes:
                process.wait()
            processes = []
        options = []
        for k, v in p.items():
            options.append("-o")
            options.append(f"{k}={v}")

        option_key = optimize_option_key("".join(options))

        run_id = f"{args.base_name}_{option_key}_{seed:06d}"

        command = [
            sys.executable,
            f"{orig_path}/SMMain.py",
            "-n",
            f"{run_id}",
            "-s",
            f"{seed}",
            "-t",
            "55000",
            "-x",
            "-g",
            "--wdb_project",
            "grasp-simulation",
            "--wdb_entity",
            "francesco-mannella",
        ]

        if args.wandb:
            command.append("-w")
        command.extend(options)

        print(f"Running: {' '.join(command)}")

        with open(f"{run_id}.log", "w") as log:
            processes.append(
                subprocess.Popen(
                    command,
                    stdout=log,
                    stderr=log,
                )
            )

exit_codes = [p.wait() for p in processes]
