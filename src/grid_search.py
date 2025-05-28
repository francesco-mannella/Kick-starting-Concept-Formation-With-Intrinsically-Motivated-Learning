"""
This script automates the execution of SMMain.py with a range of parameter
combinations.  It reads a JSON file specifying the parameter grid, generates
all possible combinations, and then launches SMMain.py as a separate process
for each combination.  It limits the number of concurrent processes to avoid
overloading the system.  The script takes two command-line arguments:

  -p or --params_json: Path to the JSON file defining the parameter grid. The
    JSON file should contain a dictionary where keys are parameter names and
    values are lists of possible values for each parameter.

  -n or --base_name:  Base name used for the run identifier and log files.
    Each run's output will be logged to a file named
    "<base_name>_<option_key>.log", where <option_key> is a slugified string
    representing the parameter combination used for that run.


Example JSON parameter file (params.json):

{
  "cum_match_stop_th": [10.0],
  "representation_sigma": [2],
  "base_match_sigma": [8],
  "match_sigma": [8],
  "predict_rl": [0.02],
  "obj_fix_prob": [0.8],
  "obj_var_prob": [0.7, 1.2, 1.6],
  "obj_x": [2],
  "obj_y": [2],
  "epochs": [800]
}

Example Usage:

./grid_search.py -p params.json -n my_experiment

This command will read the parameter grid from params.json, generate all
possible combinations of parameters, and launch SMMain.py for each combination.
The output of each run will be logged to a separate file named
my_experiment_<option_key>.log.  The script limits the number of concurrent
processes to 4.

"""

import argparse
import json
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

    for k, v in data.items():
        if (
            isinstance(v, str)
            or isinstance(v, int)
            or isinstance(v, float)
            or isinstance(v, bool)
        ):
            data[k] = [v]

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


def parse_params_string(params_string):
    """
    Parses a JSON string representing parameters into a Python dictionary.

    Args:
      params_string: A string containing JSON data representing parameters.

    Returns:
      A dictionary containing the parsed parameters, or None if parsing fails.
    """
    try:
        params = json.loads(params_string)
        return params
    except json.JSONDecodeError:
        print(
            f"Error: Invalid JSON format in params string: {params_string}"
        )  # Improved error message
        return None


def get_params_from_file(paramfile):
    """
    Reads a file containing JSON parameters and returns them as a dictionary.

    Args:
      paramfile: The path to the file containing the JSON parameters.

    Returns:
      A dictionary containing the parsed parameters, or None if an error
      occurs.
    """
    try:
        with open(paramfile, "r") as f:
            params_string = f.read()
            params = parse_params_string(params_string)
            return params
    except FileNotFoundError:
        print(f"Error: File not found: {paramfile}")
        return None
    except (
        Exception
    ) as e:  # Catch any other potential errors during file reading
        print(f"Error reading file {paramfile}: {e}")
        return None


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Run SMMain.py with various parameter combinations."
    )
    parser.add_argument(
        "-p",
        "--params_json",
        type=str,
        required=True,
        help="Path of the json dictionary of parameters",
    )
    parser.add_argument(
        "-n",
        "--base_name",
        type=str,
        required=True,
        help="Base name for the run and log files.",
    )

    parser.add_argument(
        "-s",
        "--seed",
        type=int,
        default=1000,
        help="Seed of the simulations",
    )
    parser.add_argument(
        "-w",
        "--wandb",
        type=int,
        default=False,
        help="Store data on Weights & Biases",
    )

    args = parser.parse_args()

    params_json = args.params_json
    base_name = args.base_name
    seed = args.seed
    use_wandb = "-w" if args.wandb else ""

    params = get_params_from_file(params_json)

    processes = []  # type: list[subprocess.Popen]
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
            f"-n {base_name}_{option_key} -s {seed} "
            f"-t 55000 -x -g {use_wandb} "
            f"> {base_name}_{option_key}.log 2>&1"
        )
        cmd_str = base_cmd_str + options_str
        print(cmd_str)

        print(f"Running: {cmd_str}")
        processes.append(subprocess.Popen(cmd_str, shell=True))

    # Wait for any remaining processes to finish
    for process in processes:
        process.wait()
