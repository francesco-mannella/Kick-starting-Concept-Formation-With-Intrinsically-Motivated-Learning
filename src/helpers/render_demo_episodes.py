import glob
import os, sys
from pathlib import Path

import numpy as np
import matplotlib

# Add parent directory to Python module path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from SMMainUtils import MainUtils as Main
import params

matplotlib.use("Agg")

np.set_printoptions(formatter={"float": "{:6.4f}".format})

storage_dir = "storage"
site_dir = "www"
os.makedirs(storage_dir, exist_ok=True)
os.makedirs(site_dir, exist_ok=True)

if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-g", 
        "--gpu", 
        help="Use gpu", 
        action="store_true"
    )
    args = parser.parse_args()
    gpu = bool(args.gpu)

    if not gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    if os.path.isfile("main.dump.npy"):
        main = np.load(
            "main.dump.npy", allow_pickle="True"
        )[0]
        main.demo_episodes()
