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
    parser.add_argument(
        "-w",
        "--wandb",
        help="Store simulations results to Weights and Biases",
        action="store_true",
    )
    parser.add_argument(
        "-n",
        "--name",
        help="Simulation name (to store results in named folders)",
        action="store",
        default=None
    )
    args = parser.parse_args()
    gpu = bool(args.gpu)
    use_wandb = bool(args.wandb)

    if not gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    if use_wandb:
        import wandb
        run = wandb.init(
            project="kickstarting_concept",
            entity="hill_uw",
            name=args.name
        )

    if os.path.isfile("main.dump.npy"):
        main = np.load(
            "main.dump.npy", allow_pickle="True"
        )[0]
        main.demo_episodes(n_episodes=20, unique_prototypes=True)

        if use_wandb:
            log_data = {}
            for f in glob.glob("www/demo_????.gif"):
                fname = Path(f).stem
                log_data[fname] = wandb.Image(f)
            wandb.log(log_data)
