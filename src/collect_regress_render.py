import glob
import os, sys
from pathlib import Path

import params
import numpy as np
import matplotlib
import time
import tensorflow as tf
from SMMainUtils import MainUtils
from SMMain import Main
from regress_sensory_states import regress

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
        main_core = np.load(
            "main.dump.npy", allow_pickle="True"
        )[0]

        main = MainUtils()
        main.__dict__.update(main_core.__dict__)

        #main.collect_sensory_states()
        #regress()
        main.demo_episodes()
