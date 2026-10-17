# Kick-starting-Concept-Formation-With-Intrinsically-Motivated-Learning

## Description
This code implements a simulated agent in 2D environment with [box2d](https://box2d.org/) physics. The agent is a 3-DoF arm with a 2-DoF gripper. The environment is also filled with an object. The agent is controlled by a computational model which learns a map of its sensorimotor contingencies via the interaction with the environment. The model is described in the paper: Mannella F, Tummolini L.
2022 Kick-starting concept formation with
intrinsically motivated learning: the grounding
by competence acquisition hypothesis. Phil.
Trans. R. Soc. B 20210370.
https://doi.org/10.1098/rstb.2021.0370.

![alt text](https://github.com/francesco-mannella/Kick-starting-Concept-Formation-With-Intrinsically-Motivated-Learning/blob/main/docs/demo.gif?raw=true)

## Prerequisites

The code relies on [tensorflow](https://www.tensorflow.org), although it can be run also without the gpu enhancement.
Other required libraries are:
  * gym
  * box2d

## Install

You must first install the box2dsim packege to run the simultor:

        pip install -e tools/box2dsim

You also must decompress the data archive in the source folder

        cd src
        tar xzvf data.tar.gz
  
## Run simulations

To start a simulation copy the **src** folder and run [SMMain.py](https://github.com/francesco-mannella/Kick-starting-Concept-Formation-With-Intrinsically-Motivated-Learning/blob/main/src/SMMain.py) form the copied folder. 

        python SMMain.py -h
        usage: SMMain.py [-h] [-t TIME] [-g] [-s SEED] [-x]

        optional arguments:
          -h, --help            show this help message and exit
          -t TIME, --time TIME  The maximum time for the simulation (seconds)
          -g, --gpu             Use gpu
          -s SEED, --seed SEED  Simulation seed
          -x, --plots           Plot graphs

All parameters for a simulation (included the number of epochs to be run) are described in [SMMain.py](https://github.com/francesco-mannella/Kick-starting-Concept-Formation-With-Intrinsically-Motivated-Learning/blob/main/src/params.py). Modify the **params.py** file in the copied **src** folder if you need to tweek the parameters of a simulation. 

Example:
  
        python SMMain.py -s 1000 -t 14400 -g -x  

runs a simulation lasting 4 hours with graphics swithed on and using the gpu. If the simulation does not end in the required time, all data are recorded in the folder and the simulation can be restarted from the current time just by running again the script. 
            
## Local run tracking (alternative to wandb)

Add `--local` to `SMMain.py` (or `-l` to `grid_search.py`) to store metrics and images in a local SQLite database instead of wandb. `--db PATH` selects the file (default `./localwandb.db`; images go to `<db name>_media/`).

        python SMMain.py -n myrun --local --db runs.db
        python localwandb_server.py --db runs.db --port 8080

Open http://127.0.0.1:8080 to compare runs (metric charts with `_par` parasite curves overlaid, map images by step, parameter differences). Runs from several processes can write to the same database. The Figures tab lists the images and CSV tables in `--figures DIR` (default `<db folder>/figures`), for example the output of `plot_fig567.py`.

## Parasite experiment launchers

`src/launch_run.py` starts one run in its own folder (`--mode parasite|former|agent`, `-s SEED`; `former` loads the maps of the same seed's parasite run). `src/launch_experiment.py` runs all seeds (default 0-9: a parasite run, then a former-parasite run per seed) with a job pool. Both log to `<root>/runs.db` (`--local`), so `localwandb_server.py --db <root>/runs.db` shows them.

        python launch_experiment.py --root ../test/figs567 -p 8 --epochs 1000
        python launch_run.py --mode parasite -s 3 --root ../test/figs567 --dry-run

Finished runs are skipped on relaunch; an incomplete run folder needs `--force` (it is deleted and rerun).

## License

Shield: [![CC BY 4.0][cc-by-shield]][cc-by]

This work is licensed under a
[Creative Commons Attribution 4.0 International License][cc-by].

[![CC BY 4.0][cc-by-image]][cc-by]

[cc-by]: http://creativecommons.org/licenses/by/4.0/
[cc-by-image]: https://i.creativecommons.org/l/by/4.0/88x31.png
[cc-by-shield]: https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg

## Code dependency tree

![Code dependency tree](https://github.com/francesco-mannella/Kick-starting-Concept-Formation-With-Intrinsically-Motivated-Learning/blob/main/docs/dependency_tree.svg?raw=true)
