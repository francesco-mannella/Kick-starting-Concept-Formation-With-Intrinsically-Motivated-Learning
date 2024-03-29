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
