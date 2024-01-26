# PuzzleBot

The code for this project is adapted from here: https://github.com/ZoomLabCMU/puzzlebot_anchor. Feel free to check it out for more details.

## Installation

1. Create conda environment with 
```
conda env create -f puzzle_env.yml
```
- note that this simulation was previously using python 3.7 which is no longer available through default conda. See in puzzle_env.yml, because this may cause backward compatability issues

This will create a environment named `puzzle`. You will need to activate it with
```
conda activate puzzle
```

2. Install three other dependencies with `pip` in the `puzzle` environment.
```
pip install casadi
pip install polytope
pip install pybullet
```
Follow one of the next two instructions to properly build the `puzzlebot_assembly` module which is included in this repository under src.

3. ONLY APPLICABLE IF RUNNING ON ROS (robot operating system): Rename `setup.py` to `backup.py`, and rename `setup_ros.py` to `setup.py`, this way catkin will find the correct setup.py file. Then use catkin to install the package. For more information, see catkin documentation.

4. IF NOT USING ROS: run `pip install .` to run the `setup.py` file.  Ensure that `setuptools` is included in your list of conda packages in the current environment using `conda list`. 

## Run Simulation
Run the following for the simulation.
```
python bin/run_sim2.py. run_sim.py is outdated and will fail to run
```
Let me know if you encounter any errors.
