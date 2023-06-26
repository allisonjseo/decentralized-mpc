# PuzzleBot

## Installation

1. Create conda environment with 
```
conda env create -f puzzle_env.yml
```
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
3. The package is originally in the ROS catkin source workspace due to my hardware interface. But the simulation is independent of ROS. You might need to add the directory to your python path.

## Generate URDF file
There are multiple PuzzleBot URDFs that can be generated. Currently I'm using the simplified one. Run the following in the package root directory.
```
python scripts/generate_urdf.py 2
```
This should generate `puzzlebot.urdf` in the `urdf/` directory.

## Run Simulation
Run the following for the simulation.
```
python bin/run_sim.py
```
Let me know if you encounter any errors.
