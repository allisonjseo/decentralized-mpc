import numpy as np
from puzzlebot_assembly.robots import Robots, RobotParam
from puzzlebot_assembly.control import Controller, ControlParam
from puzzlebot_assembly.behavior_lib import BehaviorLib
from puzzlebot_assembly.simulation import BulletSim
from multiprocessing.pool import Pool
from multiprocessing import Process, Queue
from time import time

if __name__ == "__main__":
    dt = 0.1
    eth = 2e-3
    start = np.array([
                    [1.0, 0.5, 0],
                    [-0.3, 0.7, 0]
                    #[-0.6, 0.7, 0],
                    #[0.0, 0.6, 0],
                    #[0.5, 0.7, 0]
                    ]).T
    N = start.shape[1]

    for i in range(2):
        r_param = RobotParam(L=5e-2, anchor_base_L=8e-3, anchor_L=1.4e-2)
        c_param = ControlParam(vmax=0.05, wmax=1.5,
                        uvmax=1.5, uwmax=3.0,
                        mpc_horizon=3, constr_horizon=3, eth=eth)
        
        c = Controller(N, dt, c_param)
        pool = Pool()
        rsys = Robots(N, c, pool, eth, robot_param=r_param, pilot_ids=[])
        sim = BulletSim(N, c_param, c, rsys, is_anchor_separate=False)
        sim.setup(start=start)
        sim.load_urdf(robot_file="urdf/puzzlebot.urdf",
                    anchor_file="urdf/puz_anchor.urdf",
                    env_file="urdf/plane.urdf")
        sim.start()
        pool.close()
        sim.end()