#!/usr/bin/env python

import numpy as np
from puzzlebot_assembly.robots import Robots
from puzzlebot_assembly.control import Controller, ControlParam
from puzzlebot_assembly.behavior_lib import BehaviorLib

if __name__ == "__main__":
    N = 3
    dt = 0.1
    eth = 1.5e-3
    start = np.array([[0.1, 0.1, 0],
                    [1.0, 1.0, 0],
                    [0.2, 0.5, 0]]).T
    c_param = ControlParam(mpc_horizon=10, cbf_horizon=10, eth=eth)
    c = Controller(N, dt, c_param)
    r = Robots(N, c, eth=eth)
    r.setup(start=start, tmax=15)
    r.start()
    r.generate_anim()
