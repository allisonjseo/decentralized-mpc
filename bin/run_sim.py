import cProfile
import numpy as np
from puzzlebot_assembly.robots import Robots, RobotParam
from puzzlebot_assembly.control import Controller, ControlParam
from puzzlebot_assembly.behavior_lib import BehaviorLib
from puzzlebot_assembly.simulation import BulletSim
from multiprocessing.pool import Pool



#def main():

if __name__ == "__main__":
    dt = 0.1
    eth = 1.5e-3
    start = np.array([
                    [0.5, 0.0, 0],
                    [-0.5, 0.4, 0],
                    [-0.6, 0.2, 0]
                    #[0.5, 0.6, 0]
                    #[0.0, 0.0, 0]
                    ]).T
    N = start.shape[1]
    r_param = RobotParam(L=5e-2, anchor_base_L=8e-3, anchor_L=1.4e-2)
    c_param = ControlParam(vmax=0.05, wmax=1.5,
                    uvmax=2.0, uwmax=3.0,
                    mpc_horizon=3, constr_horizon=3, eth=eth)
    #pool = Pool()
    c = Controller(N, dt, c_param)
    rsys = Robots(N, c, robot_param=r_param, eth=eth, pilot_ids=[])
    sim = BulletSim(N, c_param, c, rsys, is_anchor_separate=False)
    sim.setup(start=start)
    sim.load_urdf(robot_file="urdf/puzzlebot.urdf",
                anchor_file="urdf/puz_anchor.urdf",
                env_file="urdf/plane.urdf")
    sim.start()
    sim.end()

"""def pro():
    import cProfile, pstats, io
    from pstats import SortKey

    pr = cProfile.Profile()
    pr.enable()
    main()
    pr.disable()

    s = io.stringIO()
    sortby = SortKey.CUMULATIVE
    ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
    ps.print_stats()
    print(s.getvalue())"""


#if __name__ == "__main__":
"""import cProfile
    cProfile.run('main()', "output.dat")
    
    import pstats
    from pstats import SortKey
    with open("output_time.txt", "w") as f:
        p = pstats.Stats("output.dat", stream=f)
        p.sort_stats("time").print_stats()
    with open("output_calls.txt", "w") as f:
        p = pstats.Stats("output.dat", stream=f)
        p.sort_stats("calls").print_stats()"""
    #pro()

    #cProfile.run('main()', sort='cumtime')