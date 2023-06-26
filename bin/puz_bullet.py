import math
import time
import numpy as np
import pybullet as p
from puzzlebot_assembly.utils import *
from puzzlebot_assembly.robots import Robots
from puzzlebot_assembly.control import Controller, ControlParam
from puzzlebot_assembly.behavior_lib import BehaviorLib

def get_lr(v, w):
    L, R = 0.04, 0.01
    vl = ((2 * v) - (w * L)) / (2 * R)
    vr = ((2 * v) + (w * L)) / (2 * R)
    return vl, vr

if __name__ == "__main__":

    # initialize environment
    cid = p.connect(p.SHARED_MEMORY)
    if (cid < 0):
      p.connect(p.GUI)
    p.resetSimulation()
    p.setGravity(0, 0, -9.8)
    p.setRealTimeSimulation(1)
    p.loadURDF("urdf/plane.urdf")

    # set up visualizer 
    p.resetDebugVisualizerCamera(0.7, 1, -80, [0.5,0.5,0])

    # set up controllers and robots
    dt = 0.1
    eth = 2e-3
    start = np.array([[0.0, 0.7, 0],
                    [1.0, 1.0, 0],
                    [0.5, 0.1, 0],
                    #  [-0.3, 0.3, 0],
                    #  [1.3, 0.5, 0],
                    [0.3, 0.4, 0]]).T
    #  start = np.array([[i*0.07, 0, 0] for i in range(10)]).T
    N = start.shape[1]
    c_param = ControlParam(vmax=0.03, wmax=0.5,
                    mpc_horizon=10, cbf_horizon=10, eth=eth)
    c = Controller(N, dt, c_param)
    rsys = Robots(N, c, eth=eth)
    rsys.setup(start=start)
    robots = []

    for i in range(N):
        r = p.loadURDF(
                    #  "urdf/puzzlebot_conv.urdf", 
                    "urdf/puzzlebot.urdf",
                    start[0, i], start[1, i], 0.005, 
                    0, 0, 0, 1)
        robots.append(r)

    joint_id = {'left': -1, 'right': -1, 'front': -1, 'back': -1}
    for jid in range(p.getNumJoints(robots[0])):
        info = p.getJointInfo(robots[0], jid)
        if info[1].decode("utf-8") == "left_wheel_joint":
            joint_id['left'] = jid
        elif info[1].decode("utf-8") == "right_wheel_joint":
            joint_id['right'] = jid
        elif info[1].decode("utf-8") == "front_wheel_joint":
            joint_id['front'] = jid
        elif info[1].decode("utf-8") == "back_wheel_joint":
            joint_id['back'] = jid
    print(joint_id)
    assert(-1 not in joint_id.values())

    for i in range(N):
        r = robots[i]
        p.changeDynamics(r, joint_id['front'], lateralFriction=0)
        p.changeDynamics(r, joint_id['back'], lateralFriction=0)

    maxForce = 5
    t = time.time()
    while (1):
        #  u = np.array([[-0.01*(i-4), 0] for i in range(N)]).flatten()
        u = np.array([[0, 0.4] for i in range(N)]).flatten()
        x = np.zeros(3*N)
        #  is_done = rsys.step(rsys.x, rsys.u, time.time())
        #  u = rsys.u
        #  x = rsys.x
        for i in range(N):
            r = robots[i]
            vl, vr = get_lr(u[2*i], u[2*i+1])
            p.setJointMotorControl2(r, joint_id['left'], 
                            controlMode=p.VELOCITY_CONTROL, 
                            targetVelocity=vl, 
                            force=maxForce)
            p.setJointMotorControl2(r, joint_id['right'], 
                            controlMode=p.VELOCITY_CONTROL, 
                            targetVelocity=vr, 
                            force=maxForce)
            pose, quat = p.getBasePositionAndOrientation(r)
            yaw = p.getEulerFromQuaternion(quat)[2]
            x[3*i:(3*i+2)] = [pose[0], pose[1]]
            x[3*i+2] = yaw
        print('x:', x)
        print('u:', u)
        #  rsys.x = x
        p.stepSimulation()
        t_diff = time.time() - t
        if (t_diff < dt):
            time.sleep(dt - t_diff)

    p.disconnect()
