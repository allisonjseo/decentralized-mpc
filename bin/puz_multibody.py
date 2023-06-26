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

def get_anchor_vel(s, eth=1e-2):
    if np.abs(s / np.pi) < eth: return 0
    vmax = 5.0
    vel = - 3 * s
    vel = np.clip(vel, -vmax, vmax)
    return vel

def get_anchor_force(s, eth=1e-2):
    if np.abs(s) < eth: return 0
    kp, kn = 1e-2, 5
    max_p, max_n = 0.3, 1.0
    f = 0
    if s > 0:
        f = kp * s
        f = np.min([max_p, f])
    else:
        f = - kn * s
        f = np.min([max_n, f])
    return f

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
    p.resetDebugVisualizerCamera(0.05, 0, -80, [-0.02,0.0,0.1])

    # set up controllers and robots
    dt = 0.1
    eth = 2e-3
    start = np.array([
                    [0, 0, 0],
                    #  [0.07, 0, 0]
                    [0.052, 0, 0]
                    #  [0.0, 0.7, 0],
                    #  [1.0, 1.0, 0],
                    #  [0.5, 0.1, 0],
                    #  [-0.3, 0.3, 0],
                    #  [1.3, 0.5, 0],
                    #  [0.3, 0.4, 0]
                    ]).T
    #  start = np.array([[i*0.07, 0, 0] for i in range(10)]).T
    N = start.shape[1]
    c_param = ControlParam(vmax=0.03, wmax=0.5,
                    mpc_horizon=10, cbf_horizon=10, eth=eth)
    c = Controller(N, dt, c_param)
    rsys = Robots(N, c, eth=eth)
    rsys.setup(start=start)
    robots = []
    anchors = []

    for i in range(N):
        r = p.loadURDF(
                "urdf/puzzlebot.urdf",
                [start[0, i], start[1, i], 0.005], 
                [0, 0, 0, 1]
                )
        ak = p.loadURDF(
                "urdf/puz_anchor.urdf",
                [start[0, i]+0.024, start[1, i], 0.032],
                [0, 0, 0, 1]
                )
        robots.append(r)
        anchors.append(ak)

    joint_id = {'left': -1, 'right': -1, 
                'front': -1, 'back': -1,
                'left_anchor': -1, 'right_anchor': -1,
                }
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

    for jid in range(p.getNumJoints(anchors[0])):
        info = p.getJointInfo(anchors[0], jid)
        if info[1].decode("utf-8") == "front_anchor_left_joint":
            joint_id['left_anchor'] = jid
        elif info[1].decode("utf-8") == "front_anchor_right_joint":
            joint_id['right_anchor'] = jid
    print(joint_id)
    assert(-1 not in joint_id.values())

    # initialization
    # it seems in pybullet, link id = joint id
    #  print(p.getJointInfo(robots[i], joint_id['virtual_anchor']))
    for i in range(N):
        r = robots[i]
        p.changeDynamics(r, joint_id['front'], lateralFriction=0)
        p.changeDynamics(r, joint_id['back'], lateralFriction=0)

        # disable default motors
        #  p.setJointMotorControl2(r, joint_id['left_anchor'],
                    #  controlMode=p.VELOCITY_CONTROL,
                    #  force=0)
        #  p.setJointMotorControl2(r, joint_id['right_anchor'],
                    #  controlMode=p.VELOCITY_CONTROL,
                    #  force=0)
    #  print(p.getDynamicsInfo(robots[i], joint_id['left']))

    maxForce = 0.8
    t = time.time()
    while (1):
        u = np.array([[+0.01*(i-N/2), 0] for i in range(N)]).flatten()
        #  u = np.array([[0, 0.0] for i in range(N)]).flatten()
        x = np.zeros(3*N)
        #  is_done = rsys.step(rsys.x, rsys.u, time.time())
        #  u = rsys.u
        #  x = rsys.x
        for i in range(N):
            r = robots[i]
            ak = anchors[i]
            #  print(p.getContactPoints(bodyA=r, linkIndexA=joint_id['front']))

            # wheel control
            vl, vr = get_lr(u[2*i], u[2*i+1])
            print("vl:", vl, ", vr:", vr)
            p.setJointMotorControl2(r, joint_id['left'], 
                            controlMode=p.VELOCITY_CONTROL, 
                            targetVelocity=vl, 
                            force=maxForce)
            p.setJointMotorControl2(r, joint_id['right'], 
                            controlMode=p.VELOCITY_CONTROL, 
                            targetVelocity=vr, 
                            force=maxForce)

            # anchor control
            left_state = p.getJointState(ak, 
                                    joint_id['left_anchor'])[0]
            right_state = p.getJointState(ak, 
                                    joint_id['right_anchor'])[0]
            print('left:', left_state, ', right:', right_state)
            lv = get_anchor_vel(left_state)
            rv = get_anchor_vel(right_state)
            lf = get_anchor_force(left_state)
            rf = get_anchor_force(-right_state)
            print('lf:', lf, ", rf:", rf)
            p.setJointMotorControl2(ak, joint_id['left_anchor'],
                        controlMode=p.VELOCITY_CONTROL, 
                        targetVelocity=lv, 
                        force=lf)
            p.setJointMotorControl2(ak, joint_id['right_anchor'],
                        controlMode=p.VELOCITY_CONTROL, 
                        targetVelocity=rv, 
                        force=rf)

            pose, quat = p.getBasePositionAndOrientation(r)
            yaw = p.getEulerFromQuaternion(quat)[2]
            x[3*i:(3*i+2)] = [pose[0], pose[1]]
            x[3*i+2] = yaw
        print('x:', x)
        print('u:', u)
        #  rsys.x = x
        p.stepSimulation()
        t_diff = time.time() - t
        #  if (t_diff < dt):
            #  time.sleep(dt - t_diff)

    p.disconnect()
