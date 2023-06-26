import numpy as np
import polytope as pt

import matplotlib.patches as patches
import matplotlib.pyplot as plt
from matplotlib import animation

from puzzlebot_assembly.utils import *

class Canvas:
    def __init__(self, N, robot_shape=None,
                    xlim=[-0.5, 2], ylim=[-0.5, 2]):
        self.N = N
        self.robot_shape = robot_shape
        self.xlim = xlim
        self.ylim = ylim

        if self.robot_shape is None:
            self.robot_shape = self.default_robot_shape()

    def default_robot_shape(self):
        L = 0.05
        pts = np.array([[L/2, L/2], [L/2, -L/2], 
                        [-L/2, -L/2], [-L/2, L/2]])
        return pts

    def get_current_shape(self, x):
        '''
        x: len=3 vector
        '''
        assert(x.shape[0] == 3 and len(x.shape) == 1)
        g = np.eye(3)
        g[0:2, 0:2] = get_R(x[2])
        g[0:2, 2] = x[0:2]
        pts = np.vstack([self.robot_shape.T, 
                        np.ones(self.robot_shape.shape[0])])
        curr_pts = g.dot(pts)
        return curr_pts[0:2, :].T

    def animation(self, logger):
        print("Generating animation ...")
        fig, ax = plt.subplots()
        plt.axis("equal")
        ax.set_xlim(self.xlim)
        ax.set_ylim(self.ylim)
        
        rps = []
        for i in range(self.N):
            rp = patches.Polygon(np.zeros((1, 2)), closed=True, 
                                edgecolor='b', alpha=0.4)
            ax.add_patch(rp)
            rps.append(rp)
        def update(ti):
            x = logger.xs[ti]
            for ni in range(self.N):
                r_curr = patches.Polygon(
                                    self.get_current_shape(x[ni*3:(ni*3+3)]),
                                    closed=True,
                                    edgecolor='b', alpha=0.4)
                rps[ni].set_xy(r_curr.get_xy())

        time_lim = len(logger.xs)
        anim = animation.FuncAnimation(fig, update, frames=time_lim)
        anim.save("media/anim.mp4", dpi=300, 
                writer=animation.writers["ffmpeg"](fps=10))
