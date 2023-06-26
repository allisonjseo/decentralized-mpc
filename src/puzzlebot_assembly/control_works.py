import casadi as ca
import numpy as np
import math
from time import time
import multiprocessing 
from multiprocessing.pool import Pool
from puzzlebot_assembly.optimize import Optimize

from puzzlebot_assembly.utils import *

class ControlParam:
    def __init__(self, vmax=0.5,
                    uvmax=0.1,
                    wmax=1.0,
                    uwmax=0.5,
                    gamma=0.1,
                    mpc_horizon=10,
                    constr_horizon=10,
                    eth=1e-3):
        self.vmax = vmax
        self.uvmax = uvmax
        self.wmax = wmax
        self.uwmax = uwmax
        self.gamma = gamma
        self.hmpc = mpc_horizon
        self.hcst = constr_horizon
        self.eth = eth
        self.cost_Q = {
            "cp_xy": 1e5, "cp_t":1e2,      # final cost of connection pair
            "prev_xy": 1e-2, "prev_t": 1e-2,# final cost of connected cp
            #  "s_cp_xy": 1e2, "s_cp_t": 1,  # stage cost of connection pair
            "s_cp_xy": 1e0, "s_cp_t": 1e-2,  # stage cost of connection pair
            "s_prev_xy": 1e-2, "s_prev_t": 1e-3,    # stage cost of conncted cp
            "stay_xyt": 1e-5, "stay_vw": 1e-5,  # initialize cost with staying at the same position
            "zero_xyt": 1e3, # zero out the masked ids
            "smooth_v": 0, "smooth_w":0,
            "Q_u": 1e-1 
            }

class CasadiInterface:
    def __init__(self, N, dt, state_len, M=0.1):
        self.N = N
        self.dt = dt
        self.state_len = state_len
        self.M = M

    def get_local_pt(self):
        xi = ca.SX.sym('xi', 3)
        cp = ca.SX.sym('cp', 2)
        theta = xi[2]
        cp_x = ca.cos(theta)*cp[0] - ca.sin(theta)*cp[1] + xi[0]
        cp_y = ca.sin(theta)*cp[0] + ca.cos(theta)*cp[1] + xi[1]

        return ca.Function("get_local_pt", [xi, cp], [cp_x, cp_y])

    def fk_opt(self, N, dt):
        x_sym = ca.SX.sym('x', 5)
        u_sym = ca.SX.sym('u', 2)

        theta = x_sym[2]
        vs = x_sym[3]
        x_dot = ca.SX.zeros(5)
        x_dot[0] = vs * ca.cos(theta)
        x_dot[1] = vs * ca.sin(theta)
        x_dot[2] = x_sym[4]
        x_dot[3] = u_sym[0]
        x_dot[4] = u_sym[1]

        return ca.Function("fk_opt", [x_sym, u_sym], [x_sym + (x_dot * dt)])


class Controller:
    def __init__(self, N, dt, control_param):
        self.N = N
        self.dt = dt
        self.param = control_param
        self.state_len = 5
        self.ca_int = CasadiInterface(N, dt, self.state_len, M=0.1)
        self.fk_opt = self.ca_int.fk_opt(N, dt)
        self.get_local_pt = self.ca_int.get_local_pt()
        #  self.fk_opt = fk_rk4_opt(N, dt)
        #  self.fk_opt = fk_exact_opt(N, dt)
        self.ipopt_param = {"verbose": False, 
                            "ipopt.print_level": 0,
                            "print_time": 0,
                            'ipopt.sb': 'yes'
                            }
        self.opt = None
        self.x = None # 5 states [x, y, theta, v, w, ]
        self.u = None # 2 controls [uv, uw]
        self.time_lists = {}
        #self.pool = p
        for n in range(N):
            self.time_lists[n] = []
        self.cost_list = []
        self.lam_g0 = None
        self.first = True

        # for debug
        self.prev_x = None
    
    def fit_prev_x2opt(self, prev_x):
        x_curr = np.zeros([self.N, self.state_len])   #N rows, 5 cols
        x_curr[:, 0:3] = prev_x.reshape([self.N, 3])
        #x_curr = x_curr.flatten()
        return x_curr

    """def fit_prev_vw2opt(self, prev_x):
        x_curr = np.zeros"""


    def fk(self, x, u):
        '''
        Forward Kinematics
        x: 3N vector, u: 2N vector of velocity
        '''
        x_dot = dd_fx(x[2::3]).dot(u)
        return x + (x_dot * self.dt)

    def fk_rk4(self, x, u):
        '''
        Forward Kinematics
        x: 3N vector, u: 2N vector
        '''
        dt = self.dt
        k1 = dd_fx(x[2::3]).dot(u)
        k2 = dd_fx(x[2::3] + k1[2::3]*dt/2).dot(u)
        k3 = dd_fx(x[2::3] + k2[2::3]*dt/2).dot(u)
        k4 = dd_fx(x[2::3] + k3[2::3]*dt).dot(u)
        x_dot = (k1 + 2*k2 + 2*k3 + k4)/6
        return x + (x_dot * self.dt)

    def fk_exact(self, x, u):
        if np.any(u[1::2] == 0): return self.fk_rk4(x, u)
        dt = self.dt
        xx = u[0::2]/u[1::2] * (np.sin(x[2::3] + u[1::2]*dt)
                                - np.sin(x[2::3]))
        yy = - u[0::2]/u[1::2] * (np.cos(x[2::3] + u[1::2]*dt)
                                - np.cos(x[2::3]))
        tt = u[1::2] * dt
        x_dot = np.vstack([xx, yy, tt]).T.flatten()
        return (x + x_dot)

    def init_opt(self, prev_x, prev_u, prev_cp=[]):
        #multiprocessing.set_start_method('fork')
        N = self.N
        param = self.param
        opt = ca.Opti()
        sl = self.state_len
        #x = opt.variable(sl*N, param.hmpc + 1)
        #u = opt.variable(2*N, param.hmpc)
        x = []
        u = []
        for n0 in range(N):
            x.append(opt.variable(sl, param.hmpc+1))  #sl by horizon by N
            u.append(opt.variable(2, param.hmpc))     #u,w by horizon by N

        # for debug
        self.prev_x = prev_x

        # initial state constraints
        for n in range(N):
            opt.subject_to(x[n][0, 0] == prev_x[0+3*n])
            opt.subject_to(x[n][1, 0] == prev_x[1+3*n])
            opt.subject_to(x[n][2, 0] == prev_x[2+3*n])
            opt.subject_to(x[n][3, 0] == prev_u[0+2*n])
            opt.subject_to(x[n][4, 0] == prev_u[1+2*n])


        # v, w constraints - setting all the vs and ws to be between -param.max and +param.max
        for n2 in range(N):
            opt.subject_to(opt.bounded(-param.vmax, 
                                ca.vec(x[n2][3, :]), param.vmax))
            opt.subject_to(opt.bounded(-param.wmax, 
                                ca.vec(x[n2][4, :]), param.wmax))

        # uv, uw constraints
        for n3 in range(N):
            opt.subject_to(opt.bounded(-param.uvmax, 
                                ca.vec(u[n3][0, :]), param.uvmax))
            opt.subject_to(opt.bounded(-param.uwmax, 
                                ca.vec(u[n3][1, :]), param.uwmax))

        # warm start for x and u
        x_curr = self.fit_prev_x2opt(prev_x)
        for n4 in range(N):
            for ti in range(param.hmpc + 1):
                opt.set_initial(x[n4][:, ti], x_curr[n4, :])
            opt.set_initial(u[n4], 0)
        """if not self.first:
            opt.set_initial(opt.lam_g, self.lam_g0)"""

        self.opt = opt
        self.x = x
        self.u = u

    def add_dynamics_constr(self):
        opt = self.opt
        x, u = self.x, self.u
        # dynamics constraints
        for n in range(self.N):
            for ti in range(self.param.hmpc):
                opt.subject_to(x[n][:, ti+1] == self.fk_opt(x[n][:, ti], u[n][:, ti]))  #5 x 1 and 2 x 1

    def add_vwlim_constraint(self):
        opt = self.opt
        param = self.param
        sl = self.state_len
        x, u = self.x, self.u

        # try butterfly shape constraints
        for n in range(self.N):
            for ti in range(self.param.hcst):
                opt.subject_to(-1/param.vmax * ca.fabs(x[n][3, ti+1]) + 
                            1/param.wmax * x[n][4, ti+1] <= 0)

    def add_align_poly_constr(self, prev_cp, L):
        if len(prev_cp) == 0: return
        opt = self.opt
        param = self.param
        sl = self.state_len
        get_local_pt = self.get_local_pt
        x, u = self.x, self.u
        
        for cp_ids in prev_cp:
            cp_d = prev_cp[cp_ids][0:2, :]
            
            body_idx = np.where(cp_d[0, :] == L/2)[0] 
            assert(len(body_idx) > 0)
            body_idx = body_idx[0]
            body_id = cp_ids[body_idx]
            anchor_idx = 1 - body_idx
            anchor_id = cp_ids[anchor_idx]
            
            for ti in range(1, param.hcst):
                x_pt = get_local_pt(x[anchor_id][0:3, ti], 
                                        ca.MX(cp_d[:, anchor_idx]))
                xr = get_local_pt(x[body_id][0:3, ti],
                                        ca.MX([L/2, -L/2]))
                xl = get_local_pt(x[body_id][0:3, ti],
                                        ca.MX([L/2, L/2]))

                xR = x[body_id][0, ti]
                yR = x[body_id][1, ti]

                # all are variables, not values

                opt.subject_to((x_pt[1] - yR)*(xr[0] - xR) >= (xr[1] - yR)*(x_pt[0] - xR))
                opt.subject_to((x_pt[1] - yR)*(xR - xl[0]) >= (yR - xl[1])*(x_pt[0] - xR))
                opt.subject_to((x_pt[1] - xr[1])*(xl[0] - xr[0]) >= (xl[1] - xr[1])*(x_pt[0] - xr[0]))

    def add_pull_force_constr(self, prev_cp, L):
        if len(prev_cp) == 0: return
        opt = self.opt
        param = self.param
    
    def add_cp_cost(self, cp, ti, xy_param, t_param):   #this will return two cost functions for the two robots
        x, u = self.x, self.u
        prev_x = self.prev_x
        i0, i1 = next(iter(cp))   #real value - index0 and index1
        d0 = cp[(i0, i1)][:, 0]   #real value
        d1 = cp[(i0, i1)][:, 1]   #real value
        t0 = x[i0][2, ti]         #variable
        t1 = x[i1][2, ti]         #variable
        x_curr = self.fit_prev_x2opt(prev_x)    #N x 5
        print('x_curr: ' + str(x_curr))    #check that x_curr is changing per iteration
        cp_len = d0.shape[0]    #should be 3 for x,y,t
        
        x_diff1 = x[i0][0:cp_len, -1] - x_curr[i1][0:cp_len]   #3 x 1
        #x_diff1[0] += (ca.cos(t0)*d0[0] - ca.sin(t0)*d0[1]) - (math.cos(x_curr[i1][2]) - math.sin(x_curr[i1][2]))
        #x_diff1[1] += (ca.sin(t0)*d0[0] - ca.cos(t0)*d0[1]) - (math.sin(x_curr[i1][2]) - math.cos(x_curr[i1][2]))
        x_diff1 *= xy_param*2
        if cp_len > 2:
            x_diff1[2] = ca.tan(0.5*((t0-x_curr[i1][2]) - (d0[2]-d1[2])))
            x_diff1[2] *= t_param*10
        
        x_diff2 = x[i1][0:cp_len, -1] - x_curr[i0][0:cp_len]   #3 x 1
        #x_diff2[0] += (ca.cos(t1)*d1[0] - ca.sin(t1)*d1[1]) - (math.cos(x_curr[i0][2]) - math.sin(x_curr[i0][2]))
        #x_diff2[1] += (ca.sin(t1)*d1[0] - ca.cos(t1)*d1[1]) - (math.sin(x_curr[i0][2]) - math.cos(x_curr[i0][2]))
        x_diff2 *= xy_param*2
        if cp_len > 2:
            x_diff2[2] = ca.tan(0.5*((t1-x_curr[i0][2]) - (d1[2]-d0[2])))
            x_diff2[2] *= t_param*10
        
        return [ca.mtimes(x_diff1.T, x_diff1), ca.mtimes(x_diff2.T, x_diff2)]

    def align_cp_cost(self, cp, prev_cp):
        param = self.param
        cost = {}
        for ti in range(param.hmpc+1):
            for key in cp:
                curr = {key: cp[key]}
                i0, i1 = next(iter(curr))   #robot 1 and 2
                if ti < param.hmpc:
                    """cost_funs = self.add_cp_cost(curr, ti, param.cost_Q["s_cp_xy"], param.cost_Q["s_cp_t"])
                    if i0 in cost.keys():
                        cost[i0] += cost_funs[0]
                    else:
                        cost[i0] = cost_funs[0]
                    if i1 in cost.keys():
                        cost[i1] += cost_funs[1]
                    else:
                        cost[i1] = cost_funs[1]"""
                    continue
                else:
                    cost_funs = self.add_cp_cost(curr, ti, param.cost_Q["cp_xy"], param.cost_Q["cp_t"])
                    if i0 in cost.keys():
                        cost[i0] += cost_funs[0]
                    else:
                        cost[i0] = cost_funs[0]
                    if i1 in cost.keys():
                        cost[i1] += cost_funs[1]
                    else:
                        cost[i1] = cost_funs[1]
            for key in prev_cp:
                curr = {key: prev_cp[key]}
                i0, i1 = next(iter(curr))   #robot 1 and 2
                if ti < param.hmpc:
                    """cost_funs = self.add_cp_cost(curr, ti, param.cost_Q["s_prev_xy"], param.cost_Q["s_prev_t"])
                    if i0 in cost.keys():
                        cost[i0] += cost_funs[0]
                    else:
                        cost[i0] = cost_funs[0]
                    if i1 in cost.keys():
                        cost[i1] += cost_funs[1]
                    else:
                        cost[i1] = cost_funs[1]"""
                    continue
                else:
                    cost_funs = self.add_cp_cost(curr, ti, param.cost_Q["prev_xy"], param.cost_Q["prev_t"])
                    if i0 in cost.keys():
                        cost[i0] += cost_funs[0]
                    else:
                        cost[i0] = cost_funs[0]
                    if i1 in cost.keys():
                        cost[i1] += cost_funs[1]
                    else:
                        cost[i1] = cost_funs[1]
        return cost
    
    def init_cost(self, prev_x, zero_list=[]):
        param = self.param
        sl = self.state_len
        x = self.x
        x_curr = self.fit_prev_x2opt(prev_x)
            
        cost = []

        # mask zeros for zero_list
        # if len(zero_list) > 0:
        #     vs = ca.vec(x[1*(zero_list)][3, :])     #sus - indexing based on array? 
        #     ws = ca.vec(x[1*(zero_list)][4, :])
        #     for n in range(self.N):
        #         cost.append(ca.mtimes(vs.T, vs) * param.cost_Q["zero_xyt"]+ca.mtimes(ws.T, ws) * param.cost_Q["zero_xyt"])
        return cost
        
    def stage_cost(self):
        u = self.u
        param = self.param
        cost = {}
        for n in range(self.N):
            c = 0
            for ti in range(1, param.hmpc):
                c += ca.mtimes(u[n][:, ti].T, u[n][:, ti]) * param.cost_Q["Q_u"]
            cost[n] = c
        return cost

    def smooth_cost(self, prev_u):
        u = self.u
        param = self.param
        diff_u = np.zeros((self.N, 2))
        cost = {}
        for n in range(self.N):
            diff_u[n, :] = u[n][:, 0] - prev_u[n*2:n*2+1]
            diff_u[n, 0] *= param.cost_Q["smooth_v"]
            diff_u[n, 1] *= param.cost_Q["smooth_w"]
            cost[n] = ca.mtimes(diff_u[n, :], diff_u[n, :].T)
        for n2 in range(self.N):
            c = 0
            for ti in range(1, param.hmpc):
                diff_u[n2, :] = u[n2][:, ti] - u[n2][:, ti-1]
                diff_u[n2, 0] *= param.cost_Q["smooth_v"]
                diff_u[n2, 1] *= param.cost_Q["smooth_w"]
                c += ca.mtimes(diff_u[n2, :], diff_u[n2, :].T)
            cost[n] += c
        return cost

    def gdu_cost(self, gdu):
        u = self.u
        c = 0
        cost = []
        gdu = ca.MX(gdu.tolist())
        for n in range(self.N):
            c = 0
            for ti in range(self.param.hmpc):
                diff = u[n][:, ti] - gdu       #sus - unsure about dimension of gdu
                c += ca.mtimes(diff, diff.T)
            cost.append(c)
        return cost
    
    def optimize_cp(self, cost):
        opt = self.opt
        sl = self.state_len
        uv = []
        uw = []
        total_cost = 0
        for n in range(self.N):
            if n in cost.keys():
                c = cost[n]
            else:
                c = 0
            start = time()
            opt.minimize(c)
            opt.solver("ipopt", self.ipopt_param)
            try:
                ans = opt.solve()
                uv.append(ans.value(self.x[n][3, 1]))
                uw.append(ans.value(self.x[n][4, 1]))
                opt.set_initial(opt.lam_g, ans.value(opt.lam_g))#self.lam_g0 = ans.value(opt.lam_g)
                self.first = False
                total_cost += ans.value(c)
            except Exception as e:
                print(e)
                uv.append(0.0)
                uw.append(0.0)
            end = time()
            self.time_lists[n].append(end-start)
        print('return values: ' + str([uv, uw, total_cost]))
        return np.vstack([uv, uw]).T.flatten(), total_cost