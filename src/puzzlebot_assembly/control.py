import casadi as ca
import numpy as np
import math
from time import time
import multiprocessing as mp
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

########

def add_cp_cost(x, u, x_curr_big, cp, ti, xy_param, t_param, toggle):
    i0, i1 = next(iter(cp))   #real value - index0 and index1
    d0 = cp[(i0, i1)][:, 0]   #real value
    d1 = cp[(i0, i1)][:, 1]   #real value
    cp_len = d0.shape[0]    #should be 3 for x,y,t
    
    if toggle == 0:
        x_diff1 = x[0:cp_len, -1] - x_curr_big[i1][0:cp_len]   #3 x 1
        x_diff1 *= xy_param*2
        if cp_len > 2:
            x_diff1[2] = ca.tan(0.5*((x[2, ti]-x_curr_big[i1][2]) - (d0[2]-d1[2])))
            x_diff1[2] *= t_param*10
        return ca.mtimes(x_diff1.T, x_diff1)
    else:
        x_diff2 = x[0:cp_len, -1] - x_curr_big[i0][0:cp_len]   #3 x 1
        x_diff2 *= xy_param*2
        if cp_len > 2:
            x_diff2[2] = ca.tan(0.5*((x[2, ti]-x_curr_big[i0][2]) - (d1[2]-d0[2])))
            x_diff2[2] *= t_param*10
        return ca.mtimes(x_diff2.T, x_diff2)

def per_robot(N, param, sl, x_curr_big, fk_opt, get_local_pt, ipopt_param, n, prev_x, prev_u, L, cp, prev_cp=[]):
    opt = ca.Opti()
    x_curr = x_curr_big[n]
    
    x = opt.variable(sl, param.hmpc+1)
    u = opt.variable(2, param.hmpc)

    opt.subject_to(x[0, 0] == prev_x[0+3*n])
    opt.subject_to(x[1, 0] == prev_x[1+3*n])
    opt.subject_to(x[2, 0] == prev_x[2+3*n])
    opt.subject_to(x[3, 0] == prev_u[0+2*n])
    opt.subject_to(x[4, 0] == prev_u[1+2*n])
    opt.subject_to(opt.bounded(-param.vmax, ca.vec(x[3, :]), param.vmax))
    opt.subject_to(opt.bounded(-param.wmax, ca.vec(x[4, :]), param.wmax))
    opt.subject_to(opt.bounded(-param.uvmax, ca.vec(u[0, :]), param.uvmax))
    opt.subject_to(opt.bounded(-param.uwmax, ca.vec(u[1, :]), param.uwmax))
    
    for ti in range(param.hmpc + 1):
        opt.set_initial(x[:, ti], x_curr[:])
    opt.set_initial(u, 0)
    
    # dynamics constraints
    for ti in range(param.hmpc):
        opt.subject_to(x[:, ti+1] == fk_opt(x[:, ti], u[:, ti]))
        
    # try butterfly shape constraints
    for ti in range(param.hcst):
        opt.subject_to(-1/param.vmax * ca.fabs(x[3, ti+1]) + 
                    1/param.wmax * x[4, ti+1] <= 0)
                    
    # align_poly_contr
    for cp_ids in prev_cp:
        cp_d = prev_cp[cp_ids][0:2, :]
        
        body_idx = np.where(cp_d[0, :] == L/2)[0] 
        assert(len(body_idx) > 0)
        body_idx = body_idx[0]
        body_id = cp_ids[body_idx]
        anchor_idx = 1 - body_idx
        anchor_id = cp_ids[anchor_idx]
        
        if anchor_id == n:
            for ti in range(1, param.hcst):
                x_pt = get_local_pt(x[0:3, ti], ca.MX(cp_d[:, anchor_idx]))
                xr = get_local_pt(x_curr[body_id][0:3, ti], ca.MX([L/2, -L/2]))
                xl = get_local_pt(x_curr[body_id][0:3, ti], ca.MX([L/2, L/2]))
    
                xR = x_curr[body_id][0, ti]
                yR = x_curr[body_id][1, ti]
    
                opt.subject_to((x_pt[1] - yR)*(xr[0] - xR) >= (xr[1] - yR)*(x_pt[0] - xR))
                opt.subject_to((x_pt[1] - yR)*(xR - xl[0]) >= (yR - xl[1])*(x_pt[0] - xR))
                opt.subject_to((x_pt[1] - xr[1])*(xl[0] - xr[0]) >= (xl[1] - xr[1])*(x_pt[0] - xr[0]))
                
        elif body_id == n:
            for ti in range(1, param.hcst):
                x_pt = get_local_pt(x_curr[anchor_id][0:3, ti], ca.MX(cp_d[:, anchor_idx]))
                xr = get_local_pt(x[0:3, ti], ca.MX([L/2, -L/2]))
                xl = get_local_pt(x[0:3, ti], ca.MX([L/2, L/2]))
    
                xR = x[0, ti]
                yR = x[1, ti]
    
                opt.subject_to((x_pt[1] - yR)*(xr[0] - xR) >= (xr[1] - yR)*(x_pt[0] - xR))
                opt.subject_to((x_pt[1] - yR)*(xR - xl[0]) >= (yR - xl[1])*(x_pt[0] - xR))
                opt.subject_to((x_pt[1] - xr[1])*(xl[0] - xr[0]) >= (xl[1] - xr[1])*(x_pt[0] - xr[0]))
            
    # align_cp_cost
    cost = 0
    for key in cp:
        curr = {key: cp[key]}
        i0, i1 = next(iter(curr))   #robot 1 and 2
        if i0 == n:
            cost_fun = add_cp_cost(x, u, x_curr_big, curr, param.hmpc, param.cost_Q["cp_xy"], param.cost_Q["cp_t"], 0)
            if isinstance(cost, int) and cost == 0: cost = cost_fun
            else: cost += cost_fun
        elif i1 == n:
            cost_fun = add_cp_cost(x, u, x_curr_big, curr, param.hmpc, param.cost_Q["cp_xy"], param.cost_Q["cp_t"], 1)
            if isinstance(cost, int) and cost == 0: cost += cost_fun
            else: cost += cost_fun
    for key in prev_cp:
        curr = {key: prev_cp[key]}
        i0, i1 = next(iter(curr))   #robot 1 and 2
        if i0 == n:
            cost_fun = add_cp_cost(x, u, x_curr_big, curr, param.hmpc, param.cost_Q["prev_xy"], param.cost_Q["prev_t"], 0)
            if isinstance(cost, int) and cost == 0: cost = cost_fun
            else: cost += cost_fun
        elif i1 == n:
            cost_fun = add_cp_cost(x, u, x_curr_big, curr, param.hmpc, param.cost_Q["prev_xy"], param.cost_Q["prev_t"], 1)
            if isinstance(cost, int) and cost == 0: cost += cost_fun
            else: cost += cost_fun
                
    # stage_cost
    for ti in range(1, param.hmpc):
        cost += ca.mtimes(u[:, ti].T, u[:, ti]) * param.cost_Q["Q_u"]
        
    # optimize_cp
    obj_value = 0.0
    start = time()
    opt.minimize(cost)
    opt.solver("ipopt", ipopt_param)
    try:
        ans = opt.solve()
        uv = ans.value(x[3, 1])
        uw = ans.value(x[4, 1])
        #opt.set_initial(opt.lam_g, ans.value(opt.lam_g))
        obj_value = ans.value(cost)
    except Exception as e:
        print(e)
        uv = 0.0
        uw = 0.0
    end = time()
    #self.time_lists[n].append(end-start)
    return [uv, uw, obj_value, end-start] #np.vstack([uv, uw]).T.flatten(), obj_value

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
        #self.opt = None
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

    def final(self, prev_x, prev_u, L, cp, prev_cp):
        print('prev_cp: ' + str(prev_cp))
        uv_all = []
        uw_all = []
        total_cost = 0
        pool = mp.Pool()
        args = []
        x_curr = self.fit_prev_x2opt(prev_x)
        #results = []
        for n in range(self.N):
            #results.append(per_robot(self.N, self.param, self.state_len, x_curr, self.fk_opt, self.get_local_pt, self.ipopt_param, n, prev_x, prev_u, L, cp, prev_cp))
            args.append((self.N, self.param, self.state_len, x_curr, self.fk_opt, self.get_local_pt, self.ipopt_param, n, prev_x, prev_u, L, cp, prev_cp))
        results = pool.starmap(per_robot, args)
        self.prev_x = prev_x
        i = 0
        for [uv, uw, obj_value, t] in results:
            total_cost += obj_value
            uv_all.append(uv)
            uw_all.append(uw)
            self.time_lists[i].append(t)
            i += 1
        return np.vstack([uv_all, uw_all]).T.flatten(), total_cost