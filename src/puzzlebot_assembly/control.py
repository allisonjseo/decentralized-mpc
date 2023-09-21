import casadi as ca
import numpy as np
import math
from time import time
import multiprocessing as mp
#decentralized and parallelized - latest

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

    def get_relative_pt(self):
        xi = ca.SX.sym('xi', 3) # ego robot
        xj = ca.SX.sym('xj', 3) # other robot
        xj_pt = ca.SX.sym('xj_pt', 2) # point of interest on other robot
        ti = xi[2]
        tj = xj[2]
        xi_pt_x = ca.cos(tj-ti)*(xj_pt[0]) - ca.sin(tj-ti)*(xj_pt[1]) + ca.cos(-ti)*(xj[0]-xi[0]) - ca.sin(-ti)*(xj[1]-xi[1])
        return ca.Function("get_relative_pt", [xi, xj, xj_pt], [xi_pt_x])

    def fk_opt_force(self, N, dt):
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

    def fk_opt(self, N, dt):
        x_sym = ca.SX.sym('x', 3)
        u_sym = ca.SX.sym('u', 2)

        theta = x_sym[2]
        x_dot = ca.SX.zeros(3)
        x_dot[0] = u_sym[0] * ca.cos(theta)
        x_dot[1] = u_sym[0] * ca.sin(theta)
        x_dot[2] = u_sym[1]

        return ca.Function('fk_opt', [x_sym, u_sym], [x_sym + (x_dot * dt)])

########

def add_cp_cost(x, u, x_curr_big, cp, ti, xy_param, t_param, toggle):
    i0, i1 = next(iter(cp))   #real value - index0 and index1
    d0 = cp[(i0, i1)][:, 0]   #real value
    d1 = cp[(i0, i1)][:, 1]   #real value
    cp_len = d0.shape[0]    #should be 3 for x,y,t
    
    if toggle == 0:
        x_diff1 = x[0:cp_len, ti] - x_curr_big[i1][0:cp_len]   #3 x 1
        x_diff1 *= xy_param*2
        if cp_len > 2:
            x_diff1[2] = ca.tan(0.5*((x[2, ti]-x_curr_big[i1][2]) - (d0[2]-d1[2])))
            x_diff1[2] *= t_param*10
        return ca.mtimes(x_diff1.T, x_diff1)
    else:
        x_diff2 = x[0:cp_len, ti] - x_curr_big[i0][0:cp_len]   #3 x 1
        x_diff2 *= xy_param*2
        if cp_len > 2:
            x_diff2[2] = ca.tan(0.5*((x[2, ti]-x_curr_big[i0][2]) - (d1[2]-d0[2])))
            x_diff2[2] *= t_param*10
        return ca.mtimes(x_diff2.T, x_diff2)

def per_robot(N, param, sl, x_curr_big, fk_opt, get_local_pt, get_relative_pt, ipopt_param, n, prev_x, prev_u, L, cp, eth, prev_cp=[]):
    # init_opt
    opt = ca.Opti()
    x_curr = x_curr_big[n]
    
    x = opt.variable(sl, param.hmpc+1)
    u = opt.variable(2, param.hmpc)

    opt.subject_to(x[0, 0] == prev_x[0+3*n])
    opt.subject_to(x[1, 0] == prev_x[1+3*n])
    opt.subject_to(x[2, 0] == prev_x[2+3*n])
    opt.subject_to(opt.bounded(-param.vmax, ca.vec(u[0, :]), param.vmax))
    opt.subject_to(opt.bounded(-param.wmax, ca.vec(u[1, :]), param.wmax))
    
    for ti in range(param.hmpc + 1):
        opt.set_initial(x[:, ti], x_curr[:])
    for ti in range(param.hmpc):
        opt.set_initial(u[:, ti], prev_u[2*n:2*n + 2])
    
    # dynamics constraints
    for ti in range(param.hmpc):
        opt.subject_to(x[:, ti+1] == fk_opt(x[:, ti], u[:, ti]))

    # vwlim constraints    
    # try butterfly shape constraints
    """for ti in range(param.hmpc - 1):
        opt.subject_to(-1/param.vmax * ca.fabs(u[0, ti+1]) + 
                    1/param.wmax * u[1, ti+1] <= 0)"""
    
    # do diamond shape constraints instead of butterfly
    for ti in range(param.hmpc):
        opt.subject_to(1/param.vmax * u[0, ti] + 
                        1/param.wmax * u[1, ti] <=1)
        opt.subject_to(-1/param.vmax * u[0, ti] + 
                        1/param.wmax * u[1, ti] <=1)
        opt.subject_to(1/param.vmax * u[0, ti] - 
                        1/param.wmax * u[1, ti] <=1)
        opt.subject_to(-1/param.vmax * u[0, ti] - 
                        1/param.wmax * u[1, ti] <=1)
    
    cost = 0
    # stage_cost
    for ti in range(1, param.hmpc):
        cost += ca.mtimes(u[:, ti].T, u[:, ti]) * param.cost_Q["Q_u"]

    # smooth_cost
    diff_u = u[:, 0] - prev_u[2*n:2*n+2]
    diff_u[0] *= param.cost_Q["smooth_v"]
    diff_u[1] *= param.cost_Q["smooth_w"]
    cost = ca.mtimes(diff_u.T, diff_u)
    for ti in range(1, param.hmpc):
        diff_u = u[:, ti] - u[:, ti-1]
        diff_u[0] *= param.cost_Q["smooth_v"]
        diff_u[1] *= param.cost_Q["smooth_w"]
        cost += ca.mtimes(diff_u.T, diff_u)

    # check if the current constraints are satisfied
    # (this checks for all prev_cps - may change so that we only check the pair of current robot n)
    aligned_cps = {}
    unaligned_cps = cp.copy()
    for cp_ids in prev_cp:
        cp_d = prev_cp[cp_ids]
        
        body_idx = np.where(cp_d[0, :] == L/2)[0] 
        assert(len(body_idx) > 0)
        body_idx = body_idx[0]
        body_x = prev_x[3*cp_ids[body_idx]:3*(cp_ids[body_idx]+1)]
        anchor_idx = 1 - body_idx
        aid = cp_ids[anchor_idx]
        ax = (np.array([L-eth, 0]) + cp_d[0:2, anchor_idx])[0]

        anchor_pt = body2world(prev_x[3*aid:3*(aid+1)],
                cp_d[:, anchor_idx])
        poly_corners = body2world(body_x, 
                                    np.array([[L/2,L/2],[L/2,-L/2]]).T).tolist()
        poly_corners += body2world(body_x, 
                                    np.array([[ax,-L/2],[ax,L/2]]).T).tolist()
        is_in = is_inside_poly(anchor_pt, np.array(poly_corners))
        print(", is in:", is_in)
        if is_in:
            aligned_cps[cp_ids] = cp_d
        else:
            unaligned_cps[cp_ids] = cp_d

    # align_cp_cost
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
    
    # align_poly_constr
    for cp_ids in prev_cp:
        cp_d = prev_cp[cp_ids][0:2, :]
        
        body_idx = np.where(cp_d[0, :] == L/2)[0] 
        assert(len(body_idx) > 0)
        body_idx = body_idx[0]
        body_id = cp_ids[body_idx]
        anchor_idx = 1 - body_idx
        anchor_id = cp_ids[anchor_idx]

        anchor_tail_pt = cp_d[:, anchor_idx]
        anchor_body_pt = np.array([L, 0]) + anchor_tail_pt
    
        if anchor_id == n:
            for ti in range(1, param.hcst):
                anchor_tail_pt = cp_d[:, anchor_idx]
                anchor_body_pt = np.array([L, 0]) + anchor_tail_pt

                x_pt = get_local_pt(x[0:3, ti], ca.MX(anchor_tail_pt))
                xr = get_local_pt(x_curr_big[body_id][0:3], ca.MX([L/2+eth, -L/2]))
                xl = get_local_pt(x_curr_big[body_id][0:3], ca.MX([L/2+eth, L/2]))
                x_lim, y_lim = [0.01, 0.01]
                xl = get_local_pt(x_curr_big[body_id][0:3], ca.MX([L/2+eth, -y_lim]))
                xr = get_local_pt(x_curr_big[body_id][0:3], ca.MX([L/2+eth, y_lim]))
                xls = get_local_pt(x_curr_big[body_id][0:3], ca.MX([x_lim, -y_lim]))
                xrs = get_local_pt(x_curr_big[body_id][0:3], ca.MX([x_lim, y_lim]))
                opt.subject_to((x_pt[1] - xls[1])*(xl[0] - xls[0]) >= (xl[1] - xls[1])*(x_pt[0] - xls[0]))
                opt.subject_to((x_pt[1] - xrs[1])*(xls[0] - xrs[0]) >= (xls[1] - xrs[1])*(x_pt[0] - xrs[0]))
                opt.subject_to((x_pt[1] - xr[1])*(xrs[0] - xr[0]) >= (xrs[1] - xr[1])*(x_pt[0] - xr[0]))
                opt.subject_to((x_pt[1] - xl[1])*(xr[0] - xl[0]) >= (xr[1] - xl[1])*(x_pt[0] - xl[0]))
                
        elif body_id == n:
            for ti in range(1, param.hcst):
                anchor_tail_pt = cp_d[:, anchor_idx]
                anchor_body_pt = np.array([L, 0]) + anchor_tail_pt

                x_pt = get_local_pt(x_curr_big[anchor_id][0:3], ca.MX(anchor_tail_pt))
                xr = get_local_pt(x[0:3, ti], ca.MX([L/2+eth, -L/2]))
                xl = get_local_pt(x[0:3, ti], ca.MX([L/2+eth, L/2]))
                x_lim, y_lim = [0.01, 0.01]
                xl = get_local_pt(x[0:3, ti], ca.MX([L/2+eth, -y_lim]))
                xr = get_local_pt(x[0:3, ti], ca.MX([L/2+eth, y_lim]))
                xls = get_local_pt(x[0:3, ti], ca.MX([x_lim, -y_lim]))
                xrs = get_local_pt(x[0:3, ti], ca.MX([x_lim, y_lim]))
                opt.subject_to((x_pt[1] - xls[1])*(xl[0] - xls[0]) >= (xl[1] - xls[1])*(x_pt[0] - xls[0]))
                opt.subject_to((x_pt[1] - xrs[1])*(xls[0] - xrs[0]) >= (xls[1] - xrs[1])*(x_pt[0] - xrs[0]))
                opt.subject_to((x_pt[1] - xr[1])*(xrs[0] - xr[0]) >= (xrs[1] - xr[1])*(x_pt[0] - xr[0]))
                opt.subject_to((x_pt[1] - xl[1])*(xr[0] - xl[0]) >= (xr[1] - xl[1])*(x_pt[0] - xl[0]))
        
        """if anchor_id == n:
            for ti in range(1, param.hcst):
                x_pt = get_local_pt(x[0:3, ti], ca.MX(cp_d[:, anchor_idx]))
                xr = get_local_pt(x_curr_big[body_id][0:3], ca.MX([L/2, -L/2]))
                xl = get_local_pt(x_curr_big[body_id][0:3], ca.MX([L/2, L/2]))
    
                xR = x_curr_big[body_id][0]
                yR = x_curr_big[body_id][1]
    
                opt.subject_to((x_pt[1] - yR)*(xr[0] - xR) >= (xr[1] - yR)*(x_pt[0] - xR))
                opt.subject_to((x_pt[1] - yR)*(xR - xl[0]) >= (yR - xl[1])*(x_pt[0] - xR))
                opt.subject_to((x_pt[1] - xr[1])*(xl[0] - xr[0]) >= (xl[1] - xr[1])*(x_pt[0] - xr[0]))
                
        elif body_id == n:
            for ti in range(1, param.hcst):
                x_pt = get_local_pt(x_curr_big[anchor_id][0:3], ca.MX(cp_d[:, anchor_idx]))
                xr = get_local_pt(x[0:3, ti], ca.MX([L/2, -L/2]))
                xl = get_local_pt(x[0:3, ti], ca.MX([L/2, L/2]))
    
                xR = x[0, ti]
                yR = x[1, ti]
    
                opt.subject_to((x_pt[1] - yR)*(xr[0] - xR) >= (xr[1] - yR)*(x_pt[0] - xR))
                opt.subject_to((x_pt[1] - yR)*(xR - xl[0]) >= (yR - xl[1])*(x_pt[0] - xR))
                opt.subject_to((x_pt[1] - xr[1])*(xl[0] - xr[0]) >= (xl[1] - xr[1])*(x_pt[0] - xr[0]))"""
            
    # body_line_constr
    for cp_ids in prev_cp:
        cp_d = prev_cp[cp_ids][0:2, :]

        body_idx = np.where(cp_d[0, :] == L/2)[0] 
        assert(len(body_idx) > 0)
        body_idx = body_idx[0]
        body_id = cp_ids[body_idx]
        anchor_idx = 1 - body_idx
        anchor_id = cp_ids[anchor_idx]

        if anchor_id == n:
            xa_bl_curr = get_relative_pt_num(prev_x[3*body_id:(3*body_id+3)], prev_x[3*anchor_id:(3*anchor_id+3)], np.array([-L/2,L/2]))
            xa_br_curr = get_relative_pt_num(prev_x[3*body_id:(3*body_id+3)], prev_x[3*anchor_id:(3*anchor_id+3)], np.array([-L/2,-L/2]))
            xb_fl_curr = get_relative_pt_num(prev_x[3*anchor_id:(3*anchor_id+3)], prev_x[3*body_id:(3*body_id+3)], np.array([L/2,L/2]))
            xb_fr_curr = get_relative_pt_num(prev_x[3*anchor_id:(3*anchor_id+3)], prev_x[3*body_id:(3*body_id+3)], np.array([L/2,-L/2]))
            constr_mask = np.array([xa_bl_curr >= L/2-eth, xa_br_curr >= L/2-eth, xb_fl_curr <= -L/2+eth, xb_fr_curr <= -L/2+eth])
            for ti in range(1, param.hcst):
                # for anchor robot, limit the relative position of the body robot to be smaller than back line
                if constr_mask[2] and constr_mask[3]:
                    xb_fl = get_relative_pt(x[0:3, ti], x_curr_big[body_id][0:3], ca.MX([L/2, L/2]))
                    opt.subject_to(xb_fl <= -L/2+eth)
                    xb_fr = get_relative_pt(x[0:3, ti], x_curr_big[body_id][0:3], ca.MX([L/2, -L/2]))
                    opt.subject_to(xb_fr <= -L/2+eth)

                # for body robot, limit the relative position of the anchor robot to be larger than front line
                elif constr_mask[0] and constr_mask[1]:
                    xa_bl = get_relative_pt(x_curr_big[body_id][0:3], x[0:3, ti], ca.MX([-L/2, L/2]))
                    opt.subject_to(xa_bl >= L/2-eth)
                    xa_br = get_relative_pt(x_curr_big[body_id][0:3], x[0:3, ti], ca.MX([-L/2, -L/2]))
                    opt.subject_to(xa_br >= L/2-eth)
                
        elif body_id == n:
            xa_bl_curr = get_relative_pt_num(prev_x[3*body_id:(3*body_id+3)], prev_x[3*anchor_id:(3*anchor_id+3)], np.array([-L/2,L/2]))
            xa_br_curr = get_relative_pt_num(prev_x[3*body_id:(3*body_id+3)], prev_x[3*anchor_id:(3*anchor_id+3)], np.array([-L/2,-L/2]))
            xb_fl_curr = get_relative_pt_num(prev_x[3*anchor_id:(3*anchor_id+3)], prev_x[3*body_id:(3*body_id+3)], np.array([L/2,L/2]))
            xb_fr_curr = get_relative_pt_num(prev_x[3*anchor_id:(3*anchor_id+3)], prev_x[3*body_id:(3*body_id+3)], np.array([L/2,-L/2]))
            constr_mask = np.array([xa_bl_curr >= L/2-eth, xa_br_curr >= L/2-eth, xb_fl_curr <= -L/2+eth, xb_fr_curr <= -L/2+eth])
            for ti in range(1, param.hcst):
                # for anchor robot, limit the relative position of the body robot to be smaller than back line
                if constr_mask[2] and constr_mask[3]:
                    xb_fl = get_relative_pt(x_curr_big[anchor_id][0:3], x[0:3, ti], ca.MX([L/2, L/2]))
                    opt.subject_to(xb_fl <= -L/2+eth)
                    xb_fr = get_relative_pt(x_curr_big[anchor_id][0:3], x[0:3, ti], ca.MX([L/2, -L/2]))
                    opt.subject_to(xb_fr <= -L/2+eth)

                # for body robot, limit the relative position of the anchor robot to be larger than front line
                elif constr_mask[0] and constr_mask[1]:
                    xa_bl = get_relative_pt(x[0:3, ti], x_curr_big[anchor_id][0:3], ca.MX([-L/2, L/2]))
                    opt.subject_to(xa_bl >= L/2-eth)
                    xa_br = get_relative_pt(x[0:3, ti], x_curr_big[anchor_id][0:3], ca.MX([-L/2, -L/2]))
                    opt.subject_to(xa_br >= L/2-eth)
                
        """if anchor_id == n:
            for ti in range(1, param.hcst):
                # for body robot, limit the relative position of the anchor robot to be larger than front line
                xa_bl = get_relative_pt(x_curr_big[body_id][0:3], x[0:3, ti], ca.MX([-L/2, L/2]))
                xa_br = get_relative_pt(x_curr_big[body_id][0:3], x[0:3, ti], ca.MX([-L/2, -L/2]))
                opt.subject_to(xa_bl >= L/2)
                opt.subject_to(xa_br >= L/2)

                # for anchor robot, limit the relative position of the body robot to be smaller than back line
                xb_fl = get_relative_pt(x[0:3, ti], x_curr_big[body_id][0:3], ca.MX([L/2, L/2]))
                xb_fr = get_relative_pt(x[0:3, ti], x_curr_big[body_id][0:3], ca.MX([L/2, -L/2]))
                opt.subject_to(xb_fl <= -L/2)
                opt.subject_to(xb_fr <= -L/2)

        elif body_id == n:
            for ti in range(1, param.hcst):
                # for body robot, limit the relative position of the anchor robot to be larger than front line
                #xa_bl = get_relative_pt(x[sl*body_id:(sl*body_id+3), ti], x[sl*anchor_id:(sl*anchor_id+3), ti], ca.MX([-L/2, L/2]))
                xa_bl = get_relative_pt(x[0:3, ti], x_curr_big[anchor_id][0:3], ca.MX([-L/2, L/2]))
                xa_br = get_relative_pt(x[0:3, ti], x_curr_big[anchor_id][0:3], ca.MX([-L/2, -L/2]))
                opt.subject_to(xa_bl >= L/2)
                opt.subject_to(xa_br >= L/2)

                # for anchor robot, limit the relative position of the body robot to be smaller than back line
                xb_fl = get_relative_pt(x_curr_big[anchor_id][0:3], x[0:3, ti], ca.MX([L/2, L/2]))
                xb_fr = get_relative_pt(x_curr_big[anchor_id][0:3], x[0:3, ti], ca.MX([L/2, -L/2]))
                opt.subject_to(xb_fl <= -L/2)
                opt.subject_to(xb_fr <= -L/2)"""
        
    # optimize_cp
    obj_value = 0.0
    start = time()
    opt.minimize(cost)
    opt.solver("ipopt", ipopt_param)
    try:
        ans = opt.solve()
        #uv = ans.value(x[3, 1])
        #uw = ans.value(x[4, 1])
        v = ans.value(u[0, 0])
        w = ans.value(u[1, 0])
        #opt.set_initial(opt.lam_g, ans.value(opt.lam_g))
        obj_value = ans.value(cost)
    except Exception as e:
        print(e)
        v = 0.0
        w = 0.0
    end = time()
    #self.time_lists[n].append(end-start)
    return [v, w, obj_value, end-start] #np.vstack([uv, uw]).T.flatten(), obj_value

class Controller:
    def __init__(self, N, dt, control_param):
        self.N = N
        self.dt = dt
        self.param = control_param
        self.state_len = 3
        self.ca_int = CasadiInterface(N, dt, self.state_len, M=0.1)
        self.fk_opt = self.ca_int.fk_opt(N, dt)
        self.get_local_pt = self.ca_int.get_local_pt()
        self.get_relative_pt = self.ca_int.get_relative_pt()
        self.ipopt_param = {"verbose": False, 
                            "ipopt.print_level": 0,
                            "print_time": 0,
                            'ipopt.sb': 'yes',
                            "ipopt.constr_viol_tol": 1e-6
                            }
        self.opt = None
        self.x = None # 3 states [x, y, theta]
        self.u = None # 2 controls [v, w]
        self.time_lists = {}
        #self.pool = p
        for n in range(N):
            self.time_lists[n] = []
        self.cost_list = []
        self.lam_g0 = None
        self.first = True
        self.eth = 1.5e-3

        # for debug
        self.prev_x = None
    
    def fit_prev_x2opt(self, prev_x):
        x_curr = np.zeros([self.N, self.state_len])   #N rows, 3 cols
        x_curr[:, 0:3] = prev_x.reshape([self.N, 3])
        #x_curr = x_curr.flatten()
        return x_curr

    def final(self, prev_x, prev_u, L, cp, prev_cp, pool):
        print('prev_cp: ' + str(prev_cp))
        uv_all = []
        uw_all = []
        total_cost = 0
        args = []
        x_curr = self.fit_prev_x2opt(prev_x)
        #results = []
        for n in range(self.N):
            #results.append(per_robot(self.N, self.param, self.state_len, x_curr, self.fk_opt, self.get_local_pt, self.ipopt_param, n, prev_x, prev_u, L, cp, prev_cp))
            args.append((self.N, self.param, self.state_len, x_curr, self.fk_opt, self.get_local_pt, self.get_relative_pt, self.ipopt_param, n, prev_x, prev_u, L, cp, self.eth, prev_cp))
        results = pool.starmap(per_robot, args)
        self.prev_x = prev_x
        i = 0
        for [uv, uw, obj_value, t] in results:
            total_cost += obj_value
            uv_all.append(uv)
            uw_all.append(uw)
            self.time_lists[i].append(t)
            print(t)
            i += 1
        return np.vstack([uv_all, uw_all]).T.flatten(), total_cost