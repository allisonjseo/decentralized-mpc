import casadi as ca
import numpy as np

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
            "smooth_v": 0.1, "smooth_w":1e0,
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
        x_sym = ca.SX.sym('x', 5*N)
        u_sym = ca.SX.sym('u', 2*N)

        theta = x_sym[2::5]
        vs = x_sym[3::5]
        x_dot = ca.SX.zeros(5*N)
        x_dot[0::5] = vs * ca.cos(theta)
        x_dot[1::5] = vs * ca.sin(theta)
        x_dot[2::5] = x_sym[4::5]
        x_dot[3::5] = u_sym[0::2]
        x_dot[4::5] = u_sym[1::2]

        return ca.Function("fk_opt", [x_sym, u_sym], [x_sym + (x_dot * dt)])

    def fk_opt(self, N, dt):
        x_sym = ca.SX.sym('x', 3*N)
        u_sym = ca.SX.sym('u', 2*N)

        theta = x_sym[2::3]
        x_dot = ca.SX.zeros(3*N)
        x_dot[0::3] = u_sym[0::2] * ca.cos(theta)
        x_dot[1::3] = u_sym[0::2] * ca.sin(theta)
        x_dot[2::3] = u_sym[1::2]
        return ca.Function("fk_opt", [x_sym, u_sym], [x_sym + (x_dot * dt)])

    def dd_fx_opt(self, theta, N):
        F = ca.SX.zeros(3*N, 2*N)
        F[0::3, 0::2] = ca.diag(ca.cos(theta))
        F[1::3, 0::2] = ca.diag(ca.sin(theta))
        F[2::3, 1::2] = ca.SX.eye(N)
        return F

    def fk_exact_opt(self, N, dt):
        x_sym = ca.SX.sym('x', 3*N)
        u_sym = ca.SX.sym('u', 2*N)
        dx = ca.SX.zeros(3*N)
        dx[0::3] = u_sym[0::2]/(u_sym[1::2] + 1e-6) * (
                                ca.sin(x_sym[2::3] + u_sym[1::2]*dt)
                                - ca.sin(x_sym[2::3]))
        dx[1::3] = - u_sym[0::2]/(u_sym[1::2] + 1e-6) * (
                                ca.cos(x_sym[2::3] + u_sym[1::2]*dt)
                                - ca.cos(x_sym[2::3]))
        dx[2::3] = u_sym[1::2] * dt
        return ca.Function("fk_exact_opt", [x_sym, u_sym], [x_sym + dx])

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

        # for debug
        self.prev_x = None
    
    def fit_prev_x2opt(self, prev_x):
        x_curr = np.zeros([self.N, self.state_len])
        x_curr[:, 0:3] = prev_x.reshape([self.N, 3])
        x_curr = x_curr.flatten()
        return x_curr

    def init_opt(self, prev_x, prev_u, prev_cp=[]):
        N = self.N
        param = self.param
        opt = ca.Opti()
        sl = self.state_len
        x = opt.variable(sl*N, param.hmpc + 1)
        u = opt.variable(2*N, param.hmpc)

        # for debug
        self.prev_x = prev_x

        # initial state constraints
        opt.subject_to(x[0::sl, 0] == prev_x[0::3])
        opt.subject_to(x[1::sl, 0] == prev_x[1::3])
        opt.subject_to(x[2::sl, 0] == prev_x[2::3])

        # uv, uw constraints
        opt.subject_to(opt.bounded(-param.vmax, 
                            ca.vec(u[0::2, :]), param.vmax))
        opt.subject_to(opt.bounded(-param.wmax, 
                            ca.vec(u[1::2, :]), param.wmax))

        #try warm start
        x_curr = self.fit_prev_x2opt(prev_x)
        for ti in range(param.hmpc + 1):
            opt.set_initial(x[:, ti], x_curr)
        print("prev_u: ", prev_u)
        for ti in range(param.hmpc):
            opt.set_initial(u[:, ti], prev_u)

        self.opt = opt
        self.x = x
        self.u = u

    def add_dynamics_constr(self):
        opt = self.opt
        x, u = self.x, self.u
        # dynamics constraints
        for ti in range(self.param.hmpc):
            opt.subject_to(x[:, ti+1] == self.fk_opt(x[:, ti], u[:, ti]))

    def add_vwlim_constraint(self):
        opt = self.opt
        param = self.param
        sl = self.state_len
        x, u = self.x, self.u

        # try butterfly shape constraints
        #  for ti in range(self.param.hcst):
            #  opt.subject_to(-1/param.vmax * ca.fabs(x[3::sl, ti+1]) + 
                        #  1/param.wmax * x[4::sl, ti+1] <= 0)

        # do diamond shape constraints
        for ti in range(self.param.hmpc):
            opt.subject_to(1/param.vmax * u[0::2, ti] + 
                           1/param.wmax * u[1::2, ti] <=1)
            opt.subject_to(-1/param.vmax * u[0::2, ti] + 
                           1/param.wmax * u[1::2, ti] <=1)
            opt.subject_to(1/param.vmax * u[0::2, ti] - 
                           1/param.wmax * u[1::2, ti] <=1)
            opt.subject_to(-1/param.vmax * u[0::2, ti] - 
                           1/param.wmax * u[1::2, ti] <=1)

    def add_align_poly_constr(self, prev_cp, L):
        if len(prev_cp) == 0: return
        opt = self.opt
        param = self.param
        eth = param.eth
        sl = self.state_len
        #  get_local_pt = self.ca_int.get_local_pt
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
                anchor_tail_pt = cp_d[:, anchor_idx]
                anchor_body_pt = np.array([L, 0]) + anchor_tail_pt

                x_pt = get_local_pt(x[sl*anchor_id:(sl*anchor_id+3), ti], 
                                        ca.MX(anchor_tail_pt))
                xr = get_local_pt(x[sl*body_id:(sl*body_id+3), ti],
                                        ca.MX([L/2+eth, -L/2]))
                xl = get_local_pt(x[sl*body_id:(sl*body_id+3), ti],
                                        ca.MX([L/2+eth, L/2]))
                
                # test square constraints
                # ax = anchor_body_pt[0]
                # xrs = get_local_pt(x[sl*body_id:(sl*body_id+3), ti],
                #                         ca.mx([ax-eth, -l/2]))
                # xls = get_local_pt(x[sl*body_id:(sl*body_id+3), ti],
                #                         ca.mx([ax-eth, l/2]))

                # opt.subject_to((x_pt[1] - xrs[1])*(xr[0] - xrs[0]) >= (xr[1] - xrs[1])*(x_pt[0] - xrs[0]))
                # opt.subject_to((x_pt[1] - xls[1])*(xrs[0] - xls[0]) >= (xrs[1] - xls[1])*(x_pt[0] - xls[0]))
                # opt.subject_to((x_pt[1] - xl[1])*(xls[0] - xl[0]) >= (xls[1] - xl[1])*(x_pt[0] - xl[0]))
                # opt.subject_to((x_pt[1] - xr[1])*(xl[0] - xr[0]) >= (xl[1] - xr[1])*(x_pt[0] - xr[0]))

                # test experiment square constraints
                x_lim, y_lim = [0.01, 0.01]
                xl = get_local_pt(x[sl*body_id:(sl*body_id+3), ti],
                                        ca.MX([L/2+eth, -y_lim]))
                xr = get_local_pt(x[sl*body_id:(sl*body_id+3), ti],
                                        ca.MX([L/2+eth, y_lim]))
                xls = get_local_pt(x[sl*body_id:(sl*body_id+3), ti],
                                        ca.MX([x_lim, -y_lim]))
                xrs = get_local_pt(x[sl*body_id:(sl*body_id+3), ti],
                                        ca.MX([x_lim, y_lim]))
                
                # constraint points need to be counter-clockwise
                opt.subject_to((x_pt[1] - xls[1])*(xl[0] - xls[0]) >= (xl[1] - xls[1])*(x_pt[0] - xls[0]))
                opt.subject_to((x_pt[1] - xrs[1])*(xls[0] - xrs[0]) >= (xls[1] - xrs[1])*(x_pt[0] - xrs[0]))
                opt.subject_to((x_pt[1] - xr[1])*(xrs[0] - xr[0]) >= (xrs[1] - xr[1])*(x_pt[0] - xr[0]))
                opt.subject_to((x_pt[1] - xl[1])*(xr[0] - xl[0]) >= (xr[1] - xl[1])*(x_pt[0] - xl[0]))

                # xR = x[sl*body_id, ti]
                # yR = x[sl*body_id+1, ti]

                # test out a off center point 
                # x_ct = get_local_pt(x[sl*body_id:(sl*body_id+3)],
                #                         ca.MX(anchor_body_pt))
                # xR = x_ct[0]
                # yR = x_ct[1]

                # opt.subject_to((x_pt[1] - yR)*(xr[0] - xR) >= (xr[1] - yR)*(x_pt[0] - xR))
                # opt.subject_to((x_pt[1] - yR)*(xR - xl[0]) >= (yR - xl[1])*(x_pt[0] - xR))
                # opt.subject_to((x_pt[1] - xr[1])*(xl[0] - xr[0]) >= (xl[1] - xr[1])*(x_pt[0] - xr[0]))
    
    def add_body_line_constr(self, prev_cp, L):
        x, u = self.x, self.u
        opt = self.opt
        param = self.param
        eth = param.eth
        sl = self.state_len
        get_relative_pt = self.get_relative_pt

        for cp_ids in prev_cp:
            cp_d = prev_cp[cp_ids][0:2, :]
            body_idx = np.where(cp_d[0, :] == L/2)[0] 
            assert(len(body_idx) > 0)
            body_idx = body_idx[0]
            body_id = cp_ids[body_idx]
            anchor_idx = 1 - body_idx
            anchor_id = cp_ids[anchor_idx]

            # check if constraints can be satisfied
            xa_bl_curr = get_relative_pt_num(self.prev_x[3*body_id:(3*body_id+3)], self.prev_x[3*anchor_id:(3*anchor_id+3)], np.array([-L/2,L/2]))
            xa_br_curr = get_relative_pt_num(self.prev_x[3*body_id:(3*body_id+3)], self.prev_x[3*anchor_id:(3*anchor_id+3)], np.array([-L/2,-L/2]))
            xb_fl_curr = get_relative_pt_num(self.prev_x[3*anchor_id:(3*anchor_id+3)], self.prev_x[3*body_id:(3*body_id+3)], np.array([L/2,L/2]))
            xb_fr_curr = get_relative_pt_num(self.prev_x[3*anchor_id:(3*anchor_id+3)], self.prev_x[3*body_id:(3*body_id+3)], np.array([L/2,-L/2]))
            constr_mask = np.array([xa_bl_curr >= L/2-eth, xa_br_curr >= L/2-eth, xb_fl_curr <= -L/2+eth, xb_fr_curr <= -L/2+eth])
            #  print("xa_bl_curr: ", xa_bl_curr, ", xa_br_curr: ", xa_br_curr, ", xb_fl_curr: ", xb_fl_curr, ", xb_fr_curr: ", xb_fr_curr)
            #  print("constr_mask: ", constr_mask)
            
            for ti in range(1, param.hcst):
                # for anchor robot, limit the relative position of the body robot to be smaller than back line
                if constr_mask[2] and constr_mask[3]:
                    xb_fl = get_relative_pt(x[sl*anchor_id:(sl*anchor_id+3), ti], x[sl*body_id:(sl*body_id+3), ti], ca.MX([L/2, L/2]))
                    opt.subject_to(xb_fl <= -L/2+eth)
                    xb_fr = get_relative_pt(x[sl*anchor_id:(sl*anchor_id+3), ti], x[sl*body_id:(sl*body_id+3), ti], ca.MX([L/2, -L/2]))
                    opt.subject_to(xb_fr <= -L/2+eth)

                # for body robot, limit the relative position of the anchor robot to be larger than front line
                elif constr_mask[0] and constr_mask[1]:
                    xa_bl = get_relative_pt(x[sl*body_id:(sl*body_id+3), ti], x[sl*anchor_id:(sl*anchor_id+3), ti], ca.MX([-L/2, L/2]))
                    opt.subject_to(xa_bl >= L/2-eth)
                    xa_br = get_relative_pt(x[sl*body_id:(sl*body_id+3), ti], x[sl*anchor_id:(sl*anchor_id+3), ti], ca.MX([-L/2, -L/2]))
                    opt.subject_to(xa_br >= L/2-eth)

    def add_cp_cost(self, cp, ti, xy_param, t_param):
        x, u = self.x, self.u
        sl = self.state_len
        cost = 0
        i0, i1 = next(iter(cp))
        d0 = cp[(i0, i1)][:, 0]
        d1 = cp[(i0, i1)][:, 1]
        t0 = x[i0*sl+2, ti]
        t1 = x[i1*sl+2, ti]
        cp_len = d0.shape[0]
        x_diff = x[i0*sl:(i0*sl+cp_len), ti] - x[i1*sl:(i1*sl+cp_len), ti]
        x_diff[0] += (ca.cos(t0)*d0[0] - ca.sin(t0)*d0[1] 
                    - (ca.cos(t1)*d1[0] - ca.sin(t1)*d1[1]))
        x_diff[1] += (ca.sin(t0)*d0[0] + ca.cos(t0)*d0[1] 
                    - (ca.sin(t1)*d1[0] + ca.cos(t1)*d1[1]))
        x_diff *= xy_param
        if cp_len > 2:
            # wrap angle diff in tan(0.5x)
            x_diff[2] = ca.tan(0.5*((t0 - t1) - (d0[2] - d1[2])))
            x_diff[2] *= t_param
        cost += ca.mtimes(x_diff.T, x_diff)
        return cost

    def align_cp_cost(self, cp, prev_cp):
        '''
        cp key: (i0, i1)
        cp item: 2-by-2: [[dx0, dy0], [dx1, dy1]].T
        '''
        param = self.param
        cost = 0
        for ti in range(param.hmpc+1):
            for key in cp:
                curr = {key: cp[key]}
                if ti < param.hmpc:
                    cost += self.add_cp_cost(curr, ti, 
                                        param.cost_Q["s_cp_xy"], 
                                        param.cost_Q["s_cp_t"])
                else:
                    cost += self.add_cp_cost(curr, ti,
                                        param.cost_Q["cp_xy"],
                                        param.cost_Q["cp_t"])
            for key in prev_cp:
                curr = {key: prev_cp[key]}
                if ti < param.hmpc:
                    cost += self.add_cp_cost(curr, ti, 
                                        param.cost_Q["s_prev_xy"], 
                                        param.cost_Q["s_prev_t"])
                else:
                    cost += self.add_cp_cost(curr, ti,
                                        param.cost_Q["prev_xy"],
                                        param.cost_Q["prev_t"])
        return cost
    
    def init_cost(self, prev_x, zero_list=[]):
        param = self.param
        sl = self.state_len
        x = self.x
        x_curr = self.fit_prev_x2opt(prev_x)
            
        cost = 0
        return cost

    def goal_cost(self, goal):
        """
        goal: list of len 3 [x, y, theta]
        """
        x, u = self.x, self.u
        assert(self.state_len == 5 or self.state_len == 3)
        if self.state_len == 5:
            goal_vec = np.hstack([goal+[0, 0] for i in range(self.N)])
            x_diff = x[:, -1] - goal_vec
            cost = ca.mtimes(x_diff.T, x_diff)
        elif self.state_len == 3:
            goal_vec = np.hstack([goal for i in range(self.N)])
            x_diff = x[:, -1] - goal_vec
            cost = ca.mtimes(x_diff.T, x_diff)
        return cost

    def stage_cost(self):
        u = self.u
        param = self.param
        cost = 0
        for ti in range(1, param.hmpc):
            cost += ca.mtimes(u[:, ti].T, u[:, ti]) * param.cost_Q["Q_u"]
        return cost

    def smooth_cost(self, prev_u):
        u = self.u
        param = self.param
        diff_u = u[:, 0] - prev_u
        diff_u[0::2] *= param.cost_Q["smooth_v"]
        diff_u[1::2] *= param.cost_Q["smooth_w"]
        cost = ca.mtimes(diff_u.T, diff_u)
        for ti in range(1, param.hmpc):
            diff_u = u[:, ti] - u[:, ti-1]
            diff_u[0::2] *= param.cost_Q["smooth_v"]
            diff_u[1::2] *= param.cost_Q["smooth_w"]
            cost += ca.mtimes(diff_u.T, diff_u)
        return cost

    def gdu_cost(self, gdu):
        u = self.u
        cost = 0
        gdu = ca.MX(gdu.tolist())
        for ti in range(self.param.hmpc):
            diff = u[:, ti] - gdu
            cost += ca.mtimes(diff.T, diff)
        return cost

    def optimize_cp(self, cost):
        opt = self.opt
        sl = self.state_len
        opt.minimize(cost)
        opt.solver("ipopt", self.ipopt_param)
        try:
            ans = opt.solve()
            return ans.value(self.u[:, 0]), ans.value(cost)
        except Exception as e:
            print(e)
            #  print("Solver value: ", opt.debug.value)
            opt.debug.show_infeasibilities()
        return np.zeros(2*self.N), None

    def p_control(self, u, pv=1.0, pw=1.0):
        if u is None:
            return u
        vmax = self.param.vmax
        wmax = self.param.wmax
        u[0::2] *= pv
        u[1::2] *= pw
        u[0::2] = np.clip(u[0::2], -vmax, vmax)
        u[1::2] = np.clip(u[1::2], -wmax, wmax)
        return u