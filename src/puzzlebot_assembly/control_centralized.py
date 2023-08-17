import casadi as ca
import numpy as np
import math

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
        x_sym = ca.SX.sym('x', 5*N)
        u_sym = ca.SX.sym('u', 2*N)

        theta = x_sym[2::5]
        vs = x_sym[3::5]
        x_dot = ca.SX.zeros(5*N)   #5N x 1
        x_dot[0::5] = vs * ca.cos(theta)
        x_dot[1::5] = vs * ca.sin(theta)
        x_dot[2::5] = x_sym[4::5]
        x_dot[3::5] = u_sym[0::2]
        x_dot[4::5] = u_sym[1::2]

        return ca.Function("fk_opt", [x_sym, u_sym], [x_sym + (x_dot * dt)])

    def fk_force_opt(self, cp):
        N, dt, sl, M = self.N, self.dt, self.state_len, self.M
        x_sym = ca.SX.sym('x', sl*N)
        x_curr[:, 0:3] = prev_x.reshape([self.N, 3])
        x_curr = x_curr.flatten()
        u_sym = ca.SX.sym('u', sl*N)

        # regular differential drive dynamics without pulling forces
        theta = x_sym[2::sl]
        vs = x_sym[3::sl]
        x_dot = ca.SX.zeros(sl*N)
        x_dot[0::sl] = vs * ca.cos(theta)
        x_dot[1::sl] = vs * ca.sin(theta)
        x_dot[2::sl] = x_sym[4::sl]
        x_dot[3::sl] = u_sym[0::2]
        x_dot[4::sl] = u_sym[1::2]

        for cp_ids in prev_cp:
            cp_d = prev_cp[cp_ids]

            # find the one robot having contact point on body not anchor
            body_idx = np.where(cp_d[0, :] == L/2)[0] 
            assert(len(body_idx) > 0)
            body_idx = body_idx[0]

            b_idx = cp_ids[body_idx]        # robot contact on body
            a_idx = cp_ids[1 - body_idx]    # robot contact on anchor

            x_dot[sl*(b_idx)+4] += x_sym[sl*(b_idx)+5] / M

        return ca.Function("fk_force_opt", [x_sym, u_sym], [x_sym + (x_dot * dt)])

    def dd_fx_opt(self, theta, N):
        F = ca.SX.zeros(3*N, 2*N)
        F[0::3, 0::2] = ca.diag(ca.cos(theta))
        F[1::3, 0::2] = ca.diag(ca.sin(theta))
        F[2::3, 1::2] = ca.SX.eye(N)
        return F

    def fk_rk4_opt(self, N, dt):
        x_sym = ca.SX.sym('x', 3*N)
        u_sym = ca.SX.sym('u', 2*N)
        
        k1 = ca.mtimes(dd_fx_opt(x_sym[2::3], N), u_sym)
        k2 = ca.mtimes(dd_fx_opt(x_sym[2::3] + k1[2::3]*dt/2, N), u_sym)
        k3 = ca.mtimes(dd_fx_opt(x_sym[2::3] + k2[2::3]*dt/2, N), u_sym)
        k4 = ca.mtimes(dd_fx_opt(x_sym[2::3] + k3[2::3]*dt, N), u_sym)
        x_dot = (k1 + 2*k2 + 2*k3 + k4)/6
        return ca.Function("fk_rk4_opt", [x_sym, u_sym], [x_sym + x_dot*dt])

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
        self.state_len = 5
        self.ca_int = CasadiInterface(N, dt, self.state_len, M=0.1)
        self.fk_opt = self.ca_int.fk_opt(N, dt)
        self.get_local_pt = self.ca_int.get_local_pt()
        #  self.fk_opt = fk_rk4_opt(N, dt)
        #  self.fk_opt = fk_exact_opt(N, dt)
        self.ipopt_param = {"verbose": False, 
                            "ipopt.print_level": 0,
                            "print_time": 0,
                            'ipopt.sb': 'yes',
                            }
        self.opt = None
        self.x = None # 5 states [x, y, theta, v, w, ]
        self.u = None # 2 controls [uv, uw]

        # for debug
        self.prev_x = None
    
    def fit_prev_x2opt(self, prev_x):
        x_curr = np.zeros([self.N, self.state_len])
        x_curr[:, 0:3] = prev_x.reshape([self.N, 3])
        x_curr = x_curr.flatten()
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

    def init_opt(self, prev_x, prev_u, prev_cp=[]):
        N = self.N
        param = self.param
        opt = ca.Opti()
        sl = self.state_len
        x = opt.variable(sl*N, param.hmpc + 1)
        u = opt.variable(2*N, param.hmpc)

        #  self.fk_opt = self.ca_int.fk_force_opt(N, self.dt, sl, prev_cp)

        # for debug
        self.prev_x = prev_x

        # initial state constraints
        opt.subject_to(x[0::sl, 0] == prev_x[0::3])
        opt.subject_to(x[1::sl, 0] == prev_x[1::3])
        opt.subject_to(x[2::sl, 0] == prev_x[2::3])
        opt.subject_to(x[3::sl, 0] == prev_u[0::2])
        opt.subject_to(x[4::sl, 0] == prev_u[1::2])

        # v, w constraints - setting all the vs and ws to be between -param.max and +param.max
        opt.subject_to(opt.bounded(-param.vmax, 
                            ca.vec(x[3::sl, :]), param.vmax))
        opt.subject_to(opt.bounded(-param.wmax, 
                            ca.vec(x[4::sl, :]), param.wmax))

        # uv, uw constraints
        opt.subject_to(opt.bounded(-param.uvmax, 
                            ca.vec(u[0::2, :]), param.uvmax))
        opt.subject_to(opt.bounded(-param.uwmax, 
                            ca.vec(u[1::2, :]), param.uwmax))

        #try warm start
        x_curr = self.fit_prev_x2opt(prev_x)
        for ti in range(param.hmpc + 1):
            opt.set_initial(x[:, ti], x_curr)
        opt.set_initial(u, 0)

        self.opt = opt
        self.x = x
        self.u = u

    def add_dynamics_constr(self):
        opt = self.opt
        x, u = self.x, self.u
        # dynamics constraints
        for ti in range(self.param.hmpc):
            opt.subject_to(x[:, ti+1] == self.fk_opt(x[:, ti], u[:, ti]))  #5N x 1 and 2N x 1

    def add_vwlim_constraint(self):
        opt = self.opt
        param = self.param
        sl = self.state_len
        x, u = self.x, self.u

        # try butterfly shape constraints
        for ti in range(self.param.hcst):
            opt.subject_to(-1/param.vmax * ca.fabs(x[3::sl, ti+1]) + 
                        1/param.wmax * x[4::sl, ti+1] <= 0)


    def add_align_poly_constr(self, prev_cp, L):
        if len(prev_cp) == 0: return
        opt = self.opt
        param = self.param
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
                x_pt = get_local_pt(x[sl*anchor_id:(sl*anchor_id+3), ti], 
                                        ca.MX(cp_d[:, anchor_idx]))
                xr = get_local_pt(x[sl*body_id:(sl*body_id+3), ti],
                                        ca.MX([L/2, -L/2]))
                #  xrs = get_local_pt(x[sl*body_id:(sl*body_id+3), ti],
                                        #  ca.MX([0, -L/2]))
                xl = get_local_pt(x[sl*body_id:(sl*body_id+3), ti],
                                        ca.MX([L/2, L/2]))
                #  xls = get_local_pt(x[sl*body_id:(sl*body_id+3), ti],
                                        #  ca.MX([0, L/2]))

                #  opt.subject_to((x_pt[1] - xrs[1])*(xr[0] - xrs[0]) >= (xr[1] - xrs[1])*(x_pt[0] - xrs[0]))
                #  opt.subject_to((x_pt[1] - xls[1])*(xrs[0] - xls[0]) >= (xrs[1] - xls[1])*(x_pt[0] - xls[0]))
                #  opt.subject_to((x_pt[1] - xl[1])*(xls[0] - xl[0]) >= (xls[1] - xl[1])*(x_pt[0] - xl[0]))
                #  opt.subject_to((x_pt[1] - xr[1])*(xl[0] - xr[0]) >= (xl[1] - xr[1])*(x_pt[0] - xr[0]))

                xR = x[sl*body_id, ti]
                yR = x[sl*body_id+1, ti]

                opt.subject_to((x_pt[1] - yR)*(xr[0] - xR) >= (xr[1] - yR)*(x_pt[0] - xR))
                opt.subject_to((x_pt[1] - yR)*(xR - xl[0]) >= (yR - xl[1])*(x_pt[0] - xR))
                opt.subject_to((x_pt[1] - xr[1])*(xl[0] - xr[0]) >= (xl[1] - xr[1])*(x_pt[0] - xr[0]))

    def add_pull_force_constr(self, prev_cp, L):
        if len(prev_cp) == 0: return
        opt = self.opt
        param = self.param
     
    def add_cp_cost(self, cp, ti, xy_param, t_param):
        x, u = self.x, self.u
        assert(self.state_len == 5)
        cost = 0
        i0, i1 = next(iter(cp))
        d0 = cp[(i0, i1)][:, 0]  #dx0 and dy0
        d1 = cp[(i0, i1)][:, 1]  #dx1 and dy1
        t0 = x[i0*5+2, ti]
        t1 = x[i1*5+2, ti]
        cp_len = d0.shape[0]
        x_diff = x[i0*5:(i0*5+cp_len), -1] - x[i1*5:(i1*5+cp_len), -1]
        x_diff[0] += (ca.cos(t0)*d0[0] - ca.sin(t0)*d0[1] 
                    - (ca.cos(t1)*d1[0] - ca.sin(t1)*d1[1]))
        x_diff[1] += (ca.sin(t0)*d0[0] + ca.cos(t0)*d0[1] 
                    - (ca.sin(t1)*d1[0] + ca.cos(t1)*d1[1]))
        x_diff *= xy_param
        if cp_len > 2:
            # wrap angle diff in tan(0.5x)
            x_diff[2] = ca.tan(0.5*((t0 - t1) - (d0[2] - d1[2])))
            x_diff[2] *= t_param*4
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
                print('curr: ' + str(curr))
                print('cost: ' + str(cost))
        return cost
    
    def init_cost(self, prev_x, zero_list=[]):
        param = self.param
        sl = self.state_len
        x = self.x
        x_curr = self.fit_prev_x2opt(prev_x)
            
        cost = 0
        #  cost += ca.mtimes(ca.vec(x[3::sl, :]).T, 
                    #  ca.vec(x[3::sl, :])) * param.cost_Q["stay_vw"]
        #  cost += ca.mtimes(ca.vec(x[4::sl, :]).T, 
                    #  ca.vec(x[4::sl, :])) * param.cost_Q["stay_vw"]
        #  for ti in range(param.hmpc + 1):
            #  x_diff = x[:, ti] - x_curr
            #  cost += ca.mtimes(x_diff.T, x_diff) * param.cost_Q["stay_xyt"]

        # mask zeros for zero_list
        if len(zero_list) > 0:
            vs = ca.vec(x[sl*(zero_list)+3, :])
            ws = ca.vec(x[sl*(zero_list)+4, :])
            cost += ca.mtimes(vs.T, vs) * param.cost_Q["zero_xyt"]
            cost += ca.mtimes(ws.T, ws) * param.cost_Q["zero_xyt"]

        return cost

    def goal_cost(self, goal):
        """
        goal: list of len 3 [x, y, theta]
        """
        assert(self.state_len == 5)
        x, u = self.x, self.u
        sl = self.state_len
        goal_vec = np.hstack([goal+[0,0] for i in range(self.N)])   #so that it's includes v and w, also one copy of goal per robot
        cost = 0
        # print(x)

        for i in range(self.N):
            dx = goal_vec[sl*i+0] - x[:, -1][sl*i+0]
            dy = goal_vec[sl*i+1] - x[:, -1][sl*i+1]
            angle = ca.atan2(dy, dx)
            cost += 1.0 * ca.tan(0.5 * (x[:, -1][sl*i+2] - angle)) * ca.tan(0.5 * (x[:, -1][sl*i+2] - angle))
            cost += dy * dy
            cost += dx * dx

        return cost

    def goal_cost_one_robot(self, goal):
        """
        goal: list of len 3 [x, y, theta]
        """
        assert(self.state_len == 5)
        x, u = self.x, self.u
        goal_vec = np.hstack([goal+[0,0] for i in range(self.N)])   #so that it's same size as x (includes v, w)
        cost = 0
        # print(x)

        dx = goal[0] - x[:, -1][0]
        dy = goal[1] - x[:, -1][1]
        angle = ca.atan2(dy, dx)
        cost += 1.0 * ca.tan(0.5 * (x[:, -1][2] - angle)) * ca.tan(0.5 * (x[:, -1][2] - angle))
        cost += dy * dy
        cost += dx * dx

        """for i in range(max(0, self.param.hmpc-2), self.param.hmpc+1):
            x_diff = x[:, i] - goal_vec    #get the last few x vectors and subtract with goal vector
            cost += ca.mtimes(x_diff.T, x_diff)  #cost is basically euclidean distance squared"""
        #x_diff = x[:, -1] - goal_vec
        #cost += ca.mtimes(x_diff.T, x_diff)
        return cost
        
        """x_diff = self.x - goal
        cost = ca.mtimes(x_diff.T, x_diff)
        return cost"""

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
            #  return ans.value(self.u[:, 0]), ans.value(cost)
            uv = ans.value(self.x[3::sl, 1])
            uw = ans.value(self.x[4::sl, 1])
            return np.vstack([uv, uw]).T.flatten(), ans.value(cost)
        except Exception as e:
            print(e)
            #  print("Solver value: ", opt.debug.value)
            #  opt.debug.show_infeasibilities()

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

    
    def constraint(self, x, u):
        # Implement your system constraints using CasADi operations
        # Given the current state x and control input u, return the constraint values

        # Example: Velocity constraints
        v_max = 1.0  # Maximum velocity
        u_min = -v_max
        u_max = v_max
        return ca.vertcat(u[0] - u_min, u_max - u[0], u[1] - u_min, u_max - u[1])

    def helper(self, x0, goal=[0.0, 0.3, 0], horizon=5, dt=0.1):  #calculates cost and does optimization
        # Define dimensions of the state and control input
        state_dim = x0.shape[0]
        control_dim = 2  # Assuming 2-dimensional control inputs (e.g., velocity)

        # Define system dynamics
        x = ca.MX.sym('x', state_dim)
        u = ca.MX.sym('u', control_dim)

        # Define the optimization problem for MPC using CasADi
        X = x
        U = u

        cost_sum = 0.0
        Xk = x0
        g = []
        g_bounds = []

        for k in range(horizon):
            cost_sum += ca.norm_2(Xk - goal)
            Xk = Xk + U[k * control_dim: (k + 1) * control_dim] * dt

            # Apply system constraints
            constraints = self.constraint(Xk, U[k * control_dim: (k + 1) * control_dim])
            g.extend(constraints)
            g_bounds.extend([0.0] * constraints.shape[0])

            # Concatenate states and controls
            X = ca.vertcat(X, Xk)
            U = ca.vertcat(U, U[k * control_dim: (k + 1) * control_dim])

        # Define the objective and constraints
        objective = cost_sum
        constraints = ca.vertcat(*g)

        # Create the NLP problem
        nlp = {'x': U, 'f': objective, 'g': constraints}
        opts = {'ipopt.print_level': 0, 'print_time': 0, 'ipopt.tol': 1e-6}

        # Solve the optimization problem using CasADi's minimize function
        u_init = np.zeros(horizon * control_dim)
        res = ca.minimize(objective, constraints, U, g_bounds, opts)

        # Extract the optimal control input
        u_opt = res['x'][:control_dim]

        return u_opt
