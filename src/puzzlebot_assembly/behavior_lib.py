import numpy as np
import warnings
from puzzlebot_assembly.utils import *
from puzzlebot_assembly.planner import Planner
from pathos.multiprocessing import ProcessingPool


class BehaviorLib:
    def __init__(self, N, controller, bhav_list=[], eth=1e-3, 
                                    robot_param={}):
        self.N = N
        self.ctl = controller
        self.bhav_list = bhav_list
        self.bhav_id = 0
        self.eth = eth
        self.robot_param = robot_param
        self.align_pool_var = {}
        self.anchor_cps = {}
        self.ctl_param = controller.param
        self.fail_count = 0
        self.time_list = []
        self.status_list = []
        self.location_list = []
        self.file_time = None #open('times.txt', 'a')
        self.file_status = None #open('anchor_status.txt', 'a')

    def add_bhav(self, bhav):
        self.bhav_list.append(bhav)

    def wiggle(self, x, u, t=0, vbias=0, rid=[]):
        N = self.N
        L = self.robot_param.L
        u = np.zeros(2*N)
        rid = np.array(rid)
        if len(rid) == 0: return u 

        x_diff = np.diff(x.reshape([N, 3]).T[0:2, :], axis=1)
        #  print("x_diff:", x_diff)
        if np.all(np.linalg.norm(x_diff, axis=0) > 1.3*L):
            u[rid*2] = 0.2
            return u
        u[rid*2] = vbias
        u[rid*2+1] = 5 * np.sign(np.sin(t*15))
        return u

    def get_body_index(self, cp_d):
        L = self.robot_param.L
        body_idx = None
        conn_type = None

        # determine if the connection pair is left-right or front-back
        cp_sum = np.sum(cp_d, axis=1)
        if cp_sum[0] == 0:
            # cp is front-back
            body_idx = np.where(cp_d[0, :] == L/2)[0] 
            assert(len(body_idx) > 0)
            body_idx = body_idx[0]
            conn_type = "anchor"
        elif cp_sum[1] == 0:
            # cp is left-right
            anchor_idx = np.where(cp_d[0, :] == L/2)[0]
            assert(len(anchor_idx) > 0)
            body_idx = 1 - anchor_idx[0]
            conn_type = "knob"
        else:
            raise ValueError("Error with Connection Pair Values!")
        return body_idx, conn_type

    def init_anchor_param(self, cp_d):
        '''
        cp_d: 3-by-2 np array for the coupling pair position in local frame
        '''
        L = self.robot_param.L
        bl = self.robot_param.anchor_base_L
        al = self.robot_param.anchor_L
        eth = self.eth

        body_idx, conn_type = self.get_body_index(cp_d)

        # anchor status:
        # decoupled, head_aligned, head_insert
        anchor_param = {}
        anchor_param['status'] = "decoupled"
        # execute status: wait, now, done
        anchor_param['execute'] = "wait"
        anchor_param['anchor_index'] = 1 - body_idx
        anchor_param['type'] = conn_type

        if conn_type == "anchor":
            anchor_param['align_cp'] = np.array([[L/2, 0, 0], 
                                        [-L/2 - al -eth, 0, 0]]).T
            anchor_param['insert_cp'] = np.array([[L/2, 0, 0], [-L/2, 0, 0]]).T
        elif conn_type == "knob":
            anchor_param['align_cp'] = np.array([[-0.02, -L/2-0.02, -0.3],
                                        [L/2, L/2, 0.3]]).T
            anchor_param['insert_cp'] = np.array([[-0.001, -L/2, 0],
                                        [L/2, L/2, 0]]).T
        else:
            raise ValueError("Connection Type Error!")

        if body_idx == 1:
            anchor_param['align_cp'] = np.fliplr(anchor_param['align_cp'])
            anchor_param['insert_cp'] = np.fliplr(anchor_param['insert_cp'])
        return anchor_param

    def update_cp_anchor(self, cp_list):  #equivalent to function augmentpairs in paper
        anchor_cps = {}

        for cp_ids in cp_list:
            cp_d = cp_list[cp_ids]
            anchor_param = self.init_anchor_param(cp_d)
            anchor_cps[cp_ids] = anchor_param
        self.anchor_cps = anchor_cps

    def get_current_dicts(self, x, anchor_cps, curr_dict, conn_dict):   #updatepairs in paper
        N, eth = self.N, self.eth
        robot_param = self.robot_param
        
        for ids in anchor_cps:
            anchor_param = anchor_cps[ids]
            status = anchor_param['status']
            if anchor_param['execute'] == "wait": continue
            # print("ids:", ids)
            # print(anchor_param['status'])

            # get index of robot aligning with anchor (and without)
            anchor_idx = anchor_param['anchor_index']
            anchor_id = ids[anchor_idx]
            body_idx = 1 - anchor_idx
            body_id = ids[body_idx]
            body_x = x[3*ids[body_idx]:3*(ids[body_idx]+1)]

            if status == "decoupled":
                curr_dict[ids] = anchor_param['align_cp']

                # check if the anchor head is already aligned
                anchor_pt = body2world(x[3*anchor_id:3*(anchor_id+1)],
                            anchor_param['align_cp'][:, anchor_idx, np.newaxis])
                if (anchor_param['type'] == "anchor" and 
                    is_inside_robot(anchor_pt, body_x, robot_param.L,
                                    margin=0)):
                    anchor_param['status'] = "head_aligned"
                    continue
                if anchor_param['type'] == "knob":
                    body_pt = body2world(x[3*body_id:3*(body_id+1)],
                            anchor_param['align_cp'][:, body_idx, np.newaxis])
                    if np.linalg.norm(body_pt - anchor_pt) < 2*eth:
                        anchor_param['status'] = "head_aligned"
                        continue
            elif status == "head_aligned":
                curr_dict[ids] = anchor_param['insert_cp']

                # make sure the anchor head are indeed aligned
                anchor_pt = body2world(x[3*anchor_id:3*(anchor_id+1)],
                            anchor_param['align_cp'][:, anchor_idx, np.newaxis])
                #  print("anchor_pt:", anchor_pt)
                if anchor_param['type'] == "anchor":
                    if not is_inside_robot(anchor_pt, body_x, robot_param.L,
                                        margin=eth):
                        anchor_param['status'] = "decoupled"
                        continue

                # check if the anchor is inserted
                anchor_pt = body2world(x[3*anchor_id:3*(anchor_id+1)],
                            anchor_param['insert_cp'][:, anchor_idx, np.newaxis])
                if anchor_param['type'] == "anchor":
                    body_pt = body2world(x[3*body_id:3*(body_id+1)],
                            anchor_param['insert_cp'][:, body_idx, np.newaxis])
                    if is_inside_robot(anchor_pt, body_x, robot_param.L, #((np.linalg.norm(body_pt - anchor_pt) < eth) or is_inside_robot(anchor_pt, body_x, robot_param.L,
                                        margin=eth):
                        anchor_param['status'] = "head_insert"
                        continue
                elif anchor_param['type'] == "knob":
                    body_pt = body2world(x[3*body_id:3*(body_id+1)],
                            anchor_param['insert_cp'][:, body_idx, np.newaxis])
                    is_angle_align = get_heading_err(x, {ids: anchor_param['insert_cp']})
                    if is_angle_align and (np.linalg.norm(body_pt - anchor_pt) < 0.5*eth):
                        anchor_param['status'] = "head_insert"
                        continue

            elif status == "head_insert":
                if ids not in conn_dict:
                    conn_dict[ids] = anchor_param['align_cp']
                    curr_dict.pop(ids)
                    #  print("normal head insert")
                    continue
                
                # the head is already inserted, check if still inserted
                anchor_pt = body2world(x[3*anchor_id:3*(anchor_id+1)],
                             anchor_param['insert_cp'][:, anchor_idx, np.newaxis])
                if anchor_param['type'] == "anchor":
                    body_pt = body2world(x[3*body_id:3*(body_id+1)],
                             anchor_param['insert_cp'][:, body_idx, np.newaxis])
                    if ((np.linalg.norm(body_pt - anchor_pt) < eth) or is_inside_robot(anchor_pt, body_x, robot_param.L,
                                            margin=0.3*eth)):
                            # the head is still inserted, do nothing
                        print("head insert check pass")
                        continue
                        # head is not inserted, go back to insert
                    warnings.warn('Anchor disconnected')
                    anchor_param['status'] = "head_insert"
                    conn_dict.pop(ids)
                    curr_dict[ids] = anchor_param['insert_cp']
                elif anchor_param['type'] == "knob":
                    raise NotImplementedError("Knob type in head_insert")

        return curr_dict, conn_dict

    def get_zero_ids(self, conn_dict, busy, pilot_ids=[]):
        mask = np.ones(self.N, dtype=bool)
        conn_ids = list(conn_dict.keys())
        mask_conn_ids = []
        for cids in conn_ids:
            if busy[cids[0]] or busy[cids[1]]:
                mask_conn_ids += list(cids)
        mask[mask_conn_ids] = False
        mask[busy] = False
        mask[pilot_ids] = True
        return np.nonzero(mask)[0]

    def init_anchor_pool(self, x, pilot_ids=[]):
        N = self.N
        self.align_pool_var['planner'] = Planner(self.N)
        self.align_pool_var['busy'] = np.zeros(N, dtype=bool)
        p = self.align_pool_var['planner']
        self.align_pool_var['pair_dict'] = p.generate_pair_pool(
                                            [N], x.reshape([N, 3]).T[0:2, :],
                                            pilot_ids=pilot_ids)
        self.update_cp_anchor(self.align_pool_var['pair_dict'])
        self.align_pool_var['curr_dict'] = {}
        self.align_pool_var['conn_dict'] = {}
        self.align_pool_var['remain_dict'] = dict.fromkeys(self.align_pool_var['pair_dict'])
        self.align_pool_var['seg_dict'] = {i:[i] for i in range(N)}

    def align_anchor_pool(self, x, prev_u, pilot_ids=[]):
        N = self.N
        if len(self.align_pool_var) == 0: 
            self.init_anchor_pool(x, pilot_ids=pilot_ids)
        planner = self.align_pool_var['planner']
        curr_dict = self.align_pool_var['curr_dict']
        conn_dict = self.align_pool_var['conn_dict']
        remain_dict = self.align_pool_var['remain_dict']
        robot_busy = self.align_pool_var['busy']
        seg_dict = self.align_pool_var['seg_dict']
        anchor_cps = self.anchor_cps

        # decide which pairs to execute
        for k in remain_dict:
            if np.any(robot_busy[list(k)]):
                continue
            if k in conn_dict:
                continue
            robot_busy[list(k)] = True
            robot_busy[seg_dict[k[0]] + seg_dict[k[1]]] = True
            anchor_cps[k]['execute'] = "now"

        # update contact pair if needed
        curr_dict, conn_dict = self.get_current_dicts(x, anchor_cps,
                                    curr_dict, conn_dict)
        #  print("curr_dict", curr_dict)
        #  print("conn_dict", conn_dict)
        zero_list = self.get_zero_ids(conn_dict, robot_busy, 
                                pilot_ids=pilot_ids)
        #  print("zero_list:", zero_list)
        
        u = self.align_cp(x, prev_u, curr_dict, prev_cp=conn_dict,
                        zero_list=zero_list)
        
        u[2*zero_list] = 0
        u[2*zero_list+1] = 0

        if len(curr_dict) > 0 and len(pilot_ids) > 0:
            curr_key = next(iter(curr_dict))
            if(pilot_ids[0] in curr_key and anchor_cps[curr_key]['status'] == "head_aligned"):
                #  print("i hate hard code")
                u[0::2] = 0.1
                u[2*pilot_ids] = -0.1

        # current pairs are done executing
        for idx, cp in conn_dict.items():
            if anchor_cps[idx]['execute'] != "now":
                continue
            remain_dict.pop(idx)
            anchor_cps[idx]['execute'] = "done"
            robot_busy[list(idx)] = 0
            robot_busy[seg_dict[idx[0]]] = 0
            robot_busy[seg_dict[idx[1]]] = 0
            seg_dict[idx[0]].append(idx[1])
            seg_dict[idx[1]].append(idx[0])
        
        #  print(self.align_pool_var)
        if len(remain_dict) == 0:
            print("All pairs aligned.")
            return None
        # print("u:", u)
        return u
            
    def init_align_pool(self, x):  #try writing to text files here
        N = self.N
        self.align_pool_var['planner'] = Planner(self.N)
        self.align_pool_var['busy'] = np.zeros(N, dtype=bool)
        p = self.align_pool_var['planner']
        self.align_pool_var['pair_dict'] = p.generate_pair_pool(
                                            [N], x.reshape([N, 3]).T[0:2, :])
        self.align_pool_var['curr_dict'] = {}
        self.align_pool_var['conn_dict'] = {}
        self.align_pool_var['remain_dict'] = self.align_pool_var['pair_dict'].copy()
        self.align_pool_var['seg_dict'] = {i:[i] for i in range(N)}

    def align_cp_pool(self, x, u):
        N = self.N
        if len(self.align_pool_var) == 0: 
            self.init_align_pool(x) 
        planner = self.align_pool_var['planner']
        curr_dict = self.align_pool_var['curr_dict']
        conn_dict = self.align_pool_var['conn_dict']
        remain_dict = self.align_pool_var['remain_dict']
        robot_busy = self.align_pool_var['busy']
        seg_dict = self.align_pool_var['seg_dict']

        # decide which pairs to execute
        for k in remain_dict.keys():
            if np.any(robot_busy[list(k)]):
                continue
            if k in conn_dict:
                continue
            cp = remain_dict[k]
            robot_busy[list(k)] = True
            robot_busy[seg_dict[k[0]] + seg_dict[k[1]]] = True
            curr_dict[k] = cp

        # update contact pair if needed
        curr_dict = planner.update_contact_with_ids(x, curr_dict)
        conn_dict = planner.update_contact_with_ids(x, conn_dict)

        u = self.align_cp(x, u, curr_dict, prev_cp=conn_dict)

        if u is not None:
            return u

        # current pairs are done executing
        for idx, cp in curr_dict.items():
            conn_dict[idx] = cp
            remain_dict.pop(idx)
            robot_busy[list(idx)] = 0
            robot_busy[seg_dict[idx[0]]] = 0
            robot_busy[seg_dict[idx[1]]] = 0
            seg_dict[idx[0]].append(idx[1])
            seg_dict[idx[1]].append(idx[0])
        self.align_pool_var['curr_dict'] = {}
        
        #  print(self.align_pool_var)
        if len(remain_dict) == 0:
            print("All pairs aligned.")
            return None
        return np.zeros(2*N)

    def align_cp(self, x, u, cp, prev_cp=[], zero_list=[]):
        # print("cp_dis:", get_cp_dis(x, cp))
        if not cp: 
            return np.zeros(2 * self.N)
        #  print("prev_cp_dis:", get_cp_dis(x, prev_cp))
        self.ctl.init_opt(x, u)                             #initial values
        self.ctl.add_dynamics_constr()                      #dynamics constraints
        self.ctl.add_vwlim_constraint()                     #v and w constraints
        self.ctl.add_align_poly_constr(prev_cp, self.robot_param.L)     #constraints to keep couplings intact

        cost = self.ctl.init_cost(x, zero_list=zero_list)
        cost = self.ctl.align_cp_cost(cp, prev_cp)
        #cost += self.ctl.stage_cost()
        stage_cost = self.ctl.stage_cost()
        #smooth_cost = self.ctl.smooth_cost(u)
        for n in stage_cost.keys():
            if n in cost.keys():
                cost[n] += stage_cost[n]
            else:
                cost[n] = stage_cost[n]
            #cost[n] += smooth_cost[n]

        from time import time
        start = time()

        u_vel, obj_value = self.ctl.optimize_cp(cost)

        end = time()
        self.time_list.append(end-start)
        anchor_cps = self.anchor_cps
        s = []
        l = []
        robot_param = self.robot_param
        err = 0.005
        for ids in anchor_cps:
            anchor_param = anchor_cps[ids]
            s.append(anchor_param['status'])

            if anchor_param['status'] == 'head_insert':
                anchor_idx = anchor_param['anchor_index']
                anchor_id = ids[anchor_idx]
                body_idx = 1 - anchor_idx
                body_id = ids[body_idx]
                body_x = x[3*ids[body_idx]:3*(ids[body_idx]+1)]
                body_pt = body2world(x[3*body_id:3*(body_id+1)],
                                anchor_param['insert_cp'][:, body_idx, np.newaxis])
                anchor_pt = body2world(x[3*anchor_id:3*(anchor_id+1)],
                                anchor_param['insert_cp'][:, anchor_idx, np.newaxis])
                if np.linalg.norm(body_pt - anchor_pt) < err:
                    l.append('yes')
                else: l.append('no')
            else:
                l.append('N/A')
        self.status_list.append(s)
        self.location_list.append(l)
        
        
        if obj_value is None:
            self.fail_count += 1
        else:
            self.fail_count = 0
        if obj_value is None and self.fail_count > 3:
            # print("Recompute the optimization.")
            self.ctl.init_opt(x, u)
            self.ctl.add_dynamics_constr()
            self.ctl.add_vwlim_constraint()
            cost = 0
            cost += self.ctl.align_cp_cost(cp, prev_cp)
            #  cost += self.ctl.align_cp_cost(prev_cp, prev_cp=[])
            u_vel, obj_value = self.ctl.optimize_cp(cost)
            if obj_value is None:
                u_vel = np.zeros(2 * self.N)
                u_vel[0::2] = 0.1
            return u_vel

        return u_vel

    def go_du(self, x, u, gdu, prev_cp=[], end_x=-0.077, slow_x=-0.088):
        #  if np.min(x[0::3]) > end_x:
            #  return np.zeros(2 * self.N)
        #  if np.max(x[-1::3]) < slow_x:
            #  return gdu*0.5
        #  self.ctl.init_opt(x, u, prev_cp)
        #  self.ctl.add_dynamics_constr()
        #  self.ctl.add_vwlim_constraint()
        #  self.ctl.add_align_poly_constr(prev_cp, self.robot_param.L)
        #  cost = self.ctl.gdu_cost(gdu)
        #  u_opt, obj_value = self.ctl.optimize_cp(cost)
        #  print("u_opt:", u_opt)
        return gdu
        #  u_opt = np.zeros(2 * self.N)
        #  u_opt[0::2] = 0.15
        return u_opt

    def current(self):
        if self.bhav_id >= len(self.bhav_list):
            return None
        return self.bhav_list[self.bhav_id]
    
    def go_to_goal(self, x, u, goal=[0.0, 0.0, 0], prev_cp=[]):#mpc, note that connection pairs is empty
        goal_vec = np.hstack([goal for i in range(self.N)])    #one copy of goal per robot
        if np.linalg.norm((x - goal_vec)[0:2]) < self.eth*50:   #if robots are close enough to their goals
            return np.zeros(2*self.N)
            #return None
        self.ctl.init_opt(x, u)                                #setting initial constraints
        self.ctl.add_dynamics_constr()                         #adding dynamics constraints
        self.ctl.add_vwlim_constraint()                        #more constraints
        #self.ctl.add_align_poly_constr(prev_cp, self.robot_param.L)   #i think this has to do with anchor position being within triangle of body
        cost = 0
        cost += self.ctl.goal_cost(goal)                       #calculates cost
        cost += self.ctl.stage_cost()
        # print("cost: " + str(cost))

        u_vel, obj_value = self.ctl.optimize_cp(cost)          #get u
        # print("obj value: " + str(obj_value))
        # print("u_vel: " + str(u_vel))
        return u_vel

    def p_go_to_goal(self, x):    #p-controller
        goal_pos = np.hstack([0.0, 0.0, 0])   #currently only for one robot, or multiple robots with same goal
        
        #distance to the goal
        dx = goal_pos[0] - x[0]
        dy = goal_pos[1] - x[1]
        #angle to the goal
        dtheta = math.atan2(dy, dx)
        angle = dtheta - x[2]

        #if (dx * dx) <= 0.5 and (dy * dy) <= 0.5:   #if close enough to goal
        if np.linalg.norm((x - goal_pos)[0:2]) < self.eth*50:
            return None

        #desired translational velocity
        v = 0.03

        #normalize the angular error to be between -pi and pi
        angle = np.arctan2(np.sin(angle),np.cos(angle))

        #calculate the desired angular velocity
        w = (0.4 * angle)

        # Return the desired velocity and angular velocity
        return np.vstack([v, w]).T.flatten()
    
    def nothing(self, x, u=[], param=[], pilot_ids=[]):
        return np.zeros(2 * self.N)
