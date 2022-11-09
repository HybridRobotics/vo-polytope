import numpy as np
from math import sin, cos, atan2, asin, pi, inf, sqrt
from ir_sim.util import collision_polygon_polygon
from time import time


class reciprocal_vel_obs_polygon:

    def __init__(self, neighbor_region=5, v_max=1.0, acceler=0.5, **kwargs):
        """ Polytopic-shaped VO """
        # vxmax & vymax are max of omni vel
        self.vxmax = v_max
        self.vymax = v_max
        self.acceler = acceler
        self.nr = neighbor_region

    def cal_polygon_vel(self, agent_state, agent_vertex, nei_state_list=None, nei_vertex_list=None,
                        obs_poly_list=None, obs_vertex_list=None, obs_cir_list=None, mode='rvo'):
        """ Calculate a suitable vel for polygon agent based on VO_p """
        if nei_state_list is None:
            nei_state_list = []
        if nei_vertex_list is None:
            nei_vertex_list = []

        if obs_poly_list is None:
            obs_poly_list = []
        if obs_vertex_list is None:
            obs_vertex_list = []

        if obs_cir_list is None:
            obs_cir_list = []

        # get the state in neighbor_region scope
        agent_state, ns_list, nv_list, op_list, ov_list, oc_list = self.preprocess(agent_state, nei_state_list,
                                                                                   nei_vertex_list, obs_poly_list,
                                                                                   obs_vertex_list, obs_cir_list)

        # configure the vo or rvo or hrvo
        vo_list = self.config_polygon_vo(agent_state, agent_vertex, ns_list, nv_list, op_list, ov_list, oc_list, mode)
        vo_outside, vo_inside = self.vel_candidate(agent_state, vo_list)
        rvo_vel = self.vel_select(agent_state, vo_outside, vo_inside, ns_list, op_list, oc_list, mode)

        return rvo_vel

    def cal_polygon_vel_cir(self, agent_state, nei_state_list=None, obs_poly_list=None, obs_cir_list=None, mode='rvo'):
        """ Calculate a suitable vel for polygon agent based on VO_c """
        if nei_state_list is None:
            nei_state_list = []
        if obs_poly_list is None:
            obs_poly_list = []
        if obs_cir_list is None:
            obs_cir_list = []

        # get the state in neighbor_region scope
        agent_state, ns_list, nv_list, op_list, ov_list, oc_list = self.preprocess(agent_state, nei_state_list,
                                                                                   [], obs_poly_list,
                                                                                   [], obs_cir_list)

        # configure the vo or rvo or hrvo
        vo_list = self.config_circle_vo(agent_state, ns_list, op_list, oc_list, mode)
        vo_outside, vo_inside = self.vel_candidate(agent_state, vo_list)
        rvo_vel = self.vel_select(agent_state, vo_outside, vo_inside, ns_list, op_list, oc_list, mode)

        return rvo_vel

    @staticmethod
    def extract_data(sub_list, state_list, vertex_list):
        """ Extract the vertexes used in vo, according sub_list and state_list to extract the data in vertex_list """
        sub_vertex_list = []
        if len(sub_list) == 0 or len(vertex_list) == 0:
            return sub_vertex_list

        index = 0
        for i in range(len(state_list)):
            if index < len(sub_list):
                if (state_list[i] == sub_list[index]).all():
                    sub_vertex_list.append(vertex_list[i])
                    index = index + 1

        return sub_vertex_list

    def preprocess(self, agent_state, nei_state_list, nei_vertex_list, obs_poly_list, obs_vertex_list, obs_cir_list):
        """ Get the agent state and the other information within neighbor_region """
        # components in the region
        agent_state = np.squeeze(agent_state)

        # within the neighbor region
        ns_list = list(filter(lambda x: 0 < self.distance(agent_state, x) <= self.nr, nei_state_list))
        nv_list = reciprocal_vel_obs_polygon.extract_data(ns_list, nei_state_list, nei_vertex_list)

        op_list = list(filter(lambda y: 0 < self.distance(agent_state, y) <= self.nr, obs_poly_list))
        ov_list = reciprocal_vel_obs_polygon.extract_data(op_list, obs_poly_list, obs_vertex_list)

        oc_list = list(filter(lambda z: 0 < self.distance(agent_state, z) <= self.nr, obs_cir_list))

        return agent_state, ns_list, nv_list, op_list, ov_list, oc_list

    def config_polygon_vo(self, agent_state, agent_vertex, nei_state_list, nei_vertex_list, obs_poly_list,
                          obs_vertex_list, obs_cir_list, mode):
        """ Construct the combined velocity obstacle for polygon using polytopic-shaped VO """
        # mode: vo, rvo, hrvo

        # calculate the rvo_list for other agents
        vo_list1 = list(map(lambda r, r_vertex: self.config_vo_polytope(agent_state, agent_vertex, r, r_vertex, mode),
                            nei_state_list, nei_vertex_list))

        # calculate the vo_list for obs_polygon
        vo_list2 = list(map(lambda o, o_vertex: self.config_vo_polytope(agent_state, agent_vertex, o, o_vertex, 'vo'),
                            obs_poly_list, obs_vertex_list))

        # calculate the vo_list for obs_cir, in fact not use
        vo_list3 = list(map(lambda z: reciprocal_vel_obs_polygon.config_vo_circle(agent_state, z, 'vo'), obs_cir_list))

        # every vo_list in shape [[], [], [], []]
        return vo_list1 + vo_list2 + vo_list3

    def config_circle_vo(self, agent_state, nei_state_list, obs_poly_list, obs_cir_list, mode):
        """ Construct the combined velocity obstacle for polygon using circular-shaped VO """
        # mode: vo, rvo, hrvo

        # calculate the rvo_list for other agents
        vo_list1 = list(map(lambda r: self.config_vo_circle(agent_state, r, mode), nei_state_list))

        # calculate the vo_list for obs_polygon
        vo_list2 = list(map(lambda o: self.config_vo_circle(agent_state, o, 'vo'), obs_poly_list))

        # calculate the vo_list for obs_cir
        vo_list3 = list(map(lambda z: self.config_vo_circle(agent_state, z, 'vo'), obs_cir_list))

        # every vo_list in shape [[], [], [], []]
        return vo_list1 + vo_list2 + vo_list3

    @staticmethod
    def config_vo_circle(agent_state, obs_state, mode='rvo'):
        """ Construct VO for circular-shaped robot """

        # for agents is 'rvo' and for obs_cir is 'vo'
        x, y, theta, vx, vy, r = agent_state[0:6]
        mx, my, mtheta, mvx, mvy, mr = obs_state[0:6]

        if mvx == 0 and mvy == 0:  # for obstacle or static agent
            mode = 'vo'

        dis_mr = sqrt((my - y) ** 2 + (mx - x) ** 2)
        angle_mr = atan2(my - y, mx - x)  # y/x return for (-pi, pi)

        if dis_mr < r + mr:
            dis_mr = r + mr

        ratio = (r + mr) / dis_mr
        half_angle = asin(ratio)
        line_left_ori = reciprocal_vel_obs_polygon.wraptopi(angle_mr + half_angle)
        line_right_ori = reciprocal_vel_obs_polygon.wraptopi(angle_mr - half_angle)

        apex = []
        if mode == 'vo':
            apex = [mvx, mvy]

        elif mode == 'rvo':
            apex = [(vx + mvx) / 2, (vy + mvy) / 2]

        elif mode == 'hrvo':

            rvo_apex = [(vx + mvx) / 2, (vy + mvy) / 2]
            vo_apex = [mvx, mvy]

            cl_vector = [mx - x, my - y]

            cur_v = [vx - rvo_apex[0], vy - rvo_apex[1]]

            dis_rv = reciprocal_vel_obs_polygon.distance(rvo_apex, vo_apex)
            radians_rv = atan2(rvo_apex[1] - vo_apex[1], rvo_apex[0] - vo_apex[0])

            diff = line_left_ori - radians_rv

            temp = pi - 2 * half_angle

            if temp == 0:
                temp = temp + 0.01

            dis_diff = dis_rv * sin(diff) / sin(temp)

            if reciprocal_vel_obs_polygon.cross_product(cl_vector, cur_v) <= 0:
                apex = [rvo_apex[0] - dis_diff * cos(line_right_ori), rvo_apex[1] - dis_diff * sin(line_right_ori)]
            else:
                apex = [vo_apex[0] + dis_diff * cos(line_right_ori), vo_apex[1] + dis_diff * sin(line_right_ori)]

        # [vx, vy, vl, vr]
        return apex + [line_left_ori, line_right_ori]

    def config_vo_polytope(self, agent_state, agent_vertex, obs_state, obs_vertex, mode='rvo', **kwargs):
        """ Construct VO for polytopic robot """
        x, y, theta, vx, vy, r = agent_state[0:6]
        mx, my, mtheta, mvx, mvy, mr = obs_state[0:6]

        if mvx == 0 and mvy == 0:  # for obstacle
            mode = 'vo'

        agent_vertex_num = agent_vertex.shape[1]
        obs_vertex_num = obs_vertex.shape[1]
        line_left_ori = -np.pi  # big orientation
        line_right_ori = np.pi  # small orientation

        for i in range(agent_vertex_num):
            for j in range(obs_vertex_num):
                dy = obs_vertex[1, j] - agent_vertex[1, i]
                dx = obs_vertex[0, j] - agent_vertex[0, i]
                theta = atan2(dy, dx)
                if theta >= line_left_ori:
                    line_left_ori = theta
                if theta <= line_right_ori:
                    line_right_ori = theta

        # case (b) in paper
        if line_left_ori - line_right_ori > np.pi:
            line_left_ori = -np.pi
            line_right_ori = np.pi
            for i in range(agent_vertex_num):
                for j in range(obs_vertex_num):
                    dy = obs_vertex[1, j] - agent_vertex[1, i]
                    dx = obs_vertex[0, j] - agent_vertex[0, i]
                    theta = atan2(dy, dx)

                    if line_left_ori < theta < 0:
                        line_left_ori = theta
                    elif 0 < theta < line_right_ori:
                        line_right_ori = theta

        # to reduce computation
        if reciprocal_vel_obs_polygon.distance([x, y], [x, y]) < r + mr + 0.1:
            # for extended veretxes:
            if collision_polygon_polygon(agent_vertex, obs_vertex):
                angle_mr = atan2(my - y, mx - x)
                line_left_ori = reciprocal_vel_obs_polygon.wraptopi(angle_mr + pi / 2)
                line_right_ori = reciprocal_vel_obs_polygon.wraptopi(angle_mr - pi / 2)

        apex = None
        if mode == 'rvo':
            apex = [(vx + mvx) / 2, (vy + mvy) / 2]

        elif mode == 'vo':
            apex = [mvx, mvy]

        elif mode == 'hrvo':

            rvo_apex = [(vx + mvx) / 2, (vy + mvy) / 2]
            vo_apex = [mvx, mvy]
            cur_v = [vx - rvo_apex[0], vy - rvo_apex[1]]

            if line_right_ori - line_left_ori > pi:
                edge_angle = 2 * pi - line_right_ori + line_left_ori
                center_line = self.wraptopi(line_right_ori + edge_angle / 2.0)
            else:
                edge_angle = line_left_ori - line_right_ori
                center_line = line_left_ori - edge_angle / 2.0

            cl_vector = [cos(center_line), sin(center_line)]
            dis_rv = reciprocal_vel_obs_polygon.distance(rvo_apex, vo_apex)
            radians_rv = atan2(rvo_apex[1] - vo_apex[1], rvo_apex[0] - vo_apex[0])

            diff = line_left_ori - radians_rv

            temp = pi - edge_angle
            if temp == 0:
                temp = temp + 0.01

            dis_diff = dis_rv * sin(diff) / sin(temp)

            if reciprocal_vel_obs_polygon.cross_product(cl_vector, cur_v) <= 0:
                apex = [rvo_apex[0] - dis_diff * cos(line_right_ori), rvo_apex[1] - dis_diff * sin(line_right_ori)]
            else:
                apex = [vo_apex[0] + dis_diff * cos(line_right_ori), vo_apex[1] + dis_diff * sin(line_right_ori)]

        return apex + [line_left_ori, line_right_ori]

    @staticmethod
    def config_vo_line(agent_state, line):
        """ Construct VO between circular-shaped robot and line obstacle """
        x, y, theta, vx, vy, r = agent_state[0:6]

        # r = r + 0.2
        # line in [[sx, sy], [ex, ey]]
        apex = [0, 0]
        theta1 = atan2(line[0][1] - y, line[0][0] - x)
        theta2 = atan2(line[1][1] - y, line[1][0] - x)

        dis_mr1 = sqrt((line[0][1] - y) ** 2 + (line[0][0] - x) ** 2)
        dis_mr2 = sqrt((line[1][1] - y) ** 2 + (line[1][0] - x) ** 2)

        half_angle1 = asin(reciprocal_vel_obs_polygon.clamp(r / dis_mr1, 0, 1))
        half_angle2 = asin(reciprocal_vel_obs_polygon.clamp(r / dis_mr2, 0, 1))

        if reciprocal_vel_obs_polygon.wraptopi(theta2 - theta1) > 0:
            line_left_ori = reciprocal_vel_obs_polygon.wraptopi(theta2 + half_angle2)
            line_right_ori = reciprocal_vel_obs_polygon.wraptopi(theta1 - half_angle1)
        else:
            line_left_ori = reciprocal_vel_obs_polygon.wraptopi(theta1 + half_angle1)
            line_right_ori = reciprocal_vel_obs_polygon.wraptopi(theta2 - half_angle2)

        return apex + [line_left_ori, line_right_ori]

    def vel_candidate(self, agent_state, vo_list):
        """ Split the velocity in two sets """
        # vo_list: [vx, vy, vl,vr]
        vo_outside, vo_inside = [], []

        # state: [x, y, theta, vx, vy, max_radius, vx_des, vy_des]
        cur_vx, cur_vy = agent_state[3:5]
        cur_vx_range = np.clip([cur_vx - self.acceler, cur_vx + self.acceler], -self.vxmax, self.vxmax)
        cur_vy_range = np.clip([cur_vy - self.acceler, cur_vy + self.acceler], -self.vymax, self.vymax)

        # all velocity
        # cur_vx_range = np.array([-self.vxmax, self.vxmax])
        # cur_vy_range = np.array([-self.vymax, self.vymax])

        for new_vx in np.arange(cur_vx_range[0], cur_vx_range[1], 0.05):
            for new_vy in np.arange(cur_vy_range[0], cur_vy_range[1], 0.05):  # 0.05
                if self.vo_out2(new_vx, new_vy, vo_list):
                    vo_outside.append([new_vx, new_vy])
                else:
                    vo_inside.append([new_vx, new_vy])

        # print('vo_outside:', vo_outside)
        return vo_outside, vo_inside

    @staticmethod
    def vo_out(vx, vy, vo_list):
        for rvo in vo_list:
            theta = atan2(vy - rvo[1], vx - rvo[0])
            if reciprocal_vel_obs_polygon.between_angle(rvo[2], rvo[3], theta):
                return False

        return True

    @staticmethod
    def vo_out2(vx, vy, vo_list):
        """ Return True if v is outside the combined velocity obstacle """
        for vo in vo_list:
            # vo_list = rvo + obs_cir_vo + obs_line_vo
            line_left_vector = [cos(vo[2]), sin(vo[2])]
            line_right_vector = [cos(vo[3]), sin(vo[3])]
            line_vector = [vx - vo[0], vy - vo[1]]
            if reciprocal_vel_obs_polygon.between_vector(line_left_vector, line_right_vector, line_vector):
                return False
        return True

    def vel_select(self, agent_state, vo_outside, vo_inside, nei_state_list, obs_poly_list, obs_cir_list, vo_mode):
        """ Selct the most optimal velocity """
        # get the reference velocity
        vel_des = [agent_state[6], agent_state[7]]

        # not very crowded
        if len(vo_outside) > 0:
            temp = min(vo_outside, key=lambda v: reciprocal_vel_obs_polygon.distance(v, vel_des))
            return temp

        else:  # crowded
            # adjust the weight for time 1 -> 1.2
            # print('environment is crowded!')
            temp = min(vo_inside,
                       key=lambda v: self.penalty(vo_mode, v, vel_des, agent_state, nei_state_list, obs_poly_list,
                                                  obs_cir_list, 4.0))
            return temp

    def penalty(self, vo_mode, vel, vel_des, agent_state, nei_state_list, obs_poly_list, obs_cir_list, factor):
        """ Get the penalty for the candidate velocity """
        # agent_state: [x, y, theta, vx, vy, extended_radius, vx_des, vy_des, radius]
        # nei_state: [x, y, theta, vx, vy, extended_radius, radius]
        tc_list = []
        for moving in nei_state_list:
            rel_x, rel_y = agent_state[0:2] - moving[0:2]
            if vo_mode == 'vo':
                rel_vx = vel[0] - moving[3]
                rel_vy = vel[1] - moving[4]
            else:  # for rvo and hrvo, in fact, for hrvo is not suitable
                rel_vx = 2 * vel[0] - moving[3] - agent_state[3]
                rel_vy = 2 * vel[1] - moving[4] - agent_state[4]

            tc = self.cal_exp_tim(rel_x, rel_y, rel_vx, rel_vy, agent_state[5] + moving[5])
            # tc = self.cal_exp_tim(rel_x, rel_y, rel_vx, rel_vy, agent_state[8] + moving[6])
            tc_list.append(tc)

        for obs_poly in obs_poly_list:
            rel_x, rel_y = agent_state[0:2] - obs_poly[0:2]
            rel_vx = vel[0] - obs_poly[3]
            rel_vy = vel[1] - obs_poly[4]

            tc = self.cal_exp_tim(rel_x, rel_y, rel_vx, rel_vy, agent_state[5] + obs_poly[5])
            # tc = self.cal_exp_tim(rel_x, rel_y, rel_vx, rel_vy, agent_state[8] + obs_poly[6])
            tc_list.append(tc)

        for obs_cir in obs_cir_list:
            rel_x, rel_y = agent_state[0:2] - obs_cir[0:2]
            rel_vx = vel[0] - obs_cir[2]
            rel_vy = vel[1] - obs_cir[3]

            tc = self.cal_exp_tim(rel_x, rel_y, rel_vx, rel_vy, agent_state[5] + obs_cir[4])
            tc_list.append(tc)

        # get the mininum time to collision with any other object or obstacle
        tc_min = min(tc_list)

        if tc_min == 0:
            tc_inv = inf
        else:
            tc_inv = 1 / tc_min

        penalty_vel = factor * tc_inv + reciprocal_vel_obs_polygon.distance(vel_des, vel)
        return penalty_vel

    # judge the direction by vector
    @staticmethod
    def between_vector(line_left_vector, line_right_vector, line_vector):
        """ in the scope (line_left, line_right) """
        if reciprocal_vel_obs_polygon.cross_product(line_left_vector, line_vector) < 0 and \
                reciprocal_vel_obs_polygon.cross_product(line_right_vector, line_vector) > 0:
            return True
        else:
            return False

    @staticmethod
    def between_angle(line_left_ori, line_right_ori, line_ori):
        """ True in the scope """
        if reciprocal_vel_obs_polygon.wraptopi(line_ori - line_left_ori) <= 0 and reciprocal_vel_obs_polygon.wraptopi(
                line_ori - line_right_ori) >= 0:
            return True
        else:
            return False

    @staticmethod
    def distance(point1, point2):
        """ calculate the distance between two points """
        return sqrt((point2[0] - point1[0]) ** 2 + (point2[1] - point1[1]) ** 2)

    @staticmethod
    def cross_product(vector1, vector2):
        """ x1 * y2 - x2 * y1 """
        return float(vector1[0] * vector2[1] - vector2[0] * vector1[1])

    @staticmethod
    def cal_exp_tim(rel_x, rel_y, rel_vx, rel_vy, r):
        """ Calculate expect collision time """
        # rel_x: xa - xb
        # rel_y: ya - yb

        # (vx2 + vy2)*t2 + (2x*vx + 2*y*vy)*t+x2+y2-r2 = 0

        a = rel_vx ** 2 + rel_vy ** 2
        b = 2 * rel_x * rel_vx + 2 * rel_y * rel_vy
        c = rel_x ** 2 + rel_y ** 2 - r ** 2

        # the current distance is less than the radius
        if c <= 0:
            return 0
        temp = b ** 2 - 4 * a * c

        if temp <= 0:
            t = inf
        else:
            t1 = (-b + sqrt(temp)) / (2 * a)
            t2 = (-b - sqrt(temp)) / (2 * a)

            t3 = t1 if t1 >= 0 else inf
            t4 = t2 if t2 >= 0 else inf
            t = min(t3, t4)
        return t

    @staticmethod
    def segment_in_circle(x, y, r, line):
        """ Get the segment of line in circle, return the two endpoints """
        # [x, y, r] center point and the radius of the circle
        # line: given in two poit [s_x, s_y, e_x, e_y]
        # reference: https://stackoverflow.com/questions/1073336/circle-line-segment-collision-detection-algorithm
        start_point = np.array(line[0:2])

        d = np.array([line[2] - line[0], line[3] - line[1]])
        f = np.array([line[0] - x, line[1] - y])

        # t2 * (d · d) + 2t*( f · d ) + ( f · f - r2 ) = 0
        a = d @ d
        b = 2 * f @ d
        c = f @ f - r ** 2

        # Discriminant of root
        discriminant = b ** 2 - 4 * a * c

        if discriminant < 0:  # Disjoint
            return None
        else:
            # calculate the root
            t1 = (-b - sqrt(discriminant)) / (2 * a)
            t2 = (-b + sqrt(discriminant)) / (2 * a)

            if 0 <= t1 <= 1 and 0 <= t2 <= 1:
                segment_point1 = start_point + t1 * d
                segment_point2 = start_point + t2 * d

            elif 0 <= t1 <= 1 and t2 > 1:
                segment_point1 = start_point + t1 * d
                segment_point2 = np.array(line[2:4])

            elif t1 < 0 and 0 <= t2 <= 1:
                segment_point1 = np.array(line[0:2])
                segment_point2 = start_point + t2 * d

            elif t1 < 0 and t2 > 1:
                segment_point1 = np.array(line[0:2])
                segment_point2 = np.array(line[2:4])
            else:
                return None

        diff_norm = np.linalg.norm(segment_point1 - segment_point2)

        if diff_norm == 0:
            return None

        return [segment_point1, segment_point2]

    @staticmethod
    def wraptopi(theta):
        """ Convert the theta to (-pi, pi) """
        if theta > pi:
            theta = theta - 2 * pi
        if theta < -pi:
            theta = theta + 2 * pi

        return theta

    @staticmethod
    def clamp(n, minn, maxn):
        """ clip n in (minn, maxn) """
        return max(min(maxn, n), minn)

    @staticmethod
    def exp_collision_segment(obs_seg, x, y, vx, vy, r):
        """ get the time collision with segment for circle robot """
        # [[point], [point]] in np.array()
        point1 = obs_seg[0]
        point2 = obs_seg[1]

        t1 = reciprocal_vel_obs_polygon.cal_exp_tim(x - point1[0], y - point1[1], vx, vy, r)
        t2 = reciprocal_vel_obs_polygon.cal_exp_tim(x - point2[0], y - point2[1], vx, vy, r)

        c_point = np.array([x, y])

        l0 = (point2 - point1) @ (point2 - point1)
        t = (c_point - point1) @ (point2 - point1) / l0
        project = point1 + t * (point2 - point1)  # get the project point with min distance
        distance = sqrt((project - c_point) @ (project - c_point))
        theta1 = atan2((project - c_point)[1], (project - c_point)[0])
        theta2 = atan2(vy, vx)
        theta3 = reciprocal_vel_obs_polygon.wraptopi(theta2 - theta1)

        # project the direction in the vel
        real_distance = (distance - r) / cos(theta3)
        speed = sqrt(vy ** 2 + vx ** 2)

        if speed == 0:
            t3 = inf
        else:
            t3 = real_distance / speed
        if t3 < 0:
            t3 = inf

        return min([t1, t2, t3])
