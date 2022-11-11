import numpy as np
from math import sin, cos, atan2, pi, sqrt
from ir_sim.world import motion_diff, motion_omni, lidar2d
from collections import namedtuple
from ir_sim.util import collision_polygon_polygon, collision_polygon_circle, collision_seg_matrix, collision_seg_seg


class Polygon_Robot:
    def __init__(self, index, mode='diff', init_vertexes=None, vel=np.zeros((2, 1)), vel_max=1 * np.ones((2, 1)),
                 goal=np.zeros((2, 1)), goal_threshold=0.3, step_time=0.1, collision_thick=1, safe_margin=0.15,
                 **kwargs):
        """ Init the polytopic robot """
        self.id = index
        self.mode = mode
        self.step_time = step_time

        # vertexes given in [n, 2], change to (2, n)
        if isinstance(init_vertexes, list):
            init_vertexes = np.array(init_vertexes, ndmin=2).T

        if isinstance(vel, list):
            vel = np.array(vel, ndmin=2).T

        # in shape (2, 1) [v, w]
        if isinstance(vel_max, list):
            vel_max = np.array(vel_max, ndmin=2).T

        if isinstance(goal, list):
            goal = np.array(goal, ndmin=2).T

        # vertex_list in size of (n, 2), numpy in size of (2, n)
        self.vertexes = init_vertexes
        # extended vertexes, for constructing VO, size in (2, n)
        self.extended_vertexes = np.ones_like(init_vertexes)

        # get the vertex number
        self.ver_num = self.vertexes.shape[1]

        # the vertex vector
        self.vertexes_vector = None
        # the extended vertex vector
        self.extended_vertexes_vector = None

        # the maxlength of vertex vector
        self.max_radius = None
        # the maxlength of the extended vertex vector
        self.extended_radius = None

        # safe margin
        self.safe_distance = safe_margin

        # diff vel
        self.vel = vel
        self.vel_max = vel_max
        # omni vel
        self.vel_omni = np.zeros((2, 1))
        self.vel_omni_max = np.array([[self.vel_max[0, 0]], [self.vel_max[0, 0]]])
        # total travel distance
        self.travel_distance = 0

        self.goal = goal
        self.goal_threshold = goal_threshold

        self.arrive_flag = False
        self.collision_flag = False

        # collision
        self.edge_list = None
        self.A = None
        self.b = None
        self.b_collision = None
        self.collision_thick = collision_thick

        lidar_args = kwargs.get('lidar2d', None)

        if lidar_args is not None:
            id_list = lidar_args['id_list']
            if self.id in id_list:
                self.lidar = lidar2d(**lidar_args)
        else:
            self.lidar = None

        # init
        self.state = None

        self.get_center_point()
        self.get_vertex_vector()
        self.update_vertex()

        self.gen_edges()
        self.gen_matrix()

        self.previous_state = self.state
        self.init_state = self.state
        self.diff2omni()

        # noise
        self.__noise = kwargs.get('noise', False)
        self.__alpha = kwargs.get('alpha', [0.03, 0, 0, 0.03, 0, 0])
        self.__control_std = kwargs.get('control_std', [0.01, 0.01])

    def update_info(self, state, vel):
        """ Update the information of the robot manully """
        self.state = state
        self.vel = vel

    def get_travel_distance(self):
        return self.travel_distance

    def get_center_point(self):
        """ Get the center point of the polygon, that is the state """
        center_x = 0.0
        center_y = 0.0
        for i in range(self.ver_num):
            center_x = center_x + self.vertexes[0, i]
            center_y = center_y + self.vertexes[1, i]

        center_x = center_x / self.ver_num
        center_y = center_y / self.ver_num

        # set the theta as the direction for start to goal
        center_theta = atan2(self.goal[1][0] - center_y, self.goal[0][0] - center_x)
        self.state = np.array([center_x, center_y, pi], ndmin=2).T

    def get_vertex_vector(self):
        """ Get the vertex vertor and the max vector radius (including extended) """
        self.vertexes_vector = np.ones_like(self.vertexes)
        self.extended_vertexes_vector = np.ones_like(self.vertexes)

        for i in range(self.ver_num):
            self.vertexes_vector[0, i] = self.vertexes[0, i] - self.state[0, 0]
            self.vertexes_vector[1, i] = self.vertexes[1, i] - self.state[1, 0]

            # for extended vertex_vector
            theta = atan2(self.vertexes_vector[1, i], self.vertexes_vector[0, i])
            length = np.linalg.norm(self.vertexes_vector[:, i])

            # # method 1 changing as the length of robot
            # self.extended_vertexes_vector[0, i] = (1 + self.safe_distance) * length * cos(theta)
            # self.extended_vertexes_vector[1, i] = (1 + self.safe_distance) * length * sin(theta)

            # method 2  fixed distance
            # if i >= 2:
            #     length = length + 0.05
            self.extended_vertexes_vector[0, i] = (length + self.safe_distance) * cos(theta)
            self.extended_vertexes_vector[1, i] = (length + self.safe_distance) * sin(theta)

        # get the max radius and the extended radius
        self.max_radius = -1.0
        self.extended_radius = -1.0
        for i in range(self.ver_num):
            radius = np.linalg.norm(self.vertexes_vector[:, i])
            if radius > self.max_radius:
                self.max_radius = radius

            extended_radius = np.linalg.norm(self.extended_vertexes_vector[:, i])
            if extended_radius > self.extended_radius:
                self.extended_radius = extended_radius

    def update_vertex(self):
        """ Update the all vertex when the center point moves, (2, n) """
        # first need to update the vertex_vector
        rotation_matrix = np.array(
            [[cos(self.state[2, 0]), -sin(self.state[2, 0])],
             [sin(self.state[2, 0]), cos(self.state[2, 0])]])

        # update the vertexes and the extended vertexes
        temp_vertexes_vector = rotation_matrix @ self.vertexes_vector
        temp_extended_vertexes_vector = rotation_matrix @ self.extended_vertexes_vector

        for i in range(self.ver_num):
            self.vertexes[0, i] = self.state[0, 0] + temp_vertexes_vector[0, i]
            self.vertexes[1, i] = self.state[1, 0] + temp_vertexes_vector[1, i]

            self.extended_vertexes[0, i] = self.state[0, 0] + temp_extended_vertexes_vector[0, i]
            self.extended_vertexes[1, i] = self.state[1, 0] + temp_extended_vertexes_vector[1, i]

    def get_vertexes(self):
        """ Get the vertexes of the polytopic robot, for collision avoidance """
        return self.vertexes

    def get_extended_vertexes(self):
        """ Get the extended vertexes of the polytopic robot, for constructing VO """
        return self.extended_vertexes

    def get_previous_vertex(self):
        """ Get the previous vertices of the polytopic robot """
        # first need to update the vertex_vector
        rotation_matrix = np.array(
            [[cos(self.previous_state[2, 0]), -sin(self.previous_state[2, 0])],
             [sin(self.previous_state[2, 0]), cos(self.previous_state[2, 0])]])

        # update the vertexes
        temp_vertexes_vector = rotation_matrix @ self.vertexes_vector
        previous_vertex = np.ones_like(self.vertexes)

        for i in range(self.ver_num):
            previous_vertex[0, i] = self.previous_state[0, 0] + temp_vertexes_vector[0, i]
            previous_vertex[1, i] = self.previous_state[1, 0] + temp_vertexes_vector[1, i]

        return previous_vertex

    def gen_edges(self):
        """ Get all the edge of the polygon need to update """
        point = namedtuple('point', 'x y')
        self.edge_list = []
        for i in range(self.ver_num - 1):
            edge = [point(self.vertexes[0, i], self.vertexes[1, i]),
                    point(self.vertexes[0, i + 1], self.vertexes[1, i + 1])]
            self.edge_list.append(edge)

        edge_final = [point(self.vertexes[0, self.ver_num - 1], self.vertexes[1, self.ver_num - 1]),
                      point(self.vertexes[0, 0], self.vertexes[1, 0])]
        self.edge_list.append(edge_final)

    def gen_matrix(self):
        """ Get the matrix A, b, b_collision """
        self.A = np.zeros((self.ver_num, 2))
        self.b = np.zeros((self.ver_num, 1))
        self.b_collision = np.zeros((self.ver_num, 1))

        for i in range(self.ver_num):
            if i + 1 < self.ver_num:
                pre_point = self.vertexes[:, i]
                next_point = self.vertexes[:, i + 1]
            else:
                pre_point = self.vertexes[:, i]
                next_point = self.vertexes[:, 0]

            diff = next_point - pre_point

            a = diff[1]
            b = -diff[0]
            c = a * pre_point[0] + b * pre_point[1]

            self.A[i, 0] = a
            self.A[i, 1] = b
            self.b[i, 0] = c

            if b != 0:
                self.b_collision[i, 0] = c + self.collision_thick * abs(b)
            else:
                self.b_collision[i, 0] = c + self.collision_thick * abs(a)

        return self.A, self.b

    def inside(self, point):
        """ Return True if the point inside the  polygon """
        assert point.shape == (2, 1)
        temp = self.A @ point - self.b
        return (self.A @ point < self.b).all(), temp

    def inside_collision(self, point):
        """ Judge if a point have collision with the polygon """
        assert point.shape == (2, 1)
        temp = self.A @ point - self.b_collision
        return (self.A @ point < self.b_collision).all(), temp

    def move_to_goal(self):
        """ Get a velocity move to the goal """
        vel = self.cal_des_vel()
        self.move_forward(vel)

    def cal_des_vel(self):
        vel = None
        if self.mode == 'diff':
            vel = self.cal_des_diff_vel()
        elif self.mode == 'omni':
            vel = self.cal_des_vel_omni()

        return vel

    def cal_des_diff_vel(self, tolerance=0.12):
        """ Caculate the diff velocity to the goal """
        dis, radian = Polygon_Robot.relative(self.state[0:2], self.goal)
        robot_radian = self.state[2, 0]

        v_max = self.vel_max[0, 0]
        w_max = self.vel_max[1, 0]
        diff_radian = Polygon_Robot.to_pi(radian - robot_radian)

        w_opti = 0
        # w > 0 so counterclockwise rotation
        if diff_radian > tolerance:
            w_opti = w_max
        elif diff_radian < - tolerance:
            w_opti = - w_max

        if dis < self.goal_threshold:
            v_opti = 0
            w_opti = 0
        else:
            # the car move along the current direction
            v_opti = v_max * cos(diff_radian)
            if v_opti < 0:
                v_opti = 0

        return np.array([[v_opti], [w_opti]])

    def cal_des_vel_omni(self):
        """ Caculate the omni velocity to the goal """
        dis, radian = Polygon_Robot.relative(self.state[0:2], self.goal)

        if dis > self.goal_threshold:
            vx = self.vel_max[0, 0] * cos(radian)
            vy = self.vel_max[0, 0] * sin(radian)
        else:
            vx = 0
            vy = 0
        return np.array([[vx], [vy]])

    def move_forward(self, vel, vel_type='diff', stop=True, **kwargs):
        """ Move with the diff velocity or omni vel """
        if isinstance(vel, list):
            vel = np.array(vel, ndmin=2).T

        if vel.shape == (2,):
            vel = vel[:, np.newaxis]

        assert vel.shape == (2, 1)
        # the robot is permitted to stop its action
        if stop:
            if self.arrive_flag or self.collision_flag:
                vel = np.zeros((2, 1))

        assert self.mode == "diff" or self.mode == 'omni'
        self.previous_state = self.state
        if self.mode == 'diff':
            if vel_type == 'diff':
                vel = np.clip(vel, -self.vel_max, self.vel_max)
                self.move_with_diff(vel, self.__noise, self.__alpha)

            elif vel_type == 'omni':
                vel = np.clip(vel, -self.vel_omni_max, self.vel_omni_max)
                self.move_from_omni(vel, self.__noise, self.__alpha, **kwargs)
        else:
            vel = np.clip(vel, -self.vel_omni_max, self.vel_omni_max)
            self.move_with_omni(vel, self.__noise, self.__control_std)

        # check if arrive the goal
        if not self.arrive_flag:
            self.arrive()

        # update the information, include vertexes, edges, matrix
        self.update_vertex()
        self.gen_edges()
        self.gen_matrix()

    def move_with_diff(self, vel, noise=False, alpha=None):
        """ Move in differential mode """
        if alpha is None:
            alpha = [0.01, 0, 0, 0.01, 0, 0]
        next_state, distance = motion_diff(self.state, vel, self.step_time, noise, alpha)
        self.travel_distance = self.travel_distance + distance
        self.state = next_state
        self.vel = vel
        self.diff2omni()

    def diff2omni(self):
        """ Convert the diff_vel to omni_vel """
        # the direction of vel_diff is the heading direction
        vel_linear = self.vel[0, 0]
        theta = self.state[2, 0]

        vx = vel_linear * cos(theta)
        vy = vel_linear * sin(theta)
        self.vel_omni = np.array([[vx], [vy]])

    def omni2diff(self, vel_omni, guarantee_time=0.2, tolerance=0.001, mini_speed=0.001):
        """ Convert the omni_vel to diff_vel """

        vel_radians = atan2(vel_omni[1, 0], vel_omni[0, 0])
        robot_radians = self.state[2, 0]
        # project the velocity to the heading direction, that is, finally the heading direction will be the velocity dir
        diff_radians = robot_radians - vel_radians
        diff_radians = Polygon_Robot.to_pi(diff_radians)

        # calculate the w
        w_max = self.vel_max[1, 0]
        if tolerance > diff_radians > -tolerance:
            w = 0
        else:
            w = -diff_radians / guarantee_time
            if w > w_max:
                w = w_max
            elif w < -w_max:
                w = -w_max

        # calculate the v_diff
        speed = sqrt(vel_omni[0, 0] ** 2 + vel_omni[1, 0] ** 2)
        if speed > self.vel_max[0, 0]:
            speed = self.vel_max[0, 0]

        v = speed * cos(diff_radians)
        # that is mean spin in place, no reverse driving
        if v < 0:
            v = 0
        # if speed is small, then stop the robot, inclding the case which [vx, vy] = [0, 0]
        if speed <= mini_speed:
            v = 0
            w = 0
        vel_diff = np.array([[v], [w]])
        return vel_diff

    def move_from_omni(self, vel_omni, noise=False, alpha=None, **kwargs):
        """ the mode of robot is diff but the mode of vel is omni """
        if alpha is None:
            alpha = [0.01, 0, 0, 0.01, 0, 0]

        # change from omni velocity
        vel_diff = np.round(self.omni2diff(vel_omni, **kwargs), 2)
        next_state, distance = motion_diff(self.state, vel_diff, self.step_time, noise, alpha)
        self.travel_distance = self.travel_distance + distance
        self.state = next_state
        self.vel = vel_diff
        self.diff2omni()

    def move_with_omni(self, vel_omni, noise, std):
        # vel_omni: np.array([[vx], [vy]])

        next_state = motion_omni(self.state[0:2], vel_omni, self.step_time, noise, std)
        cur_direction = np.array([[atan2(vel_omni[1, 0], vel_omni[0, 0])]])
        self.state = np.concatenate((next_state, cur_direction), axis=0)
        self.vel_omni = vel_omni
        self.vel = np.round(self.omni2diff(vel_omni), 2)

    def arrive(self):
        """ Judge if reach the end point """
        position = self.state[0:2]
        dist = np.linalg.norm(position - self.goal[0:2])

        if dist < self.goal_threshold:
            print('polygon robot_{} have arrive the goal'.format(self.id))
            self.arrive_flag = True
            self.robot_stop()
            return True
        else:
            self.arrive_flag = False
            return False

    def collision_check(self, components):
        if self.collision_flag:
            return True

        circle = namedtuple('circle', 'x y r')
        point = namedtuple('point', 'x y')

        # check collision with the map
        for segment in self.edge_list:
            if collision_seg_matrix(segment, components['map_matrix'], components['xy_reso'], components['offset']):
                self.collision_flag = True
                print('collisions between obstacle map')
                return True

        # check collision with other polygon_robots
        for polygon_robot in components['polygon_robots'].robot_list:
            if polygon_robot.id == self.id:  # for same condition
                continue

            if collision_polygon_polygon(self.vertexes, polygon_robot.vertexes):
                self.collision_flag = True
                polygon_robot.collision_flag = True
                print('Collision betwenn polygon robot_{} and polygon robot_{}'.format(self.id, polygon_robot.id))
                return True

        # check collision with the circle_robots
        for robot in components['robots'].robot_list:
            temp_circle = circle(robot.state[0, 0], robot.state[1, 0], robot.radius)
            if collision_polygon_circle(self.vertexes, temp_circle):
                self.collision_flag = True
                robot.collision_flag = True
                print('collisions between circle_robots and polygon_robots')
                return True

        # check collision with obs_circle
        for obs_cir in components['obs_circles'].obs_cir_list:
            temp_circle = circle(obs_cir.state[0, 0], obs_cir.state[1, 0], obs_cir.radius)
            if collision_polygon_circle(self.vertexes, temp_circle):
                self.collision_flag = True
                print('collisions with circle obstacles')
                return True

        # check collision with the obs_line
        for line in components['obs_lines'].obs_line_states:
            obs_seg = [point(line[0], line[1]), point(line[2], line[3])]
            for polygon_seg in self.edge_list:
                if collision_seg_seg(obs_seg, polygon_seg):
                    self.collision_flag = True
                    print('collisions with line obstacle')
                    return True

        # check collision with the obs_polygon
        for polygon in components['obs_polygons'].obs_poly_list:
            if collision_polygon_polygon(self.vertexes, polygon.vertexes):
                self.collision_flag = True
                print('Collision betwenn polygon obstacles')
                return True

    def omni_state(self):
        """ Get the omni state of the robot_self """
        v_des = self.cal_des_vel_omni()
        max_rc_array = self.max_radius * np.ones((1, 1))
        # radius_collision scope, use extended radius
        extended_rc_array = self.extended_radius * np.ones((1, 1))

        # concatenate the vector in the row, with [x, y, theta, vx, vy, r, vx_des, vx_des]
        return np.concatenate((self.state, self.vel_omni, extended_rc_array, v_des, max_rc_array), axis=0)

    # condtruct obstacle state
    def omni_obs_state(self):
        """ Get the omni state of the robot_self when it serves as the obstacle """
        max_rc_array = self.max_radius * np.ones((1, 1))
        extended_rc_array = self.extended_radius * np.ones((1, 1))
        return np.concatenate((self.state, self.vel_omni, extended_rc_array, max_rc_array), axis=0)

    # reset all the state
    def reset(self):
        """ Reset all the state of the robot """
        self.state[:] = self.init_state[:]
        self.previous_state[:] = self.init_state[:]

        self.vel = np.zeros((2, 1))
        self.vel_omni = np.zeros((2, 1))
        self.travel_distance = 0

        self.arrive_flag = False
        self.collision_flag = False

    @staticmethod
    def relative(state1, state2):
        """ Calculate the distance and radian between state1 and state2 """
        dif = state2[0:2] - state1[0:2]

        dis = np.linalg.norm(dif)
        radian = atan2(dif[1, 0], dif[0, 0])

        return dis, radian

    @staticmethod
    def to_pi(radian):
        """ Convert the radian to (-pi, pi) """
        if radian > pi:
            radian = radian - 2 * pi
        elif radian < -pi:
            radian = radian + 2 * pi

        return radian

    def robot_stop(self):
        self.vel = np.zeros((2, 1))
        self.vel_omni = np.zeros((2, 1))
