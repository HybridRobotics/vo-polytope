import numpy as np
from math import cos, sin, atan2
from ir_sim.world import motion_omni


class obs_polygon:
    def __init__(self, vertex=None, vel=np.zeros((2, 1)), step_time=0.1, goal=np.zeros((2, 1)), goal_threshold=0.1,
                 obs_model='static', collision_thick=1, vel_max=1.0 * np.ones((2, 1)), safe_margin=0.15, **kwargs):

        self.ver_num = None
        self.vertexes = None
        if vertex is not None:
            # convert the shape from (n, 2) to (2, n)
            self.vertexes = np.array(vertex).T
            self.ver_num = self.vertexes.shape[1]

        if isinstance(vel, list):
            vel = np.array(vel, ndmin=2).T
        if isinstance(vel_max, list):
            vel_max = np.array(vel_max, ndmin=2).T
        if isinstance(goal, list):
            goal = np.array(goal, ndmin=2).T

        self.safe_distance = safe_margin
        self.vertexes_vector = None
        # for constructing VO for polytopic robots
        self.extended_vertexes = np.ones_like(self.vertexes)
        self.extended_vertexes_vector = None

        self.max_radius = None
        self.extended_radius = None

        self.center_point = None
        self.previous_state = None

        # kinematics
        self.step_time = step_time
        # omni velocity
        self.vel = vel
        self.vel_max = vel_max

        self.goal = goal
        self.goal_threshold = goal_threshold
        self.obs_model = obs_model
        self.arrive_flag = False

        # collision check
        self.edge_list = None
        self.A = None
        self.b = None
        self.b_collision = None
        self.collision_thick = collision_thick

        # init
        self.get_center_point()
        self.get_vertex_vector()
        self.update_vertex()

        self.gen_edges()
        self.gen_matrix()

    def get_center_point(self):
        """ Get the center point (current state) of the polygon """
        center_x = 0.0
        center_y = 0.0
        for i in range(self.ver_num):
            center_x = center_x + self.vertexes[0, i]
            center_y = center_y + self.vertexes[1, i]

        center_x = center_x / self.ver_num
        center_y = center_y / self.ver_num

        # shape in (2, 1) and the initial direction is zero
        self.center_point = np.array([center_x, center_y, 0], ndmin=2).T

        self.previous_state = self.center_point

    def get_vertex_vector(self):
        """ Get the vertex vertor and the max vector radius (including extended) """
        self.vertexes_vector = np.ones_like(self.vertexes)
        self.extended_vertexes_vector = np.ones_like(self.vertexes)

        for i in range(self.ver_num):
            # Normal vertex_vector
            self.vertexes_vector[0, i] = self.vertexes[0, i] - self.center_point[0, 0]
            self.vertexes_vector[1, i] = self.vertexes[1, i] - self.center_point[1, 0]

            # for extended vertex_vector
            theta = atan2(self.vertexes_vector[1, i], self.vertexes_vector[0, i])
            length = np.linalg.norm(self.vertexes_vector[:, i])

            # # method 1 as the size of robot
            # self.extended_vertexes_vector[0, i] = (1 + self.safe_distance) * length * cos(theta)
            # self.extended_vertexes_vector[1, i] = (1 + self.safe_distance) * length * sin(theta)

            # method 2 fix distance
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
        """ Update all vertices when the robot moves"""
        # first need to update the vertex_vector
        rotation_matrix = np.array(
            [[cos(self.center_point[2, 0]), -sin(self.center_point[2, 0])],
             [sin(self.center_point[2, 0]), cos(self.center_point[2, 0])]])
        temp_vertexes_vector = rotation_matrix @ self.vertexes_vector
        temp_extended_vertexes_vector = rotation_matrix @ self.extended_vertexes_vector

        # update the vertexes and the extended vertexes
        for i in range(self.ver_num):
            self.vertexes[0, i] = self.center_point[0, 0] + temp_vertexes_vector[0, i]
            self.vertexes[1, i] = self.center_point[1, 0] + temp_vertexes_vector[1, i]

            self.extended_vertexes[0, i] = self.center_point[0, 0] + temp_extended_vertexes_vector[0, i]
            self.extended_vertexes[1, i] = self.center_point[1, 0] + temp_extended_vertexes_vector[1, i]

    def get_vertexes(self):
        """ Get the vertexes of the polytopic robot, for collision avoidance """
        return self.vertexes

    def get_extended_vertexes(self):
        """ Get the extended vertexes of the polytopic robot, for constructing VO """
        return self.extended_vertexes

    def get_previous_vertexes(self):
        """ Get the previous vertexes for the polytopic robot, for plotting """
        previous_vertexes = np.ones_like(self.vertexes)

        rotation_matrix = np.array(
            [[cos(self.previous_state[2, 0]), -sin(self.previous_state[2, 0])],
             [sin(self.previous_state[2, 0]), cos(self.previous_state[2, 0])]])
        previous_vertexes_vector = rotation_matrix @ self.vertexes_vector

        # update the previous vertexes
        for i in range(self.ver_num):
            previous_vertexes[0, i] = self.previous_state[0, 0] + previous_vertexes_vector[0, i]
            previous_vertexes[1, i] = self.previous_state[1, 0] + previous_vertexes_vector[1, i]

        return previous_vertexes

    def gen_edges(self):
        """ Get all the edge of the polygon """
        self.edge_list = []
        for i in range(self.ver_num - 1):
            edge = [
                self.vertexes[0, i],
                self.vertexes[1, i],
                self.vertexes[0, i + 1],
                self.vertexes[1, i + 1],
            ]
            self.edge_list.append(edge)

        edge_final = [
            self.vertexes[0, self.ver_num - 1],
            self.vertexes[1, self.ver_num - 1],
            self.vertexes[0, 0],
            self.vertexes[1, 0],
        ]
        self.edge_list.append(edge_final)

    def gen_matrix(self):
        """ Get Ax <= b """
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
        """ Judge if a point is inside the polygon """
        assert point.shape == (2, 1)
        temp = self.A @ point - self.b
        return (self.A @ point < self.b).all(), temp

    def inside_collision(self, point):
        """ Judge if a point have collision with the polygon """
        assert point.shape == (2, 1)
        temp = self.A @ point - self.b_collision
        return (self.A @ point < self.b_collision).all(), temp

    def move_forward(self, stop=True):
        """ Move for the dynamic obstacle """
        if self.obs_model != 'static':
            self.vel = self.cal_des_vel_omni()
            if stop:
                if self.arrive_flag:
                    self.vel = np.zeros((2, 1))

            self.previous_state = self.center_point
            next_state = motion_omni(self.center_point[0:2], self.vel, self.step_time)
            cur_direction = np.array([[atan2(self.vel[1, 0], self.vel[0, 0])]])
            self.center_point = np.concatenate((next_state, cur_direction), axis=0)

            if not self.arrive_flag:
                self.arrive()
            self.update_vertex()
            self.gen_edges()
            self.gen_matrix()

    def omni_obs_state(self):
        """ return x, y, theta, vx, vy, extended_max_radius, max_radius """
        max_rc_array = self.max_radius * np.ones((1, 1))
        extended_rc_array = self.extended_radius * np.ones((1, 1))
        return np.concatenate((self.center_point, self.vel, extended_rc_array, max_rc_array), axis=0)

    def cal_des_vel_omni(self):
        """ Caculate the omni velocity to the goal location """
        dis, radian = obs_polygon.relative(self.center_point[0:2], self.goal)

        if dis > self.goal_threshold:
            vx = self.vel_max[0, 0] * cos(radian)
            vy = self.vel_max[1, 0] * sin(radian)
        else:
            vx = 0
            vy = 0
        return np.array([[vx], [vy]])

    def arrive(self):
        """ Judge whether the obstacle arrive the target position """
        dist = np.linalg.norm(self.center_point[0:2] - self.goal[0:2])

        if dist < self.goal_threshold:
            self.arrive_flag = True
            self.vel = np.zeros((2, 1))
            return True
        else:
            self.arrive_flag = False
            return False

    @staticmethod
    def relative(state1, state2):
        """ Calculate the distance and radian between state1 and state2, state1 -> state2 """
        dif = state2[0:2] - state1[0:2]

        dis = np.linalg.norm(dif)
        radian = atan2(dif[1, 0], dif[0, 0])

        return dis, radian
