import numpy as np
from math import sin, cos, atan2, pi, sqrt
from ir_sim.world import motion_diff, motion_omni, lidar2d
from collections import namedtuple
from ir_sim.util import collision_cir_cir, collision_cir_matrix, collision_cir_seg


class mobile_robot:
    def __init__(
            self,
            index,
            mode="diff",
            init_state=np.zeros((3, 1)),
            vel_diff=np.zeros((2, 1)),
            vel_omni=np.zeros((2, 1)),
            vel_max=2 * np.ones((2, 1)),
            goal=np.zeros((2, 1)),
            goal_threshold=0.1,
            radius=0.2,
            step_time=0.1,
            **kwargs
    ):
        """ Circular-shaped mobile robot """
        # mode: 'diff', 'omni', 'ackerman'
        # init_state: start_position, goal: target position
        # vel_diff, vel_omni: the velocity for diff or omni

        self.id = int(index)
        self.mode = mode
        self.step_time = step_time

        # Change a list like this [1, 2, 3] to [[1, 2, 3]], and then transpose
        if isinstance(init_state, list):
            init_state = np.array(init_state, ndmin=2).T

        if isinstance(vel_diff, list):
            vel_diff = np.array(vel_diff, ndmin=2).T

        if isinstance(vel_omni, list):
            vel_omni = np.array(vel_omni, ndmin=2).T

        if isinstance(vel_max, list):
            vel_max = np.array(vel_max, ndmin=2).T

        if isinstance(goal, list):
            goal = np.array(goal, ndmin=2).T

        if mode == "diff":
            self.state = init_state
            self.previous_state = init_state
            self.init_state = init_state

        # state only for [x, y]
        elif mode == "omni":
            self.state = init_state[0:2]
            self.previous_state = init_state[0:2]
            self.init_state = init_state[0:2]

        self.vel_diff = vel_diff
        self.vel_omni = vel_omni
        self.vel_max = vel_max

        self.goal = goal
        self.goal_threshold = goal_threshold

        self.radius = radius
        self.radius_collision = round(radius + kwargs.get("radius_exp", 0.1), 2)
        self.arrive_flag = False
        self.collision_flag = False

        lidar_args = kwargs.get("lidar2d", None)

        if lidar_args is not None:
            id_list = lidar_args["id_list"]
            if self.id in id_list:
                self.lidar = lidar2d(**lidar_args)
        else:
            self.lidar = None

        self.__noise = kwargs.get("noise", False)
        self.__alpha = kwargs.get("alpha", [0.03, 0, 0, 0.03, 0, 0])
        self.__control_std = kwargs.get("control_std", [0.01, 0.01])

    def update_info(self, state, vel):
        """ Update the information of the robot manually """
        self.state = state
        if self.mode == 'diff':
            self.vel_diff = vel
        else:
            self.vel_omni = vel

    def move_forward(self, vel, vel_type="diff", stop=True, **kwargs):
        """ Move forward with the velocity """
        # default: robot mode: diff, no noise, vel_type: diff
        # vel_type: diff: np.array([[linear], [angular]])
        #           omni: np.array([[v_x], [v_y]])
        # kwargs: guarantee_time = 0.2, tolerance = 0.1, mini_speed=0.02,

        if isinstance(vel, list):
            vel = np.array(vel, ndmin=2).T

        # make sure the shape of vel is (2, 1)
        if vel.shape == (2,):
            vel = vel[:, np.newaxis]

        assert vel.shape == (2, 1)
        vel = np.clip(vel, -self.vel_max, self.vel_max)

        if stop:
            if self.arrive_flag or self.collision_flag:
                vel = np.zeros((2, 1))

        assert self.mode == "omni" or self.mode == "diff"

        self.previous_state = self.state
        if self.mode == "diff":
            if vel_type == "diff":
                self.move_with_diff(vel, self.__noise, self.__alpha)
            elif vel_type == "omni":
                self.move_from_omni(vel, self.__noise, self.__alpha, **kwargs)

        elif self.mode == "omni":
            self.move_with_omni(vel, self.__noise, self.__control_std)
        if not self.arrive_flag:
            self.arrive()

    def cal_lidar_range(self, components):
        if self.lidar is not None:
            self.lidar.cal_range(self.state, components)

    def move_with_diff(self, vel_diff, noise=False, alpha=None):
        """ vel_diff: np.array([[v], [w]]), shape is (2, 1) """
        # vel_diff: np.array([[v], [w]]), shap
        if alpha is None:
            alpha = [0.01, 0, 0, 0.01, 0, 0]
        next_state, distance = motion_diff(self.state, vel_diff, self.step_time, noise, alpha)
        self.state = next_state
        self.vel_diff = vel_diff
        self.diff2omni()

    def diff2omni(self):
        """ Convert the diff_vel to omni_vel """
        vel_linear = self.vel_diff[0, 0]
        theta = self.state[2, 0]

        # the direction of velocity is the heading direction
        vx = vel_linear * cos(theta)
        vy = vel_linear * sin(theta)
        self.vel_omni = np.array([[vx], [vy]])

    def move_from_omni(self, vel_omni, noise=False, alpha=None, **kwargs):
        """ A robot in diff mode while moves with a omni velocity """
        # vel_omni: np.array([[vx], [vy]])
        if alpha is None:
            alpha = [0.01, 0, 0, 0.01, 0, 0]
        vel_diff = np.round(self.omni2diff(vel_omni, **kwargs), 2)
        next_state, distance = motion_diff(self.state, vel_diff, self.step_time, noise, alpha)
        self.state = next_state
        self.vel_diff = vel_diff
        self.diff2omni()

    def omni2diff(self, vel_omni, guarantee_time=0.2, tolerance=0.1, mini_speed=0.02):
        """ Convert the omni_vel to the diff_vel """
        vel_radians = atan2(vel_omni[1, 0], vel_omni[0, 0])
        robot_radians = self.state[2, 0]
        # project the velocity to the heading direction
        diff_radians = robot_radians - vel_radians

        if diff_radians > pi:
            diff_radians = diff_radians - 2 * pi
        elif diff_radians < -pi:
            diff_radians = diff_radians + 2 * pi

        # calculate the w
        w_max = self.vel_max[1, 0]
        if tolerance > diff_radians > -tolerance:
            w = 0
        else:
            w = -diff_radians / guarantee_time
            if w > w_max:
                w = w_max

        # calculate the v
        speed = sqrt(vel_omni[0, 0] ** 2 + vel_omni[1, 0] ** 2)
        if speed > self.vel_max[0, 0]:
            speed = self.vel_max[0, 0]
        v = speed * cos(diff_radians)

        if v < 0:
            v = 0

        if speed <= mini_speed:
            v = 0
            w = 0

        vel_diff = np.array([[v], [w]])

        return vel_diff

    def move_with_omni(self, vel_omni, noise, std):
        """ Omni mode robot move with omni velocity """
        # vel_omni: np.array([[vx], [vy]])
        next_state = motion_omni(self.state, vel_omni, self.step_time, noise, std)

        self.state = next_state
        self.vel_omni = vel_omni

    def move_to_goal(self):
        """ Move to the goal with ignoring the obstacle """
        vel = self.cal_des_vel()
        self.move_forward(vel)

    def cal_des_vel(self, tolerance=0.12):
        """ Calculate the velocity to the destination """
        des_vel = None
        if self.mode == "diff":
            des_vel = self.cal_des_vel_diff(tolerance=tolerance)
        elif self.mode == "omni":
            des_vel = self.cal_des_vel_omni()

        return des_vel

    def cal_des_vel_diff(self, tolerance=0.12):
        """ Calculate the diff velocity to the destination """
        dis, radian = mobile_robot.relative(self.state[0:2], self.goal)
        robot_radian = self.state[2, 0]

        v_max = self.vel_max[0, 0]
        w_max = self.vel_max[1, 0]
        diff_radian = mobile_robot.to_pi(radian - robot_radian)

        w_opti = 0
        # w > 0 so counterclockwise rotation
        if diff_radian > tolerance:
            w_opti = w_max
        elif diff_radian < -tolerance:
            w_opti = -w_max

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
        """ Calculate the omni velocity to the destination """
        dis, radian = mobile_robot.relative(self.state[0:2], self.goal)

        if dis > self.goal_threshold:
            vx = self.vel_max[0, 0] * cos(radian)
            vy = self.vel_max[1, 0] * sin(radian)
        else:
            vx = 0
            vy = 0

        return np.array([[vx], [vy]])

    def if_goal(self, goal):

        position = self.state[0:2]
        dist = np.linalg.norm(position - goal)

        if dist < self.radius:
            return True
        else:
            return False

    def arrive(self):
        """ Judge if reach the end point """
        position = self.state[0:2]
        dist = np.linalg.norm(position - self.goal[0:2])

        if dist < self.goal_threshold:
            print('circle robot_{} have arrive the goal'.format(self.id))
            self.arrive_flag = True
            return True
        else:
            self.arrive_flag = False
            return False

    def collision_check(self, components):
        """ Return True there is a collision between robots or robot and obstacles """
        circle = namedtuple("circle", "x y r")
        point = namedtuple("point", "x y")

        self_circle = circle(self.state[0, 0], self.state[1, 0], self.radius)

        if self.collision_flag:
            return True

        # check collision among robots
        for robot in components["robots"].robot_list:
            temp_circle = circle(robot.state[0, 0], robot.state[1, 0], robot.radius)
            if collision_cir_cir(self_circle, temp_circle):
                robot.collision_flag = True
                self.collision_flag = True
                print('Collisions between robot_{} and robot_{}'.format(robot.id, self.id))
                return True

        # check collision with obstacles in circle form
        for obs_cir in components["obs_circles"].obs_cir_list:
            temp_circle = circle(
                obs_cir.state[0, 0], obs_cir.state[1, 0], obs_cir.radius
            )
            if collision_cir_cir(self_circle, temp_circle):
                self.collision_flag = True
                print("Collisions with obstacles")
                return True

        # check collision with map
        if collision_cir_matrix(
                self_circle,
                components["map_matrix"],
                components["xy_reso"],
                components["offset"],
        ):
            self.collision_flag = True
            print("Collisions with map obstacles")
            return True

        # check collision with line obstacles
        for line in components["obs_lines"].obs_line_states:
            segment = [point(line[0], line[1]), point(line[2], line[3])]
            if collision_cir_seg(self_circle, segment):
                self.collision_flag = True
                print("Collisions between obstacles")
                return True

        # check collision with the polygon obstacle
        for polygon in components["obs_polygons"].obs_poly_list:
            for edge in polygon.edge_list:
                segment = [point(edge[0], edge[1]), point(edge[2], edge[3])]
                if collision_cir_seg(self_circle, segment):
                    self.collision_flag = True
                    print("Collisions between polygon obstacles")
                    return True

    def omni_state(self):
        """ Get the omni state of the robot self """
        v_des = self.cal_des_vel_omni()
        rc_array = self.radius_collision * np.ones((1, 1))

        # concatenate the vector in the row
        return np.concatenate((self.state[0:2], self.vel_omni, rc_array, v_des), axis=0)

    def omni_obs_state(self):
        """ Get the omni state of the robot as a obstacle """
        rc_array = self.radius * np.ones((1, 1))
        return np.concatenate((self.state[0:2], self.vel_omni, rc_array), axis=0)

    def reset(self, random_bear=False):
        """ Reset all the state """
        self.state[:] = self.init_state[:]
        self.previous_state[:] = self.init_state[:]
        self.vel_omni = np.zeros((2, 1))
        self.vel_diff = np.zeros((2, 1))
        self.arrive_flag = False
        self.collision_flag = False

        if random_bear:
            self.state[2, 0] = np.random.uniform(low=-pi, high=pi)

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
