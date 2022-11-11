from ir_sim.world import mobile_robot
from math import pi, cos, sin
import numpy as np
from collections import namedtuple
from ir_sim.util import collision_cir_cir, collision_cir_matrix, collision_cir_seg
from ir_sim.util import reciprocal_vel_obs


class env_robot:
    def __init__(
        self,
        robot_class=mobile_robot,
        robot_number=0,
        robot_mode="omni",
        robot_init_mode=0,
        step_time=0.1,
        components=None,
        vo_mode='rvo',
        **kwargs
    ):

        if components is None:
            components = []
        self.robot_class = robot_class
        self.robot_number = robot_number
        self.init_mode = robot_init_mode
        self.robot_list = []
        self.cur_mode = robot_init_mode
        self.com = components

        # VO
        self.vo_mode = vo_mode
        self.rvo = reciprocal_vel_obs()

        self.interval = kwargs.get("interval", 1)
        # [x_min, ymin, x_max, y_max]
        self.square = kwargs.get("square", [1, 0.5, 9.5, 9.5])
        # circular area: x, y, radius
        self.circular = kwargs.get("circular", [5, 5, 4])

        # kwargs: random_bear, random radius
        self.random_bear = kwargs.get("random_bear", False)
        self.random_radius = kwargs.get("random_radius", False)

        radius_list = []
        init_state_list = []
        goal_list = []
        # init_mode: 0 manually: initialize all the robot's start and target position
        #            1 single row:  all the robot initialize in a single row
        #            2 random: all the robot initialize in a random way
        #            3 circular: all the robot initialize in a circle
        if self.robot_number > 0:
            if self.init_mode == 0:
                assert (
                    "radius_list" and "init_state_list" and "goal_list" in kwargs.keys()
                )
                radius_list = kwargs["radius_list"]
                init_state_list = kwargs["init_state_list"]
                goal_list = kwargs["goal_list"]
            else:
                radius_list = kwargs.get("radius_list", [0.2])
                init_state_list, goal_list, radius_list = self.init_state_distribute(
                    self.init_mode, radius=radius_list[0]
                )

        # robot
        for i in range(self.robot_number):
            robot = self.robot_class(
                index=i,
                mode=robot_mode,
                radius=radius_list[i],
                init_state=init_state_list[i],
                goal=goal_list[i],
                step_time=step_time,
                **kwargs
            )
            self.robot_list.append(robot)
            self.robot = robot if i == 0 else None

    def init_state_distribute(self, init_mode=1, radius=0.2):
        # init_mode: 1 single row
        #            2 random
        #            3 circular
        # square area: x_min, y_min, x_max, y_max
        # circular area: x, y, radius

        num = self.robot_number
        state_list, goal_list = [], []

        if init_mode == 1:
            # single row
            state_list = [
                np.array([[i * self.interval], [self.square[1]], [pi / 2]])
                for i in range(int(self.square[0]), int(self.square[0]) + num)
            ]
            goal_list = [
                np.array([[i * self.interval], [self.square[3]]])
                for i in range(int(self.square[0]), int(self.square[0]) + num)
            ]
            # cross-walk
            goal_list.reverse()

        elif init_mode == 2:
            # random
            state_list, goal_list = self.random_start_goal()

        elif init_mode == 3:
            # circular
            circle_point = np.array(self.circular)
            theta_step = 2 * pi / num
            theta = 0

            while theta < 2 * pi:
                state = circle_point + np.array(
                    [
                        cos(theta) * self.circular[2],
                        sin(theta) * self.circular[2],
                        theta + pi - self.circular[2],
                    ]
                )
                goal = (
                    circle_point[0:2]
                    + np.array([cos(theta + pi), sin(theta + pi)]) * self.circular[2]
                )
                theta = theta + theta_step
                state_list.append(state[:, np.newaxis])
                goal_list.append(goal[:, np.newaxis])

        elif init_mode == 4:
            # random 2
            circle_point = np.array(self.circular)
            theta_step = 2 * pi / num
            theta = 0

            while theta < 2 * pi:
                state = circle_point + np.array(
                    [
                        cos(theta) * self.circular[2],
                        sin(theta) * self.circular[2],
                        theta + pi - self.circular[2],
                    ]
                )
                goal = (
                    circle_point[0:2]
                    + np.array([cos(theta + pi), sin(theta + pi)]) * self.circular[2]
                )
                theta = theta + theta_step
                state_list.append(state[:, np.newaxis])
                goal_list.append(goal[:, np.newaxis])

        elif init_mode == 5:
            # start 7 6 5 4
            # goal  3 2 1 0

            # goal  4 5 6 7
            # start 0 1 2 3

            half_num = int(num / 2)

            # start
            state_list1 = [
                np.array([[i * self.interval], [self.square[1]], [pi / 2]])
                for i in range(int(self.square[0]), int(self.square[0]) + half_num)
            ]

            state_list2 = [
                np.array([[i * self.interval], [self.square[3]], [pi / 2]])
                for i in range(int(self.square[0]), int(self.square[0]) + num - half_num)
            ]
            state_list2.reverse()

            goal_list1 = [
                np.array([[i * self.interval], [self.square[3]], [pi / 2]])
                for i in range(int(self.square[0]), int(self.square[0]) + half_num)
            ]
            goal_list1.reverse()

            goal_list2 = [
                np.array([[i * self.interval], [self.square[1]], [pi / 2]])
                for i in range(int(self.square[0]), int(self.square[0]) + num - half_num)
            ]

            state_list, goal_list = state_list1 + state_list2, goal_list1 + goal_list2

        # random heading direction
        if self.random_bear:
            for state in state_list:
                state[2, 0] = np.random.uniform(low=-pi, high=pi)

        if self.random_radius:
            radius_list = np.random.uniform(low=0.2, high=1, size=(num,))
        else:
            radius_list = [radius for i in range(num)]

        return state_list, goal_list, radius_list

    def random_start_goal(self):

        num = self.robot_number
        random_list = []
        goal_list = []

        # get start_point and end_point
        while len(random_list) < 2 * num:

            new_point = np.random.uniform(
                low=self.square[0:2] + [-pi], high=self.square[2:4] + [pi], size=(1, 3)
            ).T

            if not env_robot.check_collision(
                new_point, random_list, self.com, self.interval
            ):
                random_list.append(new_point)

        start_list = random_list[0:num]
        goal_temp_list = random_list[num: 2 * num]

        # goal only need for x, y, don't need theta and deleta theta, goal in shape (2, 1)
        for goal in goal_temp_list:
            goal_list.append(np.delete(goal, 2, 0))

        return start_list, goal_list

    def random_goal(self):
        """ used for generate random goal"""
        num = self.robot_number
        random_list = []
        goal_list = []
        while len(random_list) < num:

            new_point = np.random.uniform(
                low=self.square[0:2] + [-pi], high=self.square[2:4] + [pi], size=(1, 3)
            ).T

            if not env_robot.check_collision(
                new_point, random_list, self.com, self.interval
            ):
                random_list.append(new_point)

        goal_temp_list = random_list[:]
        for goal in goal_temp_list:
            goal_list.append(np.delete(goal, 2, 0))

        return goal_list

    @staticmethod
    def distance(point1, point2):
        diff = point2[0:2] - point1[0:2]
        return np.linalg.norm(diff)

    @staticmethod
    def check_collision(check_point, point_list, components, obs_range):
        """ Check if the random start and goal have collision with env or other robot's start and goal position """
        circle = namedtuple("circle", "x y r")
        point = namedtuple("point", "x y")
        self_circle = circle(check_point[0, 0], check_point[1, 0], obs_range / 2)

        # check collision with obs_cir
        for obs_cir in components["obs_circles"].obs_cir_list:
            temp_circle = circle(
                obs_cir.state[0, 0], obs_cir.state[1, 0], obs_cir.radius
            )
            if collision_cir_cir(self_circle, temp_circle):
                return True

        # check collision with map
        if collision_cir_matrix(
            self_circle,
            components["map_matrix"],
            components["xy_reso"],
            components["offset"],
        ):
            return True

        # check collision with line obstacles
        for line in components["obs_lines"].obs_line_states:
            segment = [point(line[0], line[1]), point(line[2], line[3])]
            if collision_cir_seg(self_circle, segment):
                return True

        for point in point_list:
            if env_robot.distance(check_point, point) < range:
                return True

        return False

    def step(self, vel_list=None, **vel_kwargs):

        # vel_kwargs: vel_type = 'diff', 'omni'
        #             stop=True, whether stop when arrive at the goal
        #             noise=False,
        #             alpha = [0.01, 0, 0, 0.01, 0, 0], noise for diff
        #             control_std = [0.01, 0.01], noise for omni

        if vel_list is None:
            vel_list = []
        for robot, vel in zip(self.robot_list, vel_list):
            robot.move_forward(vel, **vel_kwargs)

    def cal_des_list(self):
        """ Return a velocity list that contains all robots' preferred velocity """
        vel_list = list(map(lambda x: x.cal_des_vel(), self.robot_list))
        return vel_list

    def cal_des_omni_list(self):
        """ Return a velocity list that contains all robots' preferred velocity in omni mode"""
        vel_list = list(map(lambda x: x.cal_des_vel_omni(), self.robot_list))
        return vel_list

    def arrive_all(self):
        """ Judge if all robots have arrived the goal """
        for robot in self.robot_list:
            if not robot.arrive():
                return False

        return True

    def robots_reset(self, reset_mode=1, **kwargs):
        """ Reset all the status for robots """
        # reset all the status
        if reset_mode == 0:
            for robot in self.robot_list:
                robot.reset(self.random_bear)

        # reset the status of robots and change the goal
        elif reset_mode == 4:
            goal_list = self.random_goal()
            for i in range(self.robot_number):
                self.robot_list[i].goal = goal_list[i]
                self.robot_list[i].reset(self.random_bear)

        # change the mode and reset the status of robots
        elif self.cur_mode != reset_mode:
            # radius is not changed
            state_list, goal_list, _ = self.init_state_distribute(init_mode=reset_mode)

            for i in range(self.robot_number):
                self.robot_list[i].init_state = state_list[i]
                self.robot_list[i].goal = goal_list[i]
                self.robot_list[i].reset(self.random_bear)

            self.cur_mode = reset_mode

        # just random all robots' start and target position
        elif reset_mode == 2:
            state_list, goal_list = self.random_start_goal()
            for i in range(self.robot_number):
                self.robot_list[i].init_state = state_list[i]
                self.robot_list[i].goal = goal_list[i]
                self.robot_list[i].reset(self.random_bear)

        else:
            for robot in self.robot_list:
                robot.reset(self.random_bear)

    def robot_reset(self, index=0):
        """ Reset the status of one robot """
        self.robot_list[index].reset(self.random_bear)

    def total_states(self):
        """ Get the total states to construct VO """
        robot_state_list = list(
            map(lambda r: np.squeeze(r.omni_state()), self.robot_list)
        )
        nei_state_list = list(
            map(lambda r: np.squeeze(r.omni_obs_state()), self.robot_list)
        )
        obs_circular_list = list(
            map(
                lambda o: np.squeeze(o.omni_obs_state()),
                self.com["obs_circles"].obs_cir_list,
            )
        )
        obs_line_list = self.com["obs_lines"].obs_line_states

        return [robot_state_list, nei_state_list, obs_circular_list, obs_line_list]

    def get_rvo_vel_list(self):
        """ Get the velocity calculated by VO for all robots """
        ts = self.total_states()
        rvo_vel_list = list(map(lambda r: self.rvo.cal_vel(r, ts[1], ts[2], ts[3], mode=self.vo_mode), ts[0]))

        return rvo_vel_list

    # def render(self, time=0.1, save=False, path=None, i = 0, **kwargs):

    #     self.world_plot.draw_robot_diff_list(**kwargs)
    #     self.world_plot.draw_obs_cir_list()
    #     self.world_plot.pause(time)

    #     if save == True:
    #         self.world_plot.save_gif_figure(path, i)

    #     self.world_plot.com_cla()

    # def seg_dis(self, segment, point):

    #     point = np.squeeze(point[0:2])
    #     sp = np.array(segment[0:2])
    #     ep = np.array(segment[2:4])

    #     l2 = (ep - sp) @ (ep - sp)

    #     if (l2 == 0.0):
    #         return np.linalg.norm(point - sp)

    #     t = max(0, min(1, ((point-sp) @ (ep-sp)) / l2 ))

    #     projection = sp + t * (ep-sp)

    #     distance = np.linalg.norm(point - projection)

    #     return distance
