from vo_polytope.world import obs_circle
from math import pi, cos, sin
import numpy as np
from collections import namedtuple
from vo_polytope.util import (
    collision_cir_cir,
    collision_cir_matrix,
    collision_cir_seg,
    reciprocal_vel_obs,
)


class env_obs_cir:
    def __init__(
        self,
        obs_cir_class=obs_circle,
        obs_model="static",
        obs_cir_num=1,
        dist_mode=0,
        step_time=0.1,
        components=None,
        **kwargs
    ):

        if components is None:
            components = []
        self.obs_cir_class = obs_cir_class
        # 'static' or 'dynamic'
        self.obs_model = obs_model
        self.obs_num = obs_cir_num
        # distributed mode
        self.dist_mode = dist_mode

        self.obs_cir_list = []
        self.components = components

        # square area: x_min, y_min, x_max, y_max
        self.obs_square = kwargs.get("obs_square", [0, 0, 10, 10])
        self.obs_interval = kwargs.get("obs_interval", 1)
        self.random_bear = kwargs.get('random_bear', False)

        obs_radius_list = []
        obs_state_list = []
        obs_goal_list = []

        if self.obs_num > 0:
            if self.dist_mode == 0:
                assert "obs_radius_list" and "obs_state_list" in kwargs.keys()
                obs_radius_list = kwargs["obs_radius_list"]
                obs_state_list = kwargs["obs_state_list"]
                obs_goal_list = kwargs.get("obs_goal_list", [0] * self.obs_num)

                if len(obs_radius_list) < self.obs_num:
                    temp_end = obs_radius_list[-1]
                    obs_radius_list += [
                        temp_end for _ in range(self.obs_num - len(obs_radius_list))
                    ]

            else:
                obs_radius_list = kwargs.get("obs_radius_list", [0.2])
                obs_state_list, obs_goal_list, obs_radius_list = self.obs_state_dis(
                    obs_init_mode=self.dist_mode, radius=obs_radius_list[0], **kwargs
                )

        if self.obs_model == "dynamic":
            self.rvo = reciprocal_vel_obs(vxmax=1.5, vymax=1.5, **kwargs)

        for i in range(self.obs_num):
            obs_cir = self.obs_cir_class(
                index=i,
                state=obs_state_list[i],
                radius=obs_radius_list[i],
                step_time=step_time,
                obs_model=obs_model,
                goal=obs_goal_list[i],
                **kwargs
            )
            self.obs_cir_list.append(obs_cir)

    def step_wander(self, **kwargs):
        """ Make the obs_cir wander in the world """
        ts = self.obs_total_states()
        # Get collision avoidance velocity use VO
        rvo_vel_list = list(
            map(lambda agent_s: self.rvo.cal_vel(agent_s, nei_state_list=ts[1]), ts[0])
        )
        arrive_flag = False

        for i, obs_cir in enumerate(self.obs_cir_list):
            obs_cir.move_forward(rvo_vel_list[i], **kwargs)

            if obs_cir.arrive():
                arrive_flag = True

        if arrive_flag:
            # reset the goal position
            goal_list = self.random_goal(**kwargs)

            for i, obs_cir in enumerate(self.obs_cir_list):
                obs_cir.goal = goal_list[i]

    def obs_state_dis(
        self,
        obs_init_mode=1,
        radius=0.2,
        circular=None,
        min_radius=0.2,
        max_radius=1,
        **kwargs
    ):
        # init_mode: 1 single row
        #            2 random
        #            3 circular
        # square area: x_min, y_min, x_max, y_max
        # circular area: x, y, radius

        # [x, y, radius]
        if circular is None:
            circular = [5, 5, 4]
        random_radius = kwargs.get("random_radius", False)

        num = self.obs_num
        state_list, goal_list = [], []

        if obs_init_mode == 1:
            # the obs_cir is generated in a single row
            # start in y_min, end in y_max
            # 5 4 3 2 1
            # 1 2 3 4 5
            state_list = [
                np.array([[i * self.obs_interval], [self.obs_square[1]]])
                for i in range(int(self.obs_square[0]), int(self.obs_square[0]) + num)
            ]
            goal_list = [
                np.array([[i * self.obs_interval], [self.obs_square[3]]])
                for i in range(int(self.obs_square[0]), int(self.obs_square[0]) + num)
            ]
            goal_list.reverse()

        elif obs_init_mode == 2:
            # random
            state_list, goal_list = self.random_start_goal(**kwargs)

        elif obs_init_mode == 3:
            # circular
            circle_point = np.array(circular)
            theta_step = 2 * pi / num
            theta = 0

            while theta < 2 * pi:
                state = circle_point + np.array(
                    [
                        cos(theta) * circular[2],
                        sin(theta) * circular[2],
                        theta + pi - circular[2],
                    ]
                )
                goal = (
                    circle_point[0:2]
                    + np.array([cos(theta + pi), sin(theta + pi)]) * circular[2]
                )
                theta = theta + theta_step
                state_list.append(state[:, np.newaxis])
                goal_list.append(goal[:, np.newaxis])

        if random_radius:
            radius_list = np.random.uniform(
                low=min_radius, high=max_radius, size=(num,)
            )
        else:
            radius_list = [radius for _ in range(num)]

        return state_list, goal_list, radius_list

    def random_start_goal(self, **kwargs):
        """ Get the start and goal position randomly """
        num = self.obs_num
        random_list = []

        while len(random_list) < 2 * num:

            new_point = np.random.uniform(
                low=self.obs_square[0:2], high=self.obs_square[2:4], size=(1, 2)
            ).T

            if not env_obs_cir.check_collision(
                new_point, random_list, self.components, self.obs_interval
            ):
                random_list.append(new_point)

        start_list = random_list[0:num]
        goal_list = random_list[num: 2 * num]

        return start_list, goal_list

    def random_goal(self, **kwargs):
        """ Reset the goal position of all robots, to realize wander """
        num = self.obs_num
        goal_list = []

        while len(goal_list) < num:

            new_point = np.random.uniform(
                low=self.obs_square[0:2], high=self.obs_square[2:4], size=(1, 2)
            ).T

            if not env_obs_cir.check_collision(
                new_point, goal_list, self.components, self.obs_interval
            ):
                goal_list.append(new_point)

        return goal_list

    @staticmethod
    def check_collision(check_point, point_list, components, obs_range):
        """ Check collision with the map and other obstacles """
        circle = namedtuple("circle", "x y r")
        point = namedtuple("point", "x y")
        self_circle = circle(check_point[0, 0], check_point[1, 0], obs_range / 2)

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
            if env_obs_cir.distance(check_point, point) < obs_range:
                return True

        return False

    @staticmethod
    def distance(point1, point2):
        diff = point2[0:2] - point1[0:2]
        return np.linalg.norm(diff)

    def obs_total_states(self):

        agent_state_list = list(
            map(lambda a: np.squeeze(a.omni_state()), self.obs_cir_list)
        )
        nei_state_list = list(
            map(lambda a: np.squeeze(a.omni_obs_state()), self.obs_cir_list)
        )

        return agent_state_list, nei_state_list
