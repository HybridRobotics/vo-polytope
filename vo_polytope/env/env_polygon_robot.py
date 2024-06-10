import random
import numpy as np
from vo_polytope.world import Polygon_Robot
from math import pi, cos, sin, atan2, fabs
from collections import namedtuple
from vo_polytope.util import collision_polygon_polygon, collision_polygon_circle, collision_seg_seg, collision_seg_matrix
from vo_polytope.util import reciprocal_vel_obs_polygon


class env_polygon_robot:
    def __init__(self, polygon_robot_class=Polygon_Robot, robot_number=0, polygon_center_list=None, robot_mode='diff',
                 robot_init_mode=0, step_time=0.1, vector_size=1.0, components=None, vel_max=np.ones((2, 1)), **kwargs):

        if polygon_center_list is None:
            polygon_center_list = []
        if components is None:
            components = []

        self.robot_class = polygon_robot_class
        self.robot_number = robot_number
        self.polygon_center_list = polygon_center_list
        # storage all robots
        self.robot_list = []

        self.robot_mode = robot_mode
        self.init_mode = robot_init_mode
        self.cur_mode = robot_init_mode
        self.com = components
        self.step_time = step_time
        self.vel_max = vel_max

        # obstacle interval
        self.interval = kwargs.get('interval', 0.8)
        # [x_min, ymin, x_max, y_max]
        self.square = kwargs.get('square', [1.0, 1.0, 9.0, 9.0])
        # circular area: x, y, radius
        self.circular = kwargs.get('circular', [5, 5, 4])

        # vertex_vector of the polytopic robot
        self.vertex_vector = kwargs.get('vertex_vector', [[-0.2, -0.15], [0.2, -0.15], [0.2, 0.15], [-0.2, 0.15]])
        self.polytope_vertex_vector = kwargs.get('polytope_vertex_vector', [[[-0.2, -0.15], [0.2, -0.15], [0.2, 0.15],
                                                                             [-0.2, 0.15]]])
        self.vector_size = vector_size
        self.polytope_category_num = len(self.polytope_vertex_vector)
        self.random_vectices_list = []

        self.rvo = reciprocal_vel_obs_polygon(v_max=self.vel_max[0])
        self.vo_mode = kwargs.get('vo_mode', 'vo')

        # kwargs: random_bear random radius
        self.random_bear = kwargs.get('random_bear', False)
        self.random_radius = kwargs.get('random_radius', False)

        # init_mode:
        # 0 manually initialize all the robot's start point and end_point according the yaml
        # 1 single row    all the robot initialize in a single row
        # 2 random +cir   all the robot initialize on a circle in a random way
        # 3 circular      all the robot initialize in a circle (same shape)
        # 4 circular      all the robot initialize in a circle (different shape)

        polygon_init_vertexes = []
        goal_list = []
        if self.robot_number > 0:
            if self.init_mode == 0:
                assert 'goal_list' in kwargs.keys()
                goal_list = kwargs['goal_list']

                polytope_index = 0
                for polytope_center in self.polygon_center_list:
                    # for same rectangled-shape
                    init_vertex = self.get_all_vertex(polytope_center, self.vertex_vector)

                    # for different shape
                    # init_vertex = self.get_all_vertex(polytope_center, self.polytope_vertex_vector[polytope_index])
                    polygon_init_vertexes.append(init_vertex)

                    polytope_index = polytope_index + 1
                    if polytope_index >= self.polytope_category_num:
                        polytope_index = 0
            else:
                polygon_init_vertexes, goal_list = self.init_state_distribute(self.init_mode)

        # generate the robot
        for i in range(self.robot_number):
            robot = self.robot_class(index=i, mode=robot_mode, init_vertexes=polygon_init_vertexes[i],
                                     goal=goal_list[i], step_time=step_time, vel_max=self.vel_max, **kwargs)
            self.robot_list.append(robot)
            self.robot = robot if i == 0 else None

    def get_all_vertex(self, polygon_center, vectices_vector=None):
        """ Get the all the vertex of a polytope, for initialization """
        polygon_vertexes = []
        for vertex_vector in vectices_vector:
            polygon_vertex = [polygon_center[0] + self.vector_size * vertex_vector[0],
                              polygon_center[1] + self.vector_size * vertex_vector[1]]
            polygon_vertexes.append(polygon_vertex)

        # in size (n, 2)
        return polygon_vertexes

    @staticmethod
    def get_edge_list(vertexes):
        """ Get the all edge of the polygon """
        point = namedtuple('point', 'x y')

        edge_list = []
        polygon_vertexes = vertexes
        if isinstance(vertexes, list):
            polygon_vertexes = np.array(vertexes).T
        ver_num = polygon_vertexes.shape[1]

        assert polygon_vertexes.shape[0] == 2
        for i in range(ver_num - 1):
            edge = [point(polygon_vertexes[0, i], polygon_vertexes[1, i]),
                    point(polygon_vertexes[0, i + 1], polygon_vertexes[1, i + 1])]
            edge_list.append(edge)

        edge_final = [point(polygon_vertexes[0, ver_num - 1], polygon_vertexes[1, ver_num - 1]),
                      point(polygon_vertexes[0, 0], polygon_vertexes[1, 0])]
        edge_list.append(edge_final)

        return edge_list

    def init_state_distribute(self, init_mode=0):
        num = self.robot_number
        vertexes_list, goal_list = [], []

        # init_mode:
        # 1 single row
        # 2 random on a circle
        # 3 circular same shape
        # 4 circular different shape
        if init_mode == 1:
            # single row
            for i in range(int(self.square[0]), int(self.square[0]) + num):
                polygon_center = [i * self.interval, self.square[1]]
                polygon_vertexes = self.get_all_vertex(polygon_center, self.vertex_vector)
                goal = np.array([[i * self.interval], [self.square[3]]])

                vertexes_list.append(polygon_vertexes)
                goal_list.append(goal)
            goal_list.reverse()  # cross-walk

        elif init_mode == 2:
            # random generate start_vertex and goal
            vertexes_list, goal_list = self.random_vertexes_goal_circle()
            # vertexes_list, goal_list = self.random_vertexes_goal()

        elif init_mode == 3:
            # circular same shape
            circle_point = np.array(self.circular)
            theta_step = 2 * pi / num
            theta = 0

            while theta < 2 * pi:
                center = circle_point[0:2] + np.array([cos(theta) * self.circular[2], sin(theta) * self.circular[2]])
                polygon_center = list(center)
                polygon_vertexes = self.get_all_vertex(polygon_center, self.vertex_vector)
                vertexes_list.append(polygon_vertexes)

                # goal in semi_circle
                goal = circle_point[0:2] + np.array([cos(theta + pi), sin(theta + pi)]) * self.circular[2]
                # change to (2, 1)
                goal_list.append(goal[:, np.newaxis])

                theta = theta + theta_step

        elif init_mode == 4:
            # circular different shape
            circle_point = np.array(self.circular)
            theta_step = 2 * pi / num
            theta = 0
            polytope_index = 0

            while theta < 2 * pi:
                center = circle_point[0:2] + np.array([cos(theta) * self.circular[2], sin(theta) * self.circular[2]])
                polygon_center = list(center)
                polygon_vertexes = self.get_all_vertex(polygon_center, self.polytope_vertex_vector[polytope_index])
                vertexes_list.append(polygon_vertexes)

                # goal in semi_circle
                goal = circle_point[0:2] + np.array([cos(theta + pi), sin(theta + pi)]) * self.circular[2]
                goal_list.append(goal[:, np.newaxis])

                theta = theta + theta_step
                polytope_index = polytope_index + 1
                if polytope_index >= self.polytope_category_num:
                    polytope_index = 0

        return vertexes_list, goal_list

    def random_vertexes_goal_circle(self):
        """ Random generate the vertexes and goal on a circle """
        num = self.robot_number
        vertexes_list = []
        goal_list = []

        # get center list
        centers_list = []
        circle_point = np.array(self.circular)
        theta_step = 2 * pi / num
        theta = 0
        while theta < 2 * pi:
            center = circle_point[0:2] + np.array([cos(theta) * self.circular[2], sin(theta) * self.circular[2]])
            polygon_center = list(center)
            centers_list.append(polygon_center)
            theta = theta + theta_step

        # get random order of index, in size ()
        start_goal_index = self.random_index()

        # get vextexes and goal
        polytope_index = 0
        for start_goal in start_goal_index:
            # vertexes
            polygon_center = centers_list[start_goal[0]]
            # for same shape
            polygon_vertexes = self.get_all_vertex(polygon_center, self.vertex_vector)

            # for different shape
            # polygon_vertexes = self.get_all_vertex(polygon_center, self.polytope_vertex_vector[polytope_index])
            vertexes_list.append(polygon_vertexes)

            # goal
            goal = centers_list[start_goal[1]]
            goal_list.append(goal)  # list form

            polytope_index = polytope_index + 1
            if polytope_index >= self.polytope_category_num:
                polytope_index = 0

        return vertexes_list, goal_list

    def random_index(self):
        """ Get a random order for center list, in size [robot_num, 2] """
        num = self.robot_number

        # get start and goal index
        start_goal_index = []
        start_index = [False for _ in range(num)]
        goal_index = [False for _ in range(num)]

        # all the start index are used
        while not min(start_index):
            random_start_index = random.randint(0, num - 1)

            # this start index is not used
            if not start_index[random_start_index]:

                # if there is a valid solution
                effective_flag = False
                for i in range(num):
                    if not goal_index[i] and (fabs(i - random_start_index) + num) % (num - 1) > 3:
                        effective_flag = True
                        break

                # if no valid solution, reset all the status
                if not effective_flag:
                    start_index = [False for _ in range(num)]
                    goal_index = [False for _ in range(num)]
                    start_goal_index = []
                    continue

                # exist valid solution, until find it
                while not start_index[random_start_index]:
                    random_goal_index = random.randint(0, num - 1)
                    flag = (fabs(random_goal_index - random_start_index) + num) % (num - 1) > 3
                    if not goal_index[random_goal_index] and flag:
                        start_index[random_start_index] = True
                        goal_index[random_goal_index] = True
                        start_goal_index.append([random_start_index, random_goal_index])

        return start_goal_index

    def random_vertexes_goal(self):
        """ random generate the vertexes and goal """
        num = self.robot_number
        vertexes_list = []
        goal_list = []

        start_list = []
        self.random_vectices_list = []
        polytope_index = 0

        # get start_list
        while len(start_list) < num:
            # new point size in (2, 1)
            new_point = np.random.uniform(low=self.square[0:2], high=self.square[2:4], size=(1, 2)).T
            if not self.check_collision(new_point, self.polytope_vertex_vector[polytope_index],
                                        start_list, self.random_vectices_list, self.com, self.interval):
                start_list.append(new_point)
                self.random_vectices_list.append(self.polytope_vertex_vector[polytope_index])

                polytope_index = polytope_index + 1
                if polytope_index >= self.polytope_category_num:
                    polytope_index = 0

        # get goal_list
        while len(goal_list) < num:
            # new point size in (2, 1)
            new_point = np.random.uniform(low=self.square[0:2], high=self.square[2:4], size=(1, 2)).T
            if not self.check_collision(new_point, self.random_vectices_list[len(goal_list)], start_list + goal_list,
                                        self.random_vectices_list + self.random_vectices_list[0:len(goal_list)],
                                        self.com, self.interval):
                goal_list.append(new_point)

        for current_point, current_vector in zip(start_list, self.random_vectices_list):
            polygon_center = [current_point[0][0], current_point[1][0]]
            polygon_vertexes = self.get_all_vertex(polygon_center, current_vector)
            vertexes_list.append(polygon_vertexes)

        return vertexes_list, goal_list

    def random_goal(self):
        """ used for generate random goal """
        num = self.robot_number
        goal_list = []
        while len(goal_list) < num:
            new_point = np.random.uniform(low=self.square[0:2], high=self.square[2:4], size=(1, 2)).T
            if not self.check_collision(new_point, self.random_vectices_list[len(goal_list)], goal_list,
                                        self.random_vectices_list[0:len(goal_list)], self.com, self.interval):
                goal_list.append(new_point)

        return goal_list

    def collision_between_polygons(self, polygon1_center, polygon1_vector, polygon2_center, polygon2_vector, obs_range):
        """ Check collision between polygons """

        polygon1_vertexes = self.get_all_vertex(polygon1_center, polygon1_vector)
        polygon2_vertexes = self.get_all_vertex(polygon2_center, polygon2_vector)
        if collision_polygon_polygon(polygon1_vertexes, polygon2_vertexes, obs_range):
            return True

        return False

    def collision_with_circle(self, polygon_center, polygon_vector, circle):
        """ Check collision between polygon and circle """
        polygon_vertexes = self.get_all_vertex(polygon_center, polygon_vector)
        if collision_polygon_circle(polygon_vertexes, circle):
            return True

        return False

    def check_collision(self, center, polygon_vector, point_list, point_list_vector, components, obs_range):
        """ check if the random start and goal have collision with env and other start and goal """

        circle = namedtuple('circle', 'x y r')
        point = namedtuple('point', 'x y')

        # center size in shape (2, 1)
        polygon_center = [center[0][0], center[1][0]]
        polygon_vertexes = self.get_all_vertex(polygon_center, polygon_vector)
        polygon_edge = env_polygon_robot.get_edge_list(polygon_vertexes)

        # check collision with the map
        for segment in polygon_edge:
            if collision_seg_matrix(segment, components['map_matrix'], components['xy_reso'], components['offset']):
                return True

        # check collision with other random point for polytope
        for current_point, current_vector in zip(point_list, point_list_vector):
            point_center = [current_point[0][0], current_point[1][0]]
            if self.collision_between_polygons(polygon_center, polygon_vector, point_center, current_vector, obs_range):
                return True

        # check collision with obs_circle
        for obs_cir in components['obs_circles'].obs_cir_list:
            temp_circle = circle(obs_cir.state[0, 0], obs_cir.state[1, 0], obs_cir.radius)
            if self.collision_with_circle(polygon_center, polygon_vector, temp_circle):
                return True

        # check collision with obs_polygon
        for polygon in components['obs_polygons'].obs_poly_list:
            if collision_polygon_polygon(polygon_vertexes, polygon.vertexes):
                return True

        # check collision with obs_line
        for line in components['obs_lines'].obs_line_states:
            obs_seg = [point(line[0], line[1]), point(line[2], line[3])]
            for polygon_seg in polygon_edge:
                if collision_seg_seg(obs_seg, polygon_seg):
                    return True

        return False

    def step(self, vel_list=None, **vel_kwargs):
        """ robot step """
        # vel_kwargs:
        # vel_type = 'diff', 'omni'
        # stop=True, whether stop when arrive at the goal
        # noise=False,
        # alpha = [0.01, 0, 0, 0.01, 0, 0], noise for diff
        # control_std = [0.01, 0.01], noise for omni

        if vel_list is None:
            vel_list = []
        for robot, vel in zip(self.robot_list, vel_list):
            robot.move_forward(vel, **vel_kwargs)

    def cal_des_list(self):
        """ Calculate the preferred vel for all the polytopic robots """
        vel_list = list(map(lambda x: x.cal_des_vel(), self.robot_list))
        return vel_list

    def cal_des_omni_list(self):
        """ Calculate the preferred omni_vel for all the polygon_robots """
        vel_list = list(map(lambda x: x.cal_des_vel_omni(), self.robot_list))
        return vel_list

    def arrive_all(self):
        """ Judge if all the robot arrive the end point """
        for robot in self.robot_list:
            if not robot.arrive():
                return False
        return True

    def robots_reset(self, reset_mode=0, **kwargs):
        """ reset all the status for robots """
        # 0 reset all the staus
        # 1 reset the goal, useless
        # 2 random reset all the status
        # X according to the mode and reset all the status
        if reset_mode == 0:
            for robot in self.robot_list:
                robot.reset()

        # only change the goal and reset the status of the robot, useless
        elif reset_mode == 1:
            goal_list = self.random_goal()
            for i in range(self.robot_number):
                self.robot_list[i].goal = goal_list[i]
                self.robot_list[i].reset()

        elif reset_mode == 2:
            vertexes_list, goal_list = self.random_vertexes_goal_circle()
            self.robot_list = []
            for i in range(self.robot_number):
                # init_state is changed
                robot = self.robot_class(index=i, mode=self.robot_mode, init_vertexes=vertexes_list[i],
                                         goal=goal_list[i], step_time=self.step_time, **kwargs)
                self.robot_list.append(robot)

        elif self.cur_mode != reset_mode:
            vertexes_list, goal_list = self.init_state_distribute(init_mode=reset_mode)
            self.robot_list = []
            for i in range(self.robot_number):
                # init_state is changed
                robot = self.robot_class(index=i, mode=self.robot_mode, init_vertexes=vertexes_list[i],
                                         goal=goal_list[i], step_time=self.step_time, **kwargs)
                self.robot_list.append(robot)
            self.cur_mode = reset_mode

    def robot_reset(self, index=0):
        """ reset the status of the specific robot """
        self.robot_list[index].reset()

    def total_states(self):
        """ get the all state """
        robot_state_list = list(map(lambda r: np.squeeze(r.omni_state()), self.robot_list))
        nei_state_list = list(map(lambda r: np.squeeze(r.omni_obs_state()), self.robot_list))
        obs_poly_list = list(map(lambda o: np.squeeze(o.omni_obs_state()), self.com['obs_polygons'].obs_poly_list))
        obs_circular_list = list(map(lambda o: np.squeeze(o.omni_obs_state()), self.com['obs_circles'].obs_cir_list))

        return [robot_state_list, nei_state_list, obs_poly_list, obs_circular_list]

    def get_rvo_vel_list(self, vo_mode='vo'):
        """ Get the velocity selected based VO_p for each polytopic robots """
        ts = self.total_states()

        # for extended vertexes
        robot_vertexes = list(map(lambda r: r.get_extended_vertexes(), self.robot_list))
        obstacle_vertexes = list(map(lambda o: o.get_extended_vertexes(), self.com['obs_polygons'].obs_poly_list))

        # robot_vertexes = list(map(lambda r: r.get_vertexes(), self.robot_list))
        # obstacle_vertexes = list(map(lambda o: o.get_vertexes(), self.com['obs_polygons'].obs_poly_list))

        rvo_vel_list = list(map(lambda r, r_vertex: self.rvo.cal_polygon_vel(agent_state=r, agent_vertex=r_vertex,
                                                                             nei_state_list=ts[1],
                                                                             nei_vertex_list=robot_vertexes,
                                                                             obs_poly_list=ts[2],
                                                                             obs_vertex_list=obstacle_vertexes,
                                                                             obs_cir_list=ts[3],
                                                                             mode=vo_mode), ts[0], robot_vertexes))
        return rvo_vel_list

    def get_rvo_vel_list_hybrid(self, vo_mode='vo'):
        """ Get the velocity selected based VO_c for each polytopic robots """
        ts = self.total_states()

        rvo_vel_list = list(map(lambda r: self.rvo.cal_polygon_vel_cir(agent_state=r, nei_state_list=ts[1],
                                                                       obs_poly_list=ts[2], obs_cir_list=ts[3],
                                                                       mode=vo_mode), ts[0]))

        return rvo_vel_list

    @staticmethod
    def distance(point1, point2):
        diff = point2[0:2] - point1[0:2]
        return np.linalg.norm(diff)
