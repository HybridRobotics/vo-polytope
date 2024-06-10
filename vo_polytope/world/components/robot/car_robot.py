import numpy as np
from math import pi, sin, cos, tan, atan2
from vo_polytope.world import motion_ackermann, lidar2d
from collections import namedtuple
from vo_polytope.util import collision_cir_seg, collision_seg_matrix, collision_seg_seg


class car_robot:
    def __init__(
        self,
        index=0,
        shape=None,
        init_state=np.zeros((4, 1)),
        goal=np.zeros((3, 1)),
        goal_threshold=0.2,
        limit=None,
        psi_limit=pi / 4,
        step_time=0.1,
        **kwargs
    ):

        # state: 0, x
        #        1, y
        #        2, phi, heading direction
        #        3, psi, steering angle
        # shape: length, width, wheelbase, wheelbase_w
        # limit: vel_limit, vel_ang_limit

        if limit is None:
            limit = [2, 2]
        if shape is None:
            shape = [1.5, 1, 1, 1]

        self.ang_pos = None
        self.init_ang_pos = None
        self.G = None
        self.g = None

        if isinstance(init_state, list):
            init_state = np.array(init_state, ndmin=2).T

        if isinstance(goal, list):
            goal = np.array(goal, ndmin=2).T

        self.id = index

        self.shape = shape
        self.length = shape[0]
        self.width = shape[1]
        self.wheelbase = shape[2]
        self.wheelbase_w = shape[3]

        self.v_l = limit[0]
        self.w_l = limit[1]
        self.psi_limit = psi_limit
        # the min turning radius
        self.min_radius = self.wheelbase / tan(psi_limit)

        self.init_state = init_state
        self.state = init_state
        # init car model
        self.init_angular_pos()
        # origin of the coordinates
        self.init_matrix_model()

        self.goal = goal
        self.goal_th = goal_threshold
        self.previous_state = init_state
        self.vel = np.zeros((2, 1))

        self.arrive_flag = False
        self.collision_flag = False

        self.step_time = step_time

        self.state_list = []

        lidar_args = kwargs.get("lidar2d", None)

        if lidar_args is not None:
            self.lidar = lidar2d(**lidar_args)
        else:
            self.lidar = None

    # for loop
    def move_forward(self, vel=np.zeros((2, 1)), stop=True, keep=False, **kwargs):

        if isinstance(vel, list):
            vel = np.array(vel, ndmin=2).T
        if vel.shape == (2,):
            vel = vel[:, np.newaxis]

        assert vel.shape == (2, 1)
        if stop:
            if self.arrive_flag or self.collision_flag:
                vel = np.zeros((2, 1))

        if keep:
            self.state_list.append(self.state)

        self.previous_state = self.state
        self.vel = np.clip(
            vel,
            np.array([[-self.v_l], [-self.w_l]]),
            np.array([[self.v_l], [self.w_l]]),
        )
        self.state = motion_ackermann(
            self.state,
            self.wheelbase,
            self.vel,
            self.psi_limit,
            self.step_time,
            **kwargs
        )
        self.angular_pos()

    def update_state(self, state):
        self.state = state
        self.angular_pos()

    def angular_pos(self):
        """ Get the four point's coordinate in ground axis """
        # coordinates transform
        # get the four vertex point's position through rotation and translation
        # from the car coodinate axis to the ground axis
        # [[cos(theta), -sin(theta)], [sin(theta), cos(theta)]]
        # from the ground coodinate axis to the car axis
        # [[cos(theta), sin(theta)], [-sin(theta), cos(theta)]]
        rotation_matrix = np.array(
            [
                [cos(self.state[2, 0]), -sin(self.state[2, 0])],
                [sin(self.state[2, 0]), cos(self.state[2, 0])],
            ]
        )
        transition_matrix = self.state[0:2, 0:1]

        self.ang_pos = rotation_matrix @ self.init_ang_pos + transition_matrix

    # initialize
    def init_angular_pos(self):
        """ Initialize the four vertex point's position of the car """
        # The coordinate system is established with the rear axis as the origin center (length: x, width: y )
        # car point 4 3 with anticlockwise
        #           1 2
        init_x = (self.length - self.wheelbase) / 2
        init_y = self.width / 2

        car_point1 = np.array([[-init_x], [-init_y]])
        car_point2 = np.array([[-init_x + self.length], [-init_y]])
        car_point3 = np.array([[-init_x + self.length], [-init_y + self.width]])
        car_point4 = np.array([[-init_x], [-init_y + self.width]])

        # then can get the origin point of the four vertex of the car
        self.init_ang_pos = np.column_stack(
            (car_point1, car_point2, car_point3, car_point4)
        )
        self.ang_pos = self.init_ang_pos

    def init_matrix_model(self):
        """ Init the matrix of the rectangle """
        self.G = np.zeros((4, 2))  # 4 * 2
        self.g = np.zeros((4, 1))  # 4 * 1

        for i in range(4):

            if i + 1 < 4:
                pre_point = self.init_ang_pos[:, i]
                next_point = self.init_ang_pos[:, i + 1]
            else:
                pre_point = self.init_ang_pos[:, i]
                next_point = self.init_ang_pos[:, 0]

            diff = next_point - pre_point

            # next.y - pre.y
            a = diff[1]
            # pre.x - next.x
            b = -diff[0]
            # next.y * pre.x - next.x * pre.y
            c = a * pre_point[0] + b * pre_point[1]

            self.G[i, 0] = a
            self.G[i, 1] = b
            self.g[i, 0] = c

        return self.G, self.g

    def get_trans_matrix(self):
        """ Get the translation and rotation matrix """
        # from the car coodinate axis to the ground axis
        # [[cos(theta), -sin(theta)], [sin(theta), cos(theta)]]
        rot = np.array(
            [
                [cos(self.state[2, 0]), -sin(self.state[2, 0])],
                [sin(self.state[2, 0]), cos(self.state[2, 0])],
            ]
        )
        trans = self.state[0:2, 0:1]

        return rot, trans

    def inside(self, point):

        rot = np.array(
            [
                [cos(self.state[2, 0]), -sin(self.state[2, 0])],
                [sin(self.state[2, 0]), cos(self.state[2, 0])],
            ]
        )
        trans = self.state[0:2, 0:1]

        # Get the point's coordinate in the car coordinate axis: transpoint, after translation and rotation
        trans_point = np.linalg.inv(rot) @ (point - trans)
        return (self.G @ trans_point <= self.g).all()

    def arrive(self):
        """ Judge if reach the end point """
        dis, radian = car_robot.relative(self.state[0:2], self.goal[0:2])

        if dis < self.goal_th:
            self.arrive_flag = True
            return True
        else:
            self.arrive_flag = False
            return False

    def cal_des_vel(self, tolerance=0.12):
        """ Calculate the diff vel to the destination for car """
        dis, radian = car_robot.relative(self.state[0:2], self.goal[0:2])
        car_radian = self.state[2, 0] + self.state[3, 0]

        v_max = self.v_l
        w_max = self.w_l

        diff_radian = car_robot.wraptopi(radian - car_radian)

        if diff_radian > tolerance:
            w_opti = w_max
        elif diff_radian < -tolerance:
            w_opti = -w_max
        else:
            w_opti = 0

        if dis < self.goal_th:
            v_opti = 0
            w_opti = 0
        else:
            v_opti = v_max * cos(diff_radian)

            if v_opti < 0:
                v_opti = 0

        return np.array([[v_opti], [w_opti]])

    def collision_check(self, components):
        """ Return True if the car robot has collisions with other robots or obstacles """
        circle = namedtuple("circle", "x y r")
        point = namedtuple("point", "x y")

        # the four sides of car, storage the four vertices by col
        segment1 = [
            point(self.ang_pos[0, 0], self.ang_pos[1, 0]),
            point(self.ang_pos[0, 1], self.ang_pos[1, 1]),
        ]
        segment2 = [
            point(self.ang_pos[0, 1], self.ang_pos[1, 1]),
            point(self.ang_pos[0, 2], self.ang_pos[1, 2]),
        ]
        segment3 = [
            point(self.ang_pos[0, 2], self.ang_pos[1, 2]),
            point(self.ang_pos[0, 3], self.ang_pos[1, 3]),
        ]
        segment4 = [
            point(self.ang_pos[0, 3], self.ang_pos[1, 3]),
            point(self.ang_pos[0, 0], self.ang_pos[1, 0]),
        ]

        segment_list = [segment1, segment2, segment3, segment4]

        # check collision with obstacles
        for obs_cir in components["obs_circles"].obs_cir_list:
            temp_circle = circle(
                obs_cir.state[0, 0], obs_cir.state[1, 0], obs_cir.radius
            )
            for segment in segment_list:
                if collision_cir_seg(temp_circle, segment):
                    self.collision_flag = True
                    print("collisions with obstacles")
                    return True

        # check collision with map
        for segment in segment_list:
            if collision_seg_matrix(
                segment,
                components["map_matrix"],
                components["xy_reso"],
                components["offset"],
            ):
                self.collision_flag = True
                print("collisions between obstacle map")
                return True

        # check collision with line obstacles:
        for line in components["obs_lines"].obs_line_states:
            seg1 = [point(line[0], line[1]), point(line[2], line[3])]
            for seg2 in segment_list:
                if collision_seg_seg(seg1, seg2):
                    print("collisions with line obstacle")
                    return True

        for polygon in components["obs_polygons"].obs_poly_list:
            for edge in polygon.edge_list:
                seg1 = [point(edge[0], edge[1]), point(edge[2], edge[3])]
                for seg2 in segment_list:
                    if collision_seg_seg(seg1, seg2):
                        print("collisions with polygon obstacle")
                        return True

    def cal_lidar_range(self, components):
        if self.lidar is not None:
            self.lidar.cal_range(self.state, components)

    @staticmethod
    def relative(state1, state2):

        dif = state2[0:2] - state1[0:2]

        dis = np.linalg.norm(dif)
        radian = atan2(dif[1, 0], dif[0, 0])

        return dis, radian

    @staticmethod
    def wraptopi(radian):

        if radian > pi:
            radian = radian - 2 * pi
        elif radian < -pi:
            radian = radian + 2 * pi

        return radian
