import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import imageio
import platform
import shutil
from matplotlib import image

from math import cos, sin, pi
from pathlib import Path
import inspect
import matplotlib.transforms as mtransforms
from matplotlib.patches import Polygon, Circle


class env_plot:
    def __init__(
            self,
            width=10,
            height=10,
            components=None,
            full=False,
            keep_path=False,
            map_matrix=None,
            offset_x=0,
            offset_y=0,
            **kwargs
    ):

        if components is None:
            components = dict()
        self.fig, self.ax = plt.subplots(figsize=(8, 8), dpi=120)
        # self.fig, self.ax = plt.subplots()

        self.width = width
        self.height = height
        self.offset_x = offset_x
        self.offset_y = offset_y
        self.init_car_img = None

        self.color_list = ['g', 'b', 'r', 'y', 'purple', 'c', 'm', 'gold', 'orange', 'k', 'olive', 'cyan',
                           'pink', 'blueviolet', 'brown', 'skyblue', 'plum', 'darkblue', 'orchid', 'lime']
        self.components = components

        self.keep_path = keep_path
        self.map_matrix = map_matrix

        self.car_plot_list = []
        self.car_line_list = []

        # add all robots
        self.robot_plot_list = []

        self.lidar_line_list = []
        self.car_img_show_list = []
        self.line_list = []

        # for static obstacle text
        self.static_obstacle_list = []

        # add all dynamic obstacles
        self.dyna_obs_plot_list = []

        self.init_plot(**kwargs)

        if full:
            mode = platform.system()
            if mode == "Linux":
                # mng = plt.get_current_fig_manager()
                plt.get_current_fig_manager().full_screen_toggle()
                # mng.resize(*mng.window.maxsize())
                # mng.frame.Maximize(True)

            elif mode == "Windows":
                figManager = plt.get_current_fig_manager()
                figManager.window.showMaximized()

        # plt.rcParams['pdf.fonttype'] = 42
        # plt.rcParams['ps.fonttype'] = 42

    # draw ax
    def init_plot(self, **kwargs):
        """ Init the figure """
        plt.cla()
        self.ax.set_aspect("equal")

        # # no ticks
        # plt.xticks([])
        # plt.yticks([])
        # plt.axis('off')

        # set the text in Times New Roman
        config = {
            "font.family": 'serif',
            "font.serif": ['Times New Roman'],
            "mathtext.fontset": 'stix',
        }
        plt.rcParams.update(config)

        self.ax.set_xlim(self.offset_x, self.offset_x + self.width)
        self.ax.set_ylim(self.offset_y, self.offset_y + self.height)
        # self.ax.legend(loc='upper right')

        # set the label in Times New Roman and size
        label_font = {'family': 'Times New Roman',
                      'weight': 'normal',
                      'size': 20,
                      }
        self.ax.set_xlabel('x (m)', label_font)
        self.ax.set_ylabel("y (m)", label_font)

        # set the tick in Times New Roman and size
        self.ax.tick_params(labelsize=20)
        labels = self.ax.get_xticklabels() + self.ax.get_yticklabels()
        [label.set_fontname('Times New Roman') for label in labels]

        # car image
        current_file_frame = inspect.getfile(inspect.currentframe())
        car_image_path = Path(current_file_frame).parent / "car0.png"
        self.init_car_img = image.imread(str(car_image_path))

        self.draw_static_components(**kwargs)

        # if self.map_matrix is not None:
        #   self.ax.imshow(np.flipud(self.map_matrix.T), cmap='Greys', origin='lower',
        #   extent=[0,self.width,0,self.height])
        #   self.ax.imshow(self.map_matrix.T, cmap='Greys', origin='lower', extent=[0,self.width,0,self.height])

        return self.ax.patches + self.ax.texts + self.ax.artists

    # draw components
    def draw_components(self, **kwargs):

        if self.components["map_matrix"] is not None:
            self.ax.imshow(
                self.components["map_matrix"].T,
                cmap="Greys",
                origin="lower",
                extent=[
                    self.offset_x,
                    self.offset_x + self.width,
                    self.offset_y,
                    self.offset_y + self.height,
                ],
            )

        self.draw_robots(self.components["robots"], **kwargs)
        self.draw_cars(self.components["cars"], **kwargs)
        self.draw_obs_cirs(self.components["obs_circles"], **kwargs)
        self.draw_obs_lines(self.components["obs_lines"], **kwargs)

    def draw_static_components(self, **kwargs):
        """ Draw static components, especially fot static obstacles """
        if self.components["map_matrix"] is not None:
            self.ax.imshow(
                self.components["map_matrix"].T,
                cmap="Greys",
                origin="lower",
                extent=[
                    self.offset_x,
                    self.offset_x + self.width,
                    self.offset_y,
                    self.offset_y + self.height,
                ],
            )

        self.draw_static_obs_cirs(self.components["obs_circles"], **kwargs)
        self.draw_obs_lines(self.components["obs_lines"], **kwargs)
        self.draw_static_obs_polygons(self.components["obs_polygons"], **kwargs)

    def draw_dyna_components(self, **kwargs):
        """ Draw robots and dynamic obstacles """
        robots = self.components.get("robots", None)
        cars = self.components.get("cars", None)
        polygon_robots = self.components.get('polygon_robots', None)
        obs_cirs = self.components.get("obs_circles", None)
        obs_polys = self.components.get('obs_polygons', None)

        if robots is not None:
            self.draw_robots(robots, **kwargs)

        if cars is not None:
            self.draw_cars(cars, **kwargs)

        if polygon_robots is not None:
            self.draw_polygon_robots(polygon_robots, **kwargs)

        if obs_cirs is not None:
            self.draw_dyna_obs_cirs(obs_cirs, **kwargs)

        if obs_polys is not None:
            self.draw_dyna_obs_polygons(obs_polys, **kwargs)

        self.draw_static_obstacle_text()

    def draw_robots(self, robots, **kwargs):
        if len(robots.robot_list) > 1:
            for robot, color in zip(robots.robot_list, self.color_list):
                self.draw_robot(robot, robot_color=color, **kwargs)
        else:
            for robot in robots.robot_list:
                self.draw_robot(robot, **kwargs)

    def draw_cars(self, cars, **kwargs):

        if len(cars.car_list) > 1:
            for car, color in zip(cars.car_list, self.color_list):
                self.draw_car(car, car_color=color, text=True, **kwargs)
        else:
            for car in cars.car_list:
                self.draw_car(car, **kwargs)

    def draw_polygon_robots(self, polygon_robots, **kwargs):

        if len(polygon_robots.robot_list) > 1:
            for robot, color in zip(polygon_robots.robot_list, self.color_list):
                self.draw_polygon_robot(robot, robot_color=color, **kwargs)
        else:
            for robot in polygon_robots.robot_list:
                self.draw_polygon_robot(robot, **kwargs)

    def draw_obs_cirs(self, param, param1):
        pass

    def draw_static_obs_cirs(self, obs_cirs, **kwargs):

        for obs_cir in obs_cirs.obs_cir_list:
            self.draw_static_obs_cir(obs_cir, **kwargs)

    def draw_static_obs_polygons(self, obs_polygons, **kwargs):

        for obs_polygon in obs_polygons.obs_poly_list:
            self.draw_static_obs_polygon(obs_polygon, **kwargs)

    def draw_dyna_obs_cirs(self, obs_cirs, **kwargs):

        for obs_cir in obs_cirs.obs_cir_list:
            self.draw_dyna_obs_cir(obs_cir, **kwargs)

    def draw_dyna_obs_polygons(self, obs_polys, **kwargs):

        for obs_poly in obs_polys.obs_poly_list:
            self.draw_dyna_obs_polygon(obs_poly, **kwargs)

    def draw_obs_lines(self, obs_lines, **kwargs):

        for obs_line in obs_lines.obs_line_states:
            self.ax.plot([obs_line[0], obs_line[2]], [obs_line[1], obs_line[3]], "k-")

    def draw_robot(
            self,
            robot,
            robot_color="g",
            goal_color="r",
            show_lidar=True,
            show_goal=True,
            show_text=True,
            show_traj=False,
            traj_type="-",
            **kwargs
    ):

        x = robot.state[0][0]
        y = robot.state[1][0]

        goal_x = robot.goal[0, 0]
        goal_y = robot.goal[1, 0]

        robot_circle = mpl.patches.Circle(
            xy=(x, y), radius=robot.radius, color=robot_color
        )
        robot_circle.set_zorder(2)
        goal_circle = mpl.patches.Circle(
            xy=(goal_x, goal_y), radius=robot.radius, color=goal_color, alpha=0.5
        )
        goal_circle.set_zorder(1)

        if show_goal:
            self.ax.add_patch(goal_circle)
            if show_text:
                self.ax.text(
                    goal_x + 0.3, goal_y, "g" + str(robot.id), fontsize=12, color="k"
                )
            self.robot_plot_list.append(goal_circle)

        self.ax.add_patch(robot_circle)

        if show_text:
            self.ax.text(x - 0.5, y, "r" + str(robot.id), fontsize=10, color="k")

        self.robot_plot_list.append(robot_circle)

        if show_traj:
            x_list = [robot.previous_state[0, 0], robot.state[0, 0]]
            y_list = [robot.previous_state[1, 0], robot.state[1, 0]]

            self.ax.plot(x_list, y_list, traj_type)

        if robot.lidar is not None and show_lidar:
            for point in robot.lidar.inter_points[:, :]:
                x_value = [x, point[0]]
                y_value = [y, point[1]]

                self.lidar_line_list.append(
                    self.ax.plot(x_value, y_value, color="b", alpha=0.5)
                )

        if robot.mode == "diff":
            theta = robot.state[2][0]
            arrow = mpl.patches.Arrow(
                x, y, 0.5 * cos(theta), 0.5 * sin(theta), width=0.6
            )
            self.ax.add_patch(arrow)
            self.robot_plot_list.append(arrow)

    def draw_car(
            self,
            car,
            goal_color="c",
            goal_l=2,
            text=False,
            show_lidar=True,
            show_traj=False,
            traj_type="-g",
            **kwargs
    ):

        x = car.ang_pos[0, 0]
        y = car.ang_pos[1, 0]
        r_phi = car.state[2, 0]
        # r_phi_ang = 180*r_phi/pi

        # line_rad_f = car.state[3, 0] + car.state[2, 0]
        # line_rad_b = car.state[2, 0]

        gx = car.goal[0, 0]
        gy = car.goal[1, 0]
        gdx = goal_l * cos(car.goal[2, 0])
        gdy = goal_l * sin(car.goal[2, 0])
        # self.car_line_list = []
        self.car_img_show_list = []

        # for i in range(4):

        #     if 0 < i < 3:
        #         wx = car.wheel_pos[0, i]
        #         wy = car.wheel_pos[1, i]

        #         lx0 = wx + line_length * cos(line_rad_f) / 2
        #         ly0 = wy + line_length * sin(line_rad_f) / 2

        #         lx1 = wx - line_length * cos(line_rad_f) / 2
        #         ly1 = wy - line_length * sin(line_rad_f) / 2

        #         self.car_line_list.append(self.ax.plot([lx0, lx1], [ly0, ly1], 'k-'))

        #     else:
        #         wx = car.wheel_pos[0, i]
        #         wy = car.wheel_pos[1, i]

        #         lx0 = wx + line_length * cos(line_rad_b) / 2
        #         ly0 = wy + line_length * sin(line_rad_b) / 2

        #         lx1 = wx - line_length * cos(line_rad_b) / 2
        #         ly1 = wy - line_length * sin(line_rad_b) / 2

        #         self.car_line_list.append(self.ax.plot([lx0, lx1], [ly0, ly1], 'k-'))

        # car_rect = mpl.patches.Rectangle(xy=(x, y), width=car.length, height=car.width, angle=r_phi_ang,
        # edgecolor=car_color, fill=False)
        goal_arrow = mpl.patches.Arrow(x=gx, y=gy, dx=gdx, dy=gdy, color=goal_color)

        # self.car_plot_list.append(car_rect)
        self.car_plot_list.append(goal_arrow)

        # self.ax.add_patch(car_rect)
        self.ax.add_patch(goal_arrow)

        # car image show

        car_img = self.ax.imshow(
            self.init_car_img, extent=[x, x + car.length, y, y + car.width]
        )
        degree = car.state[2, 0] * 180 / pi
        trans_data = (
                mtransforms.Affine2D().rotate_deg_around(x, y, degree) + self.ax.transData
        )
        car_img.set_transform(trans_data)
        self.car_img_show_list.append(car_img)

        if text:
            self.ax.text(x - 0.5, y, "c" + str(car.id), fontsize=10, color="k")
            self.ax.text(
                car.goal[0, 0] + 0.3,
                car.goal[1, 0],
                "cg" + str(car.id),
                fontsize=12,
                color="k",
            )

        if show_traj:
            x_list = [car.previous_state[0, 0], car.state[0, 0]]
            y_list = [car.previous_state[1, 0], car.state[1, 0]]

            self.ax.plot(x_list, y_list, traj_type)

        if car.lidar is not None and show_lidar:
            for point in car.lidar.inter_points[:, :]:
                x_value = [car.state[0, 0], point[0]]
                y_value = [car.state[1, 0], point[1]]

                self.lidar_line_list.append(
                    self.ax.plot(x_value, y_value, color="b", alpha=0.5)
                )

    def draw_polygon_robot(self,
                           robot,
                           robot_color='g',
                           goal_color='r',
                           show_lidar=True,
                           show_goal=True,
                           show_text=True,
                           show_traj=True,
                           traj_type='-',
                           show_margin=False,
                           show_previous=False,
                           **kwargs):

        # first for extended_vertices
        polygon_extended_vertexes = robot.extended_vertexes.T
        # need for shape N * 2
        extended_polygon = mpl.patches.Polygon(xy=polygon_extended_vertexes,
                                               color=robot_color, alpha=0.3, linestyle='--')
        extended_polygon.set_zorder(1)

        # previous vertices
        previous_vertexes = robot.get_previous_vertex()
        previous_polygon = mpl.patches.Polygon(xy=previous_vertexes.T, color=robot_color, alpha=0.5, linestyle=':')
        previous_polygon.set_zorder(1)

        # current vertices
        polygon_vertexes = robot.vertexes.T
        polygon = mpl.patches.Polygon(xy=polygon_vertexes, color=robot_color)
        polygon.set_zorder(2)

        x = robot.state[0, 0]
        y = robot.state[1, 0]

        goal_x = robot.goal[0, 0]
        goal_y = robot.goal[1, 0]
        goal_circle = Circle(xy=(goal_x, goal_y), radius=0.15, color=robot_color, alpha=1.0)
        goal_circle.set_zorder(3)

        if show_goal:
            position_text = 'x'
            self.ax.add_patch(goal_circle)
            if show_text:
                if goal_x <= 5.0:
                    if goal_y <= 5.0:
                        # 0.5 0.8
                        if robot.id == 0:
                            self.ax.text(goal_x - 0.7, goal_y - 0.75, fr'$\rm{position_text}_{{{str(robot.id)}}}^{{goal}}$', fontsize=24, color='k')
                        else:
                            self.ax.text(goal_x - 1.0, goal_y - 0.9, fr'$\rm{position_text}_{{{str(robot.id)}}}^{{goal}}$', fontsize=24, color='k')
                    else:
                        # 0.5 0.5
                        self.ax.text(goal_x - 0.6, goal_y + 0.35, fr'$\rm{position_text}_{{{str(robot.id)}}}^{{goal}}$', fontsize=24, color='k')
                elif goal_x > 5.0:
                    if goal_y < 5.0:
                        # 0.21 0.35
                        self.ax.text(goal_x + 0.3, goal_y - 0.35, fr'$\rm{position_text}_{{{str(robot.id)}}}^{{goal}}$', fontsize=24, color='k')
                    else:
                        # 0.21 0.35
                        self.ax.text(goal_x + 0.3, goal_y + 0.35, fr'$\rm{position_text}_{{{str(robot.id)}}}^{{goal}}$', fontsize=24, color='k')
                # # old display
                # self.ax.text(goal_x - 0.55, goal_y - 0.55, fr'$\rm{position_text}_{{{str(robot.id)}}}^{{goal}}$', fontsize=20, color='k')
            self.robot_plot_list.append(goal_circle)

        self.ax.add_patch(polygon)
        self.robot_plot_list.append(polygon)

        if show_margin:
            self.ax.add_patch(extended_polygon)
            self.robot_plot_list.append(extended_polygon)

        if show_previous:
            self.ax.add_patch(previous_polygon)
            self.robot_plot_list.append(previous_polygon)

        if show_text:
            # 'r' + str(robot.id)
            text = 'R'
            self.ax.text(x - 0.85, y + 0.45, fr'$\rm{text}_{{{robot.id}}}$', fontsize=24, color='k')

        if show_traj:
            x_list = [robot.previous_state[0, 0], robot.state[0, 0]]
            y_list = [robot.previous_state[1, 0], robot.state[1, 0]]
            self.ax.plot(x_list, y_list, linestyle=traj_type, color=robot_color)

        if robot.lidar is not None and show_lidar:
            for point in robot.lidar.inter_points[:, :]:
                x_value = [x, point[0]]
                y_value = [y, point[1]]

                self.lidar_line_list.append(self.ax.plot(x_value, y_value, color='b', alpha=0.5))

        if robot.mode == 'diff' or robot.mode == 'omni':
            theta = robot.state[2][0]
            arrow = mpl.patches.Arrow(x, y, 0.8 * cos(theta), 0.8 * sin(theta), width=0.3)
            self.ax.add_patch(arrow)
            self.robot_plot_list.append(arrow)

    def draw_static_obs_cir(self, obs_cir, obs_cir_color="k", **kwargs):

        if obs_cir.obs_model == "static":
            x = obs_cir.state[0, 0]
            y = obs_cir.state[1, 0]

            obs_circle = mpl.patches.Circle(
                xy=(x, y), radius=obs_cir.radius, color=obs_cir_color
            )
            obs_circle.set_zorder(2)
            self.ax.add_patch(obs_circle)

    def draw_dyna_obs_cir(self, obs_cir, obs_cir_color="k", **kwargs):

        if obs_cir.obs_model != "static":
            x = obs_cir.state[0, 0]
            y = obs_cir.state[1, 0]

            obs_circle = mpl.patches.Circle(
                xy=(x, y), radius=obs_cir.radius, color=obs_cir_color
            )
            obs_circle.set_zorder(2)
            self.ax.add_patch(obs_circle)

            self.dyna_obs_plot_list.append(obs_circle)

    def draw_static_obs_polygon(self, obs_polygon, obs_polygon_color="k", show_margin=False, **kwargs):

        if obs_polygon.obs_model == 'static':
            obs_poly = Polygon(obs_polygon.vertexes.T, color=obs_polygon_color)
            obs_poly.set_zorder(2)
            self.ax.add_patch(obs_poly)

            extended_obs_poly = Polygon(obs_polygon.extended_vertexes.T, color=obs_polygon_color,
                                        linestyle='--', alpha=0.3)
            extended_obs_poly.set_zorder(1)

            if show_margin:
                self.ax.add_patch(extended_obs_poly)

            self.static_obstacle_list.append(obs_polygon)

    def draw_dyna_obs_polygon(self, obs_polygon, obs_polygon_color='k', show_margin=False, show_previous=False,
                              show_traj=True, **kwargs):
        if obs_polygon.obs_model != 'static':
            obs_poly = Polygon(obs_polygon.vertexes.T, color=obs_polygon_color)
            obs_poly.set_zorder(2)

            # previous position
            previous_obs_poly = Polygon(obs_polygon.get_previous_vertexes().T, color=obs_polygon_color,
                                        linestyle=':', alpha=0.5)
            previous_obs_poly.set_zorder(1)

            # extended position
            extended_obs_poly = Polygon(obs_polygon.extended_vertexes.T, color=obs_polygon_color,
                                        linestyle='--', alpha=0.3)
            extended_obs_poly.set_zorder(1)

            self.ax.add_patch(obs_poly)
            self.dyna_obs_plot_list.append(obs_poly)

            if show_margin:
                self.ax.add_patch(extended_obs_poly)
                self.dyna_obs_plot_list.append(extended_obs_poly)

            if show_previous:
                self.ax.add_patch(previous_obs_poly)
                self.dyna_obs_plot_list.append(previous_obs_poly)

            if show_traj:
                x_list = [obs_polygon.previous_state[0, 0], obs_polygon.center_point[0, 0]]
                y_list = [obs_polygon.previous_state[1, 0], obs_polygon.center_point[1, 0]]
                self.ax.plot(x_list, y_list, linestyle=':', marker='.', color=obs_polygon_color, markersize=5)

            obstacle_text = 'O'
            self.ax.text(obs_polygon.center_point[0][0] - 0.25,
                         obs_polygon.center_point[1][0] + 0.45,
                         fr'$\rm{obstacle_text}_{str(obs_polygon.id)}$',
                         fontsize=20,
                         color='k')

    def draw_static_obstacle_text(self):
        obstacle_text = 'O'
        for obs_polygon in self.static_obstacle_list:
            self.ax.text(obs_polygon.center_point[0][0] - 1.1,
                         obs_polygon.center_point[1][0] + 0.7, # 0.8
                         fr'$\rm{obstacle_text}_{str(obs_polygon.id)}$',
                         fontsize=20,
                         color='k')

    # def draw_obs_line_list(self, **kwargs):

    #     for line in self.obs_line_list:
    #         # self.ax.plot(   line[0:2], line[2:4], 'k-')
    #         self.ax.plot( [line[0], line[2]], [line[1], line[3]], 'k-')

    def draw_vector(self, x, y, dx, dy, color="r"):
        arrow = mpl.patches.Arrow(x, y, dx, dy, width=0.2, color=color)
        self.ax.add_patch(arrow)

    def draw_trajectory(
            self,
            traj,
            style="g-",
            label="line",
            show_direction=False,
            refresh=False,
            markersize=2,
    ):
        path_x_list = []
        path_y_list = []
        u_list = []
        y_list = []

        if isinstance(traj, list):
            # traj is a set of point in shape (2, 1)
            path_x_list = [p[0, 0] for p in traj]
            path_y_list = [p[1, 0] for p in traj]

        elif isinstance(traj, np.ndarray):
            # raw*column: points * num
            # every element in (2,)
            path_x_list = [p[0] for p in traj.T]
            path_y_list = [p[1] for p in traj.T]

        line = self.ax.plot(
            path_x_list, path_y_list, style, label=label, markersize=markersize
        )

        if show_direction:

            if isinstance(traj, list):
                u_list = [cos(p[2, 0]) for p in traj]
                y_list = [sin(p[2, 0]) for p in traj]
            elif isinstance(traj, np.ndarray):
                u_list = [cos(p[2]) for p in traj.T]
                y_list = [sin(p[2]) for p in traj.T]

            self.ax.quiver(path_x_list, path_y_list, u_list, y_list)

        if refresh:
            self.line_list.append(line)

    def draw_point(self, point, label="point", markersize=2, color="k"):

        point = self.ax.plot(
            point[0],
            point[1],
            marker="o",
            markersize=markersize,
            color=color,
            label=label,
        )
        # self.ax.legend()
        return point

    def com_cla(self):
        # Dynamic clearing displays dynamic effects
        # self.ax.patches = []
        self.ax.texts.clear()

        # for all robots: circular and polytopic robots
        for robot_plot in self.robot_plot_list:
            robot_plot.remove()

        for car_plot in self.car_plot_list:
            car_plot.remove()

        # for line in self.car_line_list:
        #     line.pop(0).remove()
        for line in self.line_list:
            line.pop(0).remove()

        for lidar_line in self.lidar_line_list:
            lidar_line.pop(0).remove()

        for car_img in self.car_img_show_list:
            car_img.remove()

        for obs in self.dyna_obs_plot_list:
            obs.remove()

        self.car_plot_list = []
        self.robot_plot_list = []
        self.lidar_line_list = []
        self.car_img_show_list = []
        self.line_list = []
        self.dyna_obs_plot_list = []

    # animation method 1
    def animate(self):

        self.draw_robot_diff_list()

        return self.ax.patches + self.ax.texts + self.ax.artists

    def show_ani(self):
        ani = animation.FuncAnimation(
            self.fig,
            self.animate,
            init_func=self.init_plot,
            interval=100,
            blit=True,
            frames=100,
            save_count=100,
        )
        plt.show()

    def save_ani(self, name="animation"):
        ani = animation.FuncAnimation(
            self.fig,
            self.animate,
            init_func=self.init_plot,
            interval=1,
            blit=False,
            save_count=300,
        )
        ani.save(name + ".gif", writer="pillow")

    # # animation method 2
    def save_gif_figure(self, path, i, fig_format="png"):

        if path.exists():
            order = str(i).zfill(3)
            # plt.savefig(str(path) + "/" + order + "." + fig_format, format=fig_format)
            plt.savefig(str(path) + '/' + order + '.' + fig_format, format=fig_format, bbox_inches='tight',
                        pad_inches=0.0)
        else:
            path.mkdir()
            order = str(i).zfill(3)
            # plt.savefig(str(path) + "/" + order + "." + fig_format, format=fig_format)
            plt.savefig(str(path) + '/' + order + '.' + fig_format, format=fig_format, bbox_inches='tight',
                        pad_inches=0.0)

    def create_animate(
            self, image_path, ani_path, ani_name="animated", keep_len=30, rm_fig_path=True
    ):
        """ Get the gif from the figure """
        if not ani_path.exists():
            ani_path.mkdir()

        images = list(image_path.glob("*.png"))
        images.sort()
        image_list = []
        for i, file_name in enumerate(images):

            if i == 0:
                continue

            image_list.append(imageio.imread(file_name))
            if i == len(images) - 1:
                for j in range(keep_len):
                    image_list.append(imageio.imread(file_name))

        imageio.mimsave(str(ani_path) + "/" + ani_name + ".gif", image_list)
        print("Create animation successfully")

        # delete all the figures when successfully generating the gif
        if rm_fig_path:
            shutil.rmtree(image_path)

    # old
    def point_arrow_plot(self, point, length=0.5, width=0.3, color="r"):

        px = point[0, 0]
        py = point[1, 0]
        theta = point[2, 0]

        pdx = length * cos(theta)
        pdy = length * sin(theta)

        point_arrow = mpl.patches.Arrow(
            x=px, y=py, dx=pdx, dy=pdy, color=color, width=width
        )

        self.ax.add_patch(point_arrow)

    def point_list_arrow_plot(self, point_list=None, length=0.5, width=0.3, color="r"):

        if point_list is None:
            point_list = []
        for point in point_list:
            self.point_arrow_plot(point, length=length, width=width, color=color)

    def point_plot(self, point, markersize=2, color="k"):

        if isinstance(point, tuple):
            x = point[0]
            y = point[1]
        else:
            x = point[0, 0]
            y = point[1, 0]

        self.ax.plot([x], [y], marker="o", markersize=markersize, color=color)

    # plt
    def cla(self):
        self.ax.cla()

    def pause(self, time=0.001):
        plt.pause(time)

    def show(self):
        plt.show()

    # def draw_start(self, start):
    #     self.ax.plot(start[0, 0], start[1, 0], 'rx')

    # def plot_trajectory(self, robot, num_estimator, label_name = [''], color_line=['b-']):

    #     self.ax.plot(robot.state_storage_x, robot.state_storage_y, 'g-', label='trajectory')

    #     for i in range(num_estimator):
    #         self.ax.plot(robot.estimator_storage_x[i], robot.estimator_storage_y[i], color_line[i],
    #         label = label_name[i])

    #     self.ax.legend()

    # def plot_pre_tra(self, pre_traj):
    #     list_x = []
    #     list_y = []

    #     if pre_traj != None:
    #         for pre in pre_traj:
    #             list_x.append(pre[0, 0])
    #             list_y.append(pre[1, 0])

    #         self.ax.plot(list_x, list_y, '-b')

    # def draw_path(self, path_x, path_y, line='g-'):
    #     self.ax.plot(path_x, path_y, 'g-')

    def draw_robot_diff_list(self):
        pass
