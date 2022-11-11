import numpy as np
from ir_sim.world import obs_polygon


class env_obs_poly:
    def __init__(
        self, obs_poly_class=obs_polygon, step_time=0.1, vertex_list=None, obs_poly_num=1, goal_list=None,
            obs_model=None, **kwargs
    ):
        if vertex_list is None:
            vertex_list = []
        if goal_list is None:
            goal_list = [[0, 0] for _ in range(obs_poly_num)]
        if obs_model is None:
            obs_model = ['static' for _ in range(obs_poly_num)]

        # initial all the obs_poly
        self.obs_poly_list = []
        for i in range(obs_poly_num):
            obs_poly = obs_poly_class(vertex=vertex_list[i], step_time=step_time, goal=goal_list[i],
                                      obs_model=obs_model[i], **kwargs)
            self.obs_poly_list.append(obs_poly)
