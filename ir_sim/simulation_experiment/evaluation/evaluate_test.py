from ir_sim.env import env_base
from pathlib import Path
import time
import json

world_name = 'polygon_world.yaml'
env = env_base(world_name=world_name, plot=True)
approaches = ['polytope_vo', 'polytope_rvo', 'polytope_hrvo', 'circle_vo', 'circle_rvo', 'circle_hrvo']
travel_time = {'polytope_vo': [], 'polytope_rvo': [], 'polytope_hrvo': [], 'circle_vo': [], 'circle_rvo': [],
               'circle_hrvo': []}

image_path = Path(__file__).parent / 'image'
gif_path = Path(__file__).parent / 'gif'

for approach in approaches:
    print('current approach:', approach)
    env.reset_polygon_robot(reset_mode=0)

    time_start = time.time()
    for i in range(300):
        des_vel_list = env.get_vo_list_polygon(approach=approach)
        env.polygon_robot_step(des_vel_list, vel_type='omni', stop=True)

        env.collision_check()
        env.render()
        if env.all_stop_polygon():
            break
    time_end = time.time()
    time_sum = time_end - time_start
    if env.all_stop_polygon():
        if env.all_arrive_polygon():
            travel_time[approach].append(time_sum)

file_name = 'data_analysis0.5/travel_time.json'
with open(file_name, 'w') as f:
    json.dump(travel_time, f)

