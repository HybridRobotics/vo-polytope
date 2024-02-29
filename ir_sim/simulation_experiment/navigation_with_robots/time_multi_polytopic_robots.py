import sys
import time
import json
from pathlib import Path

root_path = '/home/hjh/ir-sim'
sys.path.append(root_path)
from ir_sim.env import env_base

world_name = 'robot_world.yaml'
env = env_base(world_name=world_name, plot=True)
approaches = ['polytope_rvo', 'circle_rvo']
travel_time = {'polytope_rvo': [], 'circle_rvo': []}

image_path = Path(__file__).parent / 'image'
gif_path = Path(__file__).parent / 'gif'

file_path = Path(__file__).parent
if file_path.exists():
    pass
else:
    file_path.mkdir()
    print(file_path)

for approach in approaches:
    print('current approach:', approach)
    env.reset_polygon_robot(reset_mode=0)

    for i in range(500):
        env.obs_polys_step()
        
        start_time = time.time()
        des_vel_list = env.get_vo_list_polygon(approach)
        travel_time[approach].append(time.time() - start_time)

        env.polygon_robot_step(des_vel_list, vel_type='omni', stop=True)
        env.collision_check()

        # env.save_fig(image_path, i)
        env.render()
        if env.all_stop_polygon():
            break


# env.show()
# storage the time
file_name = str(file_path/'travel_time_8.json')
with open(file_name, 'w') as f:
    json.dump(travel_time, f)

