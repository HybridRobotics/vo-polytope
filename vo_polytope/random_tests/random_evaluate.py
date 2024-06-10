from pathlib import Path
import time
import json
import sys

root_path = '/home/hjh/ir-sim/vo_polytope'
sys.path.append(root_path)
from vo_polytope.env import env_base

world_name = 'polygon_world.yaml'
env = env_base(world_name=world_name, plot=True)
approaches = ['polytope_vo', 'polytope_rvo', 'polytope_hrvo', 'circle_vo', 'circle_rvo', 'circle_hrvo']

# storage
dead_lock = {'polytope_vo': 0, 'polytope_rvo': 0, 'polytope_hrvo': 0, 'circle_vo': 0, 'circle_rvo': 0, 'circle_hrvo': 0}
success_rate = {'polytope_vo': 0, 'polytope_rvo': 0, 'polytope_hrvo': 0, 'circle_vo': 0, 'circle_rvo': 0,
                'circle_hrvo': 0}
travel_distance = {'polytope_vo': [], 'polytope_rvo': [], 'polytope_hrvo': [], 'circle_vo': [], 'circle_rvo': [],
                   'circle_hrvo': []}
travel_time = {'polytope_vo': [], 'polytope_rvo': [], 'polytope_hrvo': [], 'circle_vo': [], 'circle_rvo': [],
               'circle_hrvo': []}
random_index = {'polytope_vo': [], 'polytope_rvo': [], 'polytope_hrvo': [], 'circle_vo': [], 'circle_rvo': [],
                'circle_hrvo': []}

file_path = Path(__file__).parent
if file_path.exists():
    pass
else:
    file_path.mkdir()
    print(file_path)


for i in range(100):
    print('current:', i)
    # ramdom experiment for 100 times
    if i == 0:
        env.reset_polygon_robot(reset_mode=0)  # first enter
    else:
        env.reset_polygon_robot(reset_mode=2)  # reset all the status of robots

    # test for different appraoches
    for approach in approaches:
        print('current approach:', approach)
        env.reset_polygon_robot(reset_mode=0)
        time_start = time.time()

        for j in range(300):
            env.obs_polys_step()

            des_vel_list = env.get_vo_list_polygon(approach)
            env.polygon_robot_step(des_vel_list, vel_type='omni', stop=True)
            env.collision_check()

            # env.render()
            if env.all_stop_polygon():
                break

        time_end = time.time()
        time_sum = time_end - time_start

        if env.all_stop_polygon():
            if env.all_arrive_polygon():
                # only success then add
                success_rate[approach] = success_rate[approach] + 1
                travel_distance[approach].append(env.get_polygon_robot_travel_distance())
                travel_time[approach].append(time_sum)
                random_index[approach].append(i)
        # Neither to the target position, nor stopped by the collision
        else:
            dead_lock[approach] = dead_lock[approach] + 1


# storage the data
file_name = str(file_path/'success.json')
with open(file_name, 'w') as f:
    json.dump(success_rate, f)

file_name = str(file_path/'travel_time.json')
with open(file_name, 'w') as f:
    json.dump(travel_time, f)

file_name = str(file_path/'travel_distance.json')
with open(file_name, 'w') as f:
    json.dump(travel_distance, f)

file_name = str(file_path/'index.json')
with open(file_name, 'w') as f:
    json.dump(random_index, f)

file_name = str(file_path/'deadlock.json')
with open(file_name, 'w') as f:
    json.dump(dead_lock, f)


