import sys
import time
from pathlib import Path

root_path = '/home/hjh/ir-sim/vo_polytope'
sys.path.append(root_path)
from vo_polytope.env import env_base


world_name = 'coverphoto.yaml'
env = env_base(world_name=world_name, plot=True)

image_path = Path(__file__).parent / 'image'
gif_path = Path(__file__).parent / 'gif'

# commit: you could adjust the initial orientation of the robot in the code to make the action more reasonable
for i in range(500):
    start = time.time()
    env.obs_polys_step()

    des_vel_list = env.get_vo_list_polygon('polytope_rvo')

    env.polygon_robot_step(des_vel_list, vel_type='omni', stop=True)
    env.collision_check()

    env.save_fig(image_path, i)
    env.render()
    if env.all_stop_polygon():
        break

    end = time.time()

# env.show()
# env.save_ani(image_path, gif_path, 'two robot rvo')

