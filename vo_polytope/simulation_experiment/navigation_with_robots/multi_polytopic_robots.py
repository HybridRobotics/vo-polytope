import sys
from pathlib import Path

# need to change according your computer, add absolute path: 'xx/vo_polytope'
root_path = '/home/hjh/ir-sim/vo_polytope'
sys.path.append(root_path)
from vo_polytope.env import env_base

world_name = 'robot_world.yaml'
env = env_base(world_name=world_name, plot=True)

image_path = Path(__file__).parent / 'image'
gif_path = Path(__file__).parent / 'gif'

for i in range(500):
    env.obs_polys_step()

    des_vel_list = env.get_vo_list_polygon('polytope_hrvo')
  
    env.polygon_robot_step(des_vel_list, vel_type='omni', stop=True)
    env.collision_check()
    
    env.save_fig(image_path, i)
    env.render()
    if env.all_stop_polygon():
        break

env.show()

