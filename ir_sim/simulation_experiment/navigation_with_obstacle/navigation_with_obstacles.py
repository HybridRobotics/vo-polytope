import sys
from pathlib import Path
from ir_sim.env import env_base

# # only static obstacles
# world_name = 'static_obstacle.yaml'

# # only dynamic obstacles
# world_name = 'dynamic_obstacle.yaml'

# both static and dynamic obstacles
world_name = 'hybrid_obstacle.yaml'
env = env_base(world_name=world_name, plot=True)

image_path = Path(__file__).parent / 'image'
gif_path = Path(__file__).parent / 'gif'

for i in range(500):
    env.obs_polys_step()

    des_vel_list = env.get_vo_list_polygon('polytope_vo')
    env.polygon_robot_step(des_vel_list, vel_type='omni', stop=True)
    env.collision_check()

    # env.save_fig(image_path, i)
    env.render()
    if env.all_stop_polygon():
        break


env.show()

