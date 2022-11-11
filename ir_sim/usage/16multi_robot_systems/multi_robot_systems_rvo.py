from ir_sim.env import env_base
from pathlib import Path

world_name = 'multi_robot_systems.yaml'
env = env_base(world_name=world_name, plot=True)
image_path = Path(__file__).parent / 'image'
gif_path = Path(__file__).parent / 'gif'

for i in range(300):
    des_vel_list = env.get_robot_rvo_list()

    env.robot_step(des_vel_list, vel_type='omni', stop=True)
    # env.save_fig(image_path, i)
    env.render()

    env.collision_check()
    if env.all_stop_circle():
        break

print('end!')
env.show()

# env.save_ani(image_path, gif_path, 'two robot rvo')
