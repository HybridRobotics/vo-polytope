from ir_sim.env import env_base

world_name = 'polygon_world.yaml'
env = env_base(world_name=world_name, plot=True)

for i in range(300):
    des_vel_list = [robot.cal_des_vel() for robot in env.components['polygon_robots'].robot_list]

    env.polygon_robot_step(des_vel_list)
    env.render()
    env.collision_check()

env.show()
