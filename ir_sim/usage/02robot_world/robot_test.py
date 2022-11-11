import sys
import yaml

world_name = 'robot_world.yaml'
world_name = sys.path[0] + '/' + world_name
with open(world_name) as file:
    com_list = yaml.load(file, Loader=yaml.FullLoader)

print(com_list)
world_args = com_list['world']
print(world_args)
robots_args = com_list.get('robots', dict())
print(robots_args)

