import json
import numpy as np

robot_num = 8
approaches = ['polytope_vo', 'polytope_rvo', 'polytope_hrvo', 'circle_vo', 'circle_rvo', 'circle_hrvo']
average_time = {'polytope_vo': 0, 'polytope_rvo': 0, 'polytope_hrvo': 0, 'circle_vo': 0, 'circle_rvo': 0,
                'circle_hrvo': 0}
common_travel_distance = {'polytope_vo': [], 'polytope_rvo': [], 'polytope_hrvo': [], 'circle_vo': [], 'circle_rvo': [],
                          'circle_hrvo': []}
mean_travel_distance = {'polytope_vo': [], 'polytope_rvo': [], 'polytope_hrvo': [], 'circle_vo': [], 'circle_rvo': [],
                        'circle_hrvo': []}
std_travel_distance = {'polytope_vo': [], 'polytope_rvo': [], 'polytope_hrvo': [], 'circle_vo': [], 'circle_rvo': [],
                       'circle_hrvo': []}

file_name = 'data/data_analysis1.2/travel_time.json'
with open(file_name) as f:
    travel_time = json.load(f)

file_name = 'data/data_analysis1.2/index.json'
with open(file_name) as f:
    index = json.load(f)

file_name = 'data/data_analysis1.2/travel_distance.json'
with open(file_name) as f:
    travel_distance = json.load(f)


def duplicate_removal(approaches_index):
    """ extract the common parts """
    result = approaches_index[approaches[0]]
    for approach in approaches:
        if approach == approaches[0]:
            continue
        else:
            result = set(result).intersection(set(approaches_index[approach]))
            result = list(result)

    return result


def calculate_average_time(index_common, approaches_index, time):
    """ calculate the average time for different approach """
    for approach in approaches:
        # total sum
        for i in range(len(approaches_index[approach])):
            if approaches_index[approach][i] in index_common:
                average_time[approach] = average_time[approach] + time[approach][i]

        # average
        average_time[approach] = average_time[approach] / len(index_common)


def extract_common_travel_distance(index_common, approaches_index, distance):
    """ extract the common travel_distance for different approach """
    for approach in approaches:
        for i in range(len(approaches_index[approach])):
            if approaches_index[approach][i] in index_common:
                common_travel_distance[approach].append(distance[approach][i])


def calculate_mean_var_of_distance(index_common):
    """ calculate the mean and variance """
    for approach in approaches:
        # number of robot
        for i in range(robot_num):
            data = []
            # common index
            for j in range(len(index_common)):
                data.append(common_travel_distance[approach][j][i])
            mean = np.around(np.mean(data), 3)
            std = np.around(np.std(data), 3)

            mean_travel_distance[approach].append(mean)
            std_travel_distance[approach].append(std)


# get common index
common_index = duplicate_removal(index)
print(len(common_index))
extract_common_travel_distance(common_index, index, travel_distance)
calculate_mean_var_of_distance(common_index)

for app in approaches:
    print(mean_travel_distance[app])
    print(std_travel_distance[app])
