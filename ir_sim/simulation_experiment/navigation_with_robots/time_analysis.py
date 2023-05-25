import json
approaches = ['polytope_rvo', 'circle_rvo']

file_name = 'travel_time_8.json'
with open(file_name) as f:
    travel_time = json.load(f)


for approach in approaches:
    interval = len(travel_time[approach])
    sum = 0
    for t in travel_time[approach]:
        sum = sum + t

    print(approach + ' interval:', interval)
    print(approach + ' average_time:', sum / interval)




