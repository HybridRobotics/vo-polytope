# Velocity Obstacle for Polytope
The source code of the paper "Velocity Obstacle for Polytopic Collision Avoidance for Distributed Multi-Robot Systems" [**RA-Letter**] 

| **[`PDF_IEEE`](https://ieeexplore.ieee.org/document/10106436?source=authoralert)** | **[`PDF_Arxiv`](https://arxiv.org/pdf/2304.07954.pdf)** | **[`Video_Youtube`](https://www.youtube.com/watch?v=YT9aObT2VAo)** | 

## Prerequisite
- numpy
- cvxpy
- matplotlib


## Installation 
```
git clone https://github.com/junzengx14/ir-sim
cd ir-sim
pip install -e .  
```

## Run examples
### Navigation with robots
```
python ir_sim/simulation_experiment/naviagtion_with_robots/multi_polytopic_robots.py
```
You can change the number or shape of robots in 'robot_world.yaml'.

### Navigation with robots and obstacles
```
python ir_sim/simulation_experiment/naviagtion_with_obstacles/naviagtion_with_obstacles.py
```
You can switch the yaml file to see different scenarios, such as 'hybrid_obstacle.yaml' (include both dynamic and static obstacles) or 'dynamic_obstacle.yaml' (only include dynamic obstacles).

### Compare different VO
```
python ir_sim/simulation_experiment/evaluation/evaluate_scenario.py 
```
You may change some codes to see different results.
```
des_vel_list = env.get_vo_list_polygon('circle_vo') # polytope_vo
```

### Random tests
Get random data and analyze the datas
```
python ir_sim/Random_test/evaluation/random_evaluate.py 
python ir_sim/Random_test/evaluation/data_analysis.py
```
You can change in 'polygon_world.yaml' to get different results. (such as size)

## Save multimedia file
If you want to save multimedia file, you first need to save the fig, and then call 'image_to_mp4.py', for example:
```
python ir_sim/simulation_experiment/naviagtion_with_robots/multi_polytopic_robots.py
python ir_sim/simulation_experiment/naviagtion_with_robots/image_to_mp4.py
```


## Contact

Huang Jihao (jihaoh@zju.edu.cn)  
Zeng Jun (zengjunsjtu@berkeley.edu)

## Citation

If you find this code or paper is helpful, you can **star** this repository and cite our paper by the following **BibTeX** entry:

```
@ARTICLE{10106436,
  author={Huang, Jihao and Zeng, Jun and Chi, Xuemin and Sreenath, Koushil and Liu, Zhitao and Su, Hongye},
  journal={IEEE Robotics and Automation Letters}, 
  title={Velocity Obstacle for Polytopic Collision Avoidance for Distributed Multi-Robot Systems}, 
  year={2023},
  volume={8},
  number={6},
  pages={3502-3509},
  doi={10.1109/LRA.2023.3269295}
}


```