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
cd simulation_experiment/naviagtion_with_robots
python multi_polytopic_robots.py
```

### Navigation with robots and obstacles
```
cd simulation_experiment/naviagtion_with_obstacles
python naviagtion_with_obstacles.py
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
  doi={10.1109/LRA.2023.3269295}}


```