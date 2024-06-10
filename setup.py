from setuptools import setup
import sys

setup(
    name="vo_polytope",
    py_modules=["vo_polytope"],
    version="1.0",
    install_requires=[
        "matplotlib",
        "numpy",
        "scipy",
        "pyyaml",
        "pynput",
        "imageio",
        "pathlib",
        "cvxpy",
    ],
    description="Velocity Obstacle for Polytopes",
    author="Huang Jihao",
)
