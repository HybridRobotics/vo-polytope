from setuptools import setup
import sys

setup(
    name="ir_sim",
    py_modules=["ir_sim"],
    version="1.0",
    install_requires=[
        "matplotlib",
        "numpy",
        "scipy",
        "pyyaml",
        "pynput",
        "imageio",
        "pathlib",
    ],
    description="A simple 2D simulator for the intelligent mobile robots and polytopic robots",
    author="Huang Jihao",
)
