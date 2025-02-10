from setuptools import setup, find_packages

setup(
    name='mars_rover_project',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'stable-baselines3',
        'gym[all]',
        'pybullet'
    ],
)
