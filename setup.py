#!/usr/bin/env python
from distutils.core import setup
from setuptools import find_packages


setup(
    name='kiqn',
    version='0.0.1',
    description='IQN layer for Keras',
    author='Adam Wentz',
    author_email='adam@adamwentz.com',
    url='https://github.com/awentzonline/kiqn/',
    packages=find_packages(),
    install_requires=[
        'keras',
        'numpy',
    ],
)
