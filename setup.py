# -*- coding: utf-8 -*-
"""
Created on Wed Aug  5 11:53:11 2020

@author: Adrian
"""

from distutils.core import setup

setup(name='LapTracker',
      version='1.0',
      description='simplified linear assignment problem object tracker',
      author='Adrian Tschan',
      author_email='adrian.tschan@uzh.ch',
      packages=['LapTracker'],
      install_requires=['numpy', 'pandas', 'scipy', 'networkx', 'scikit-image',
                        'progress']
     )