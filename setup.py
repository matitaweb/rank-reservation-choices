# -*- coding: utf-8 -*-

# Learn more: https://github.com/kennethreitz/setup.py

from setuptools import setup, find_packages




setup(
    name='rankchoices',
    version='0.1.0',
    description='Spark experipent to rank reservation by customer behaviour',
    long_description="",
    author='Mattia Chiarini',
    author_email='Mattia Chiarini',
    url='https://github.com/matitaweb/rank-reservation-choices',
    license="",
    packages=find_packages(exclude=('tests', 'docs'))
)

