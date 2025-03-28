#!/usr/bin/env python
from distutils.core import setup

setup(
    name='geotypes',
    version="0.1",
    description='A set of useful utilities for working with geospatial data. Makes it simpler to do common operations.',
    author='JacksonShields',
    author_email='jacksonhshields@gmail.com',
    url='http://github.com/jacksonhshields/situ.git',
    packages=[
        'geotypes'
    ],
    package_data={
    },
    scripts=[
        ],
    install_requires=[
        'pandas',
        'numpy',
        'matplotlib',
        'geopandas',
        'affine',
        'rasterio',
        'pymap3d',
        'utm',
        'geopy',
        'pydantic',    
    ]
)
