#!/usr/bin/env python

from setuptools import setup

# Read the long description from the readmefile
with open("README.rst", "rb") as f:
    long_descr = f.read().decode("utf-8")

# Run setup
setup(name='molecular',
      packages=['molecular', ],
      version='0.5',
      description='Calculation of molecular atmosphere scattering parameters',
      long_description = long_descr,
      author='Ioannis Binietoglou, Mike Kottas',
      author_email='binietoglou@noa.gr',
      install_requires=[
        "numpy",
        "matplotlib",
        "scipy", 
        "sphinx",
        "numpydoc",
        "pytest",
        "sphinx-bootstrap-theme",
    ],
     )

