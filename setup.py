#!/usr/bin/env python

import os
import sys

from setuptools import setup, Command
from setuptools.command.test import test as TestCommand


long_description = '''
The optimal binning is the optimal discretization of a variable into bins
given a discrete or continuous numeric target. OptBinning is a library
written in Python implementing a rigorous and flexible mathematical
programming formulation to solving the optimal binning problem for a binary,
continuous and multiclass target type, incorporating constraints not
previously addressed.

Read the documentation at: http://gnpalencia.org/optbinning/

OptBinning is distributed under the Apache Software License (Apache 2.0).
'''


class CleanCommand(Command):
    user_options = []

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):
        os.system('rm -vrf ./build ./dist ./*.pyc ./*.tgz ./*.egg-info')


# test suites
class PyTest(TestCommand):
    def finalize_options(self):
        TestCommand.finalize_options(self)
        self.test_args = []
        self.test_suite = []

    def run_tests(self):
        # import here, because outside the eggs aren't loaded
        import pytest
        errcode = pytest.main(self.test_args)
        sys.exit(errcode)


# install requirements
install_requires = [
    'matplotlib',
    'numpy',
    'ortools>=7.2',
    'pandas',
    'scipy',
    'sklearn>=0.20.0',
    'pytest',
    'coverage'
]


setup(
    name="optbinning",
    version="0.1.0",
    description="OptBinning: The Python Optimal Binning library",
    long_description=long_description,
    author="Guillermo Navas-Palencia",
    author_email="g.navas.palencia@gmail.com",
    packages=["optbinning"],
    platforms="any",
    include_package_data=True,
    license="Apache Licence 2.0",
    url="https://github.com/guillermo-navas-palencia/optbinning",
    tests_require=['pytest'],
    cmdclass={'clean': CleanCommand, 'test': PyTest},
    python_requires='>=3.6',
    classifiers=[
        'Topic :: Scientific/Engineering :: Mathematics',
        'Topic :: Software Development :: Libraries',
        'Topic :: Software Development :: Libraries :: Python Modules',
        'Intended Audience :: Developers',
        'Intended Audience :: Education',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: Apache Software License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7']
    )
