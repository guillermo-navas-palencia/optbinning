#!/usr/bin/env python

import os
import sys

from setuptools import find_packages, setup, Command
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
    'numpy>=1.16.1',
    'ortools>=7.2',
    'pandas',
    'ropwr>=0.2.0',
    'scikit-learn>=0.22.0',
    'scipy>=1.6.0',
]

# test requirements
tests_require = [
    'pytest',
    'coverage'
]

# extra requirements
extras_require = {
    'distributed': ['pympler', 'tdigest'],
}


# Read version file
version_info = {}
with open("optbinning/_version.py") as f:
    exec(f.read(), version_info)


setup(
    name="optbinning",
    version=version_info['__version__'],
    description="OptBinning: The Python Optimal Binning library",
    long_description=long_description,
    author="Guillermo Navas-Palencia",
    author_email="g.navas.palencia@gmail.com",
    packages=find_packages(exclude=['tests', 'tests.*']),
    platforms="any",
    include_package_data=True,
    license="Apache Licence 2.0",
    url="https://github.com/guillermo-navas-palencia/optbinning",
    cmdclass={'clean': CleanCommand, 'test': PyTest},
    python_requires='>=3.7',
    install_requires=install_requires,
    tests_require=tests_require,
    extras_require=extras_require,
    classifiers=[
        'Topic :: Scientific/Engineering :: Mathematics',
        'Topic :: Software Development :: Libraries',
        'Topic :: Software Development :: Libraries :: Python Modules',
        'Intended Audience :: Developers',
        'Intended Audience :: Education',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: Apache Software License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9']
    )
