#!/usr/bin/env python

import os

from setuptools import find_packages, setup, Command

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


# install requirements
install_requires = [
    'matplotlib',
    'numpy>=1.16.1',
    'ortools>=9.4',
    'pandas',
    'ropwr>=1.0.0',
    'scikit-learn>=1.0.2',
    'scipy>=1.6.0',
]

# extra requirements
extras_require = {
    'distributed': ['pympler', 'tdigest'],
    'test': [
        'coverage', 
        'flake8',
        'pytest',
        'pyarrow',
        'pympler',
        'tdigest',
    ],
    # For ecos support: https://github.com/embotech/ecos 
    'ecos': ['ecos']
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
    cmdclass={'clean': CleanCommand},
    python_requires='>=3.7',
    install_requires=install_requires,
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
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Programming Language :: Python :: 3.12',
        ]
    )
