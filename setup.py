'''MDSINE2

Built and tested for Python >= 3.7

For installing on Windows OS
----------------------------
For python 3.7.3, `distutils` has a bug that is described issue 35893 in Python Bug
Tracker (https://bugs.python.org/issue35893). This bug does not affect MacOS.
'''
from pathlib import Path
from setuptools import setup
from distutils.core import Extension

lib_dir = Path(__file__).resolve().parent

VERSION = '0.1.0'
SHORT_DESC = 'Implements core features of the MDSINE2 model'
LONG_DESC = \
    '''
    This package provides both high and low level classes for 
    sampling, inference, visualization, clustering, and 
    saving to disk. There is a strong emphasis on scalability,
    robustness, inheritance, speed, and code transparency.

    Supports Python >= 3.7. Tested on Python 3.7.3.

    Somethings this package does well:
        - Fast sampling of distributions
        - Fast writing and reading to disk
        - Full pipeline for Bayesian inference
        - Visualization of posterior and posterior samples
        - Model-building
        - Integrated clustering and perturbation classes
        - Scaling the inference

    Somethings this package does not do well:
        - Optimization based algorithms (use PyTorch or TensorFlow)
    '''

# Package requirements: Parse from `requirements.txt`.
requirementPath = lib_dir / 'requirements.txt'
REQUIREMENTS = []
if Path(requirementPath).is_file():
    with open(requirementPath, "r") as f:
        REQUIREMENTS = f.read().splitlines()

# Custom C distributions
EXTENSIONS = [
    Extension('_distribution', ['mdsine2/pylab/c_code/distributionmodule.c']),
    Extension('_sample', ['mdsine2/pylab/c_code/_samplemodule.c'])
]

# Subpackages
PACKAGES = [
    'mdsine2',
    'mdsine2.pylab',
    'mdsine2.initializers'
]

setup(
    name='mdsine2',
    version=VERSION,
    description=SHORT_DESC,
    long_description=LONG_DESC,
    author='David Kaplan, Travis Gibson, Younhun Kim, Sawal Acharya',
    author_email='tegibson@bwh.harvard.edu',
    python_requires='>=3.7.3',
    license='GNU',
    packages=PACKAGES,
    zip_safe=False,
    install_requires=REQUIREMENTS,
    ext_modules=EXTENSIONS,
    include_package_data=True
)
