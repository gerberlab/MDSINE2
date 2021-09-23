import os
from pathlib import Path
from setuptools import setup
from distutils.core import Extension

lib_dir = Path(__file__).resolve().parent

with open(lib_dir / 'VERSION.txt', "r") as f:
    VERSION = f.readline().strip()

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
with open(lib_dir / 'requirements.txt', "r") as f:
    REQUIREMENTS = f.read().splitlines()

# This is for it to run on windows
if os.name == 'nt':
    from distutils.command import build_ext
    def get_export_symbols(self, ext):
        parts = ext.name.split(".")
        print('parts', parts)
        if parts[-1] == "__init__":
            initfunc_name = "PyInit_" + parts[-2]
        else:
            initfunc_name = "PyInit_" + parts[-1]

    build_ext.build_ext.get_export_symbols = get_export_symbols

# Custom C distributions
EXTENSIONS = [
    Extension('_distribution', ['mdsine2/pylab/c_code/distributionmodule.c']),
    Extension('_sample', ['mdsine2/pylab/c_code/_samplemodule.c'])
]

# Subpackages
PACKAGES = [
    'mdsine2',
    'mdsine2.cli',
    'mdsine2.cli.helpers',
    'mdsine2.pylab'
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
    include_package_data=True,
    entry_points={
        'console_scripts': [
            'mdsine2=mdsine2.cli:main'
        ]
    }
)
