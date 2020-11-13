'''MDSINE2

Built and tested for Python >= 3.7

For installing on Windows OS
----------------------------
For python 3.7.3, `distutils` has a bug that is described issue 35893 in Python Bug
Tracker (https://bugs.python.org/issue35893). This bug does not affect MacOS.
'''
import os

from setuptools import setup
from distutils.core import Extension, setup

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

VERSION = '4.0.0'
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
REQUIREMENTS = [
    'numpy>=1.16.4',
    'pandas>=0.25',
    'matplotlib',
    'sklearn==0.0',
    'xlrd',
    'seaborn',
    'h5py==2.9.0',
    'psutil',
    'ete3',
    'networkx==2.3',
    'numba==0.48.0']

# Custom C distributions
ext1 = Extension('_distribution', ['mdsine2/pylab/c_code/distributionmodule.c'])
ext2 = Extension('_sample', ['mdsine2/pylab/c_code/_samplemodule.c'])
EXTENSIONS = [ext1, ext2]

# Subpackages
PACKAGES = [
    'mdsine2', 
    # 'mdsine2.posterior', 
    'mdsine2.pylab',
    'mdsine2.raw_data']

# Install gibson dataset
PACKAGE_DIR = {
    'gibson': 'mdsine2/raw_data/gibson_dataset'}
PACKAGE_DATA = {
    'gibson': ['mdsine2/raw_data/gibson_dataset/*']
}

setup(
    name='mdsine2',
    version=VERSION,
    description=SHORT_DESC,
    long_description=LONG_DESC,
    author='David Kaplan',
    author_email='dkaplan65@gmail.com',
    python_requires='>=3.7',
    license='MIT',
    packages=PACKAGES,
    zip_safe=False,
    install_requires=REQUIREMENTS,
    ext_modules=EXTENSIONS,
    package_dir=PACKAGE_DIR,
    package_data=PACKAGE_DATA,
    include_package_data=True)
