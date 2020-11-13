"""To compile C extensions:

$ python setup.py build
"""

from distutils.core import Extension, setup
e = Extension('_distribution', ['distributionmodule.c'])
setup(name='_distribution', ext_modules=[e])

e = Extension('_sample', ['_samplemodule.c'])
setup(name='_sample', ext_modules=[e])
