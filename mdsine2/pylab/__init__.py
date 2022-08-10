from . import base
from . import util
from . import multiprocessing
from . import variables
from . import inference
from . import metropolis
from . import math
from . import dynamics
from . import diversity


# Get is* methods
from .util import isnumeric, isbool, isfloat, isint, isarray, issquare, isstr, \
    itercheck, istype, istuple, isdict, istree
from .variables import isVariable, isRandomVariable
from .inference import isMCMC, ismodel
from .metropolis import isMetropKernel
from .dynamics import isdynamics, isprocessvariance, isintegratable
from .multiprocessing import ispersistentworker, ispersistentpool, isDASW, isSADW

# Import commonly used Pylab functions and classes
from .util import toarray, fast_index, coarsen_phylogenetic_tree
from .base import *
from .variables import Variable, Constant, summary
from .inference import BaseMCMC
from .dynamics import integrate
