'''Exceptions for pylab
'''

class UndefinedError(Exception):
    '''Pointer in the object is not instantiated
    '''
    pass


class MathError(Exception):
    '''Mathematical functions are not defined with the
    current values
    '''
    pass


class GraphIDError(Exception):
    '''Adding a node as a parent to a node in a different graph
    '''
    pass


class InheritanceError(Exception):
    '''When a variable is the wrong subtype
    '''
    pass


class InitializationError(Exception):
    '''This error is thrown when the initialization of the inference/graph
    is not done properly.
    '''
    pass
