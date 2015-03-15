"""la.py

The Lazy Approximation algorithm

"""

import math
import numpy as np
import sympy
import sympy.abc
import sympy.functions
import sympy.functions.elementary.piecewise
import sympy.polys

# from la.exceptions import *

# def constant_function_approximation(linear_piece, left_bound, right_bound):
#     assert isinstance(linear_piece, sympy.polys.Poly)




class LazyApproximation(object):
    def __init__(self, value_function):
        self.__value_function = value_function