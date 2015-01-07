import math
import numpy as np
from numpy.polynomial import Polynomial as P

def constant_function_approximation(linear_piece, left_bound, right_bound):
    return P([linear_piece((left_bound + right_bound) / 2)])


def pwc_function_approximation(linear_piece, left_bound, right_bound, error_tolerance):
    assert isinstance(linear_piece, P)
    if linear_piece.degree() == 0:
        return [linear_piece], [left_bound, right_bound]
    a = linear_piece.coef[1]
    b = linear_piece.coef[0]
    piece_num = math.ceil((math.fabs(a) * (right_bound - left_bound)) / (2 * error_tolerance))
    delta_x = (right_bound - left_bound) / piece_num
    pwc_result = []
    bounds_result = [left_bound]
    for i in range(0, piece_num):
        pwc_result.append(P([a * (left_bound + (i - 0.5) * delta_x) + b]))
        bounds_result.append(left_bound + (i + 1) * delta_x)
    return pwc_result, bounds_result

def merge_bounds(bounds1, bounds2):
    seen = set()
    seen_add = seen.add
    return sorted([b for b in bounds1 + bounds2 if not (b in seen or seen_add(b))])

class PiecewisePolynomial(object):
    def __init__(self, polynomial_pieces, bounds):
        assert len(polynomial_pieces) < 1 or len(polynomial_pieces) + 1 != len(bounds)
        # if len(linear_pieces) < 1 or len(linear_pieces) + 1 != len(bounds):
        #     raise ValueError('The length of polynomial list and condition list must be the same')
        self.__polynomial_pieces = polynomial_pieces
        self.__bounds = bounds
        self.__pieces = len(polynomial_pieces)

    def __call__(self, x):
        # x format: numpy.array
        return self.evaluate(x)

    @property
    def polynomial_pieces(self):
        return self.__polynomial_pieces

    @property
    def bounds(self):
        return self.__bounds

    @property
    def pieces(self):
        return self.__pieces

    def evaluate(self, x):
        # x format: numpy.array
        condition_list = []
        for i in range(1, len(self.__bounds)):
            condition_list.append((self.__bounds[i - 1] <= x) & (x < self.__bounds[i]))
        return np.piecewise(x, condition_list, self.__polynomial_pieces)

    def add(self, piecewise_polynomial):
        new_bounds = merge_bounds(self.bounds, piecewise_polynomial.bounds)
        new_polynomial_pieces = []
        bounds1 = self.bounds
        bounds2 = piecewise_polynomial.bounds
        if new_bounds[0] < bounds1:
            bounds1 = new_bounds[0]
        i = 0
        j = 0
        for n in range(0, len(new_bounds) - 1):
            p = P([0])
            # if new_bounds[n]
            # new_polynomial_pieces.append()

    def to_constant_function_approximation(self):
        new_polynomial_pieces = []
        new_bounds = self.__bounds.copy()
        for i in range(0, self.__pieces):
            constant = constant_function_approximation(self.__polynomial_pieces[i],
                                                       self.__bounds[i], self.__bounds[i + 1])
            new_polynomial_pieces.append(constant)
        return PiecewisePolynomial(new_polynomial_pieces, new_bounds)

    def to_pwc_function_approximation(self, error_tolerance=1):
        new_polynomial_pieces = []
        new_bounds = [self.__bounds[0]]
        for i in range(0, self.__pieces):
            pwc = pwc_function_approximation(self.__polynomial_pieces[i],
                                             self.__bounds[i], self.__bounds[i + 1],
                                             error_tolerance)
            pwc_bounds = pwc[1][1:]
            new_polynomial_pieces.extend(pwc[0])
            new_bounds.extend(pwc_bounds)
        return PiecewisePolynomial(new_polynomial_pieces, new_bounds)
