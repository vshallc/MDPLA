import math
import numpy as np
from numpy.polynomial import Polynomial as P
import sympy
import sympy.abc
from sympy.polys import Poly

x = sympy.sympify('x')

def constant_function_approximation(linear_piece, left_bound, right_bound):
    return linear_piece.subs(x, (left_bound + right_bound) / 2)


def pwc_function_approximation(linear_piece, left_bound, right_bound, error_tolerance):
    if linear_piece.degree() == 0:
        return [linear_piece], [left_bound, right_bound]
    elif linear_piece.degree() >= 2:
        raise ValueError('The function must be constant or linear.')
    a = linear_piece.LC()
    piece_num = math.ceil((math.fabs(a) * (right_bound - left_bound)) / (2 * error_tolerance))
    delta_x = (right_bound - left_bound) / piece_num
    pwc_result = []
    bounds_result = [left_bound]
    for i in range(0, piece_num):
        pwc_result.append(linear_piece.subs(x, left_bound + (i - 0.5) * delta_x))
        bounds_result.append(left_bound + (i + 1) * delta_x)
    return pwc_result, bounds_result


# def merge_bounds(bounds1, bounds2):
#     seen = set()
#     seen_add = seen.add
#     return sorted([b for b in bounds1 + bounds2 if not (b in seen or seen_add(b))])


class PiecewisePolynomial(object):
    def __init__(self, polynomial_pieces, bounds):
        # assert len(polynomial_pieces) < 1 or len(polynomial_pieces) + 1 != len(bounds)
        if len(polynomial_pieces) < 1 or len(polynomial_pieces) + 1 != len(bounds):
            raise ValueError('The length of polynomial list and condition list must be the same')
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

    def __add__(self, other):
        new_bounds = []
        new_polynomial_pieces = []
        pieces1 = iter(self.polynomial_pieces)
        pieces2 = iter(other.polynomial_pieces)
        bounds1 = iter(self.bounds)
        bounds2 = iter(other.bounds)
        b1_next = next(bounds1)
        b2_next = next(bounds2)
        if b1_next < b2_next:
            new_bounds.append(b1_next)
            b1_next = next(bounds1)
            p1 = next(pieces1)
            p2 = Poly('0', x)
        elif b1_next == b2_next:
            new_bounds.append(b1_next)
            b1_next = next(bounds1)
            b2_next = next(bounds2)
            p1 = next(pieces1)
            p2 = next(pieces2)
        else:
            new_bounds.append(b2_next)
            b2_next = next(bounds2)
            p1 = Poly('0', x)
            p2 = next(pieces2)
        b1_flag = b2_flag = True
        while b1_flag or b2_flag:
            new_polynomial_pieces.append(p1 + p2)
            if b1_next < b2_next:
                p1 = next(pieces1, Poly('0', x))
                new_bounds.append(b1_next)
                try:
                    b1_next = next(bounds1)
                except StopIteration:
                    b1_next = float('inf')
                    b1_flag = False
            elif b1_next > b2_next:
                p2 = next(pieces2, Poly('0', x))
                new_bounds.append(b2_next)
                try:
                    b2_next = next(bounds2)
                except StopIteration:
                    b2_next = float('inf')
                    b2_flag = False
            else:
                p1 = next(pieces1, Poly('0', x))
                p2 = next(pieces2, Poly('0', x))
                new_bounds.append(b1_next)
                try:
                    b1_next = next(bounds1)
                except StopIteration:
                    b1_next = float('inf')
                    b1_flag = False
                try:
                    b2_next = next(bounds2)
                except StopIteration:
                    b2_next = float('inf')
                    b2_flag = False
        return PiecewisePolynomial(new_polynomial_pieces, new_bounds)

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
