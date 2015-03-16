import math
from numbers import Number
import numpy as np
import sympy
# import sympy.abc
from sympy.polys import Poly
import collections

x = sympy.symbols('x')


def max_onepiece(x, f: Poly, g: Poly, l, u):
    roots = sorted(set((f - g).real_roots()))
    new_polynomial_pieces = []
    new_bounds = [l]
    for r in roots:
        if l < r < u:
            m = (r + new_bounds[-1]) / 2
            if f.subs(x, m) >= g.subs(x, m):
                new_polynomial_pieces.append(f)
            else:
                new_polynomial_pieces.append(g)
            new_bounds.append(r)
    new_bounds.append(u)
    return PiecewisePolynomial(new_polynomial_pieces, new_bounds)


def constant_function_approximation(linear_piece, left_bound, right_bound):
    return linear_piece.subs(x, (left_bound + right_bound) / 2)


def pwc_function_approximation(linear_piece, left_bound, right_bound, error_tolerance):
    # print(linear_piece)
    if linear_piece.degree() <= 0:
        return [linear_piece], [left_bound, right_bound]
    elif linear_piece.degree() >= 2:
        raise ValueError('The function must be constant or linear.')
    a = linear_piece.LC()
    # print('a=', a, 'lb=', left_bound, 'rb=', right_bound)
    piece_num = math.ceil((math.fabs(a) * (right_bound - left_bound)) / (2 * error_tolerance))
    delta_x = (right_bound - left_bound) / piece_num
    pwc_result = []
    bounds_result = [left_bound]
    for i in range(0, piece_num):
        pwc_result.append(linear_piece.subs(x, left_bound + (i - 0.5) * delta_x))
        bounds_result.append(left_bound + (i + 1) * delta_x)
    return pwc_result, bounds_result


# def merge_bounds(bounds1, bounds2):
# seen = set()
# seen_add = seen.add
#     return sorted([b for b in bounds1 + bounds2 if not (b in seen or seen_add(b))])


class PiecewisePolynomial(object):
    def __init__(self, polynomial_pieces, bounds):
        # assert len(polynomial_pieces) < 1 or len(polynomial_pieces) + 1 != len(bounds)
        if len(polynomial_pieces) < 1 or len(polynomial_pieces) + 1 != len(bounds):
            raise ValueError('The length of polynomial list and condition list must be the same')
        self.__polynomial_pieces = polynomial_pieces
        self.__bounds = bounds
        self.__pieces = len(polynomial_pieces)
        self.simplify()

    def __call__(self, x):
        # x format: numpy.array
        return self.evaluate(x)

    def __str__(self):
        return str([(str(self.polynomial_pieces[i]), self.bounds[i], self.bounds[i + 1]) for i in range(self.pieces)])

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
        v = np.array([x])
        condition_list = []
        for i in range(1, len(self.__bounds)):
            cc = (self.__bounds[i - 1] <= v < self.__bounds[i])
            condition_list.append(cc)
        return np.piecewise(v, condition_list, self.__polynomial_pieces)

    def __add__(self, other):
        if isinstance(other, self.__class__):
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
        elif isinstance(other, Number):
            new_bounds = self.__bounds.copy()
            new_polynomial_pieces = [p + other for p in self.__polynomial_pieces]
            return PiecewisePolynomial(new_polynomial_pieces, new_bounds)
        else:
            raise TypeError("unsupported operand type(s) for +: '{}' and '{}'".format(self.__class__, type(other)))

    __radd__ = __add__

    def __mul__(self, other):
        if isinstance(other, self.__class__):
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
                '''
                print('p1',p1)
                print('p2',p2)
                gens = sympy.polys.polyutils._unify_gens(p1.gens, p2.gens)
                print('type1',type(p1.gens[0]))
                print('type2',type(p2.gens[0]))
                print('gen12',p1.gens,p2.gens)
                print('gens',gens,len(gens))
                print('dom12',p1.rep.dom,p2.rep.dom)
                dom = p1.rep.dom.unify_with_symbols(p2.rep.dom,gens)
                print('dom',dom)
                # dom = p1.rep.dom.unify(p2.rep.dom, gens)
                p_tmp = p1*p2
                new_polynomial_pieces.append(p_tmp)
                '''
                new_polynomial_pieces.append(p1 * p2)
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
        elif isinstance(other, Number):
            new_bounds = self.__bounds.copy()
            new_polynomial_pieces = [p * other for p in self.__polynomial_pieces]
            return PiecewisePolynomial(new_polynomial_pieces, new_bounds)
        else:
            raise TypeError("unsupported operand type(s) for +: '{}' and '{}'".format(self.__class__, type(other)))

    __rmul__ = __mul__

    def __eq__(self, other):
        if not isinstance(other, self.__class__):
            return False
        return (self.__polynomial_pieces, self.__bounds, self.__pieces) == \
               (other.polynomial_pieces, other.bounds, other.pieces)

    def __hash__(self):
        return hash(tuple(self.__polynomial_pieces))^hash(tuple(self.bounds))^hash(self.pieces)

    def simplify(self):
        if self.pieces > 1:
            i = 0
            p = self.pieces - 1
            while i < p:
                if self.polynomial_pieces[i] == self.polynomial_pieces[i + 1]:
                    del self.polynomial_pieces[i]
                    del self.bounds[i + 1]
                    p -= 1
                else:
                    i += 1
            self.__pieces = p + 1

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


def max_piece(x, f, g, lower, upper):
    roots = (f - g).real_roots()
    try:
        roots = [float(r) for r in roots if lower < r < upper]
    except TypeError:
        roots = []
    if roots:
        p = []
        b = []
        roots = [lower] + roots + [upper]
        for i in range(1, len(roots)):
            mid = (roots[i - 1] + roots[i]) / 2
            if f(mid) > g(mid):
                p.append(f)
            else:
                p.append(g)
            b.append(roots[i])
        return p, b
    else:
        mid = (lower + upper) / 2
        # return [f], [upper] if f(mid) > g(mid) else [g], [upper]
        if f(mid) > g(mid):
            return [f], [upper]
        else:
            return [g], [upper]


def max_piecewise(x, pw_f: PiecewisePolynomial, pw_g: PiecewisePolynomial):
    # for linear pieces only
    new_bounds = []
    new_polynomial_pieces = []
    pieces1 = iter(pw_f.polynomial_pieces)
    pieces2 = iter(pw_g.polynomial_pieces)
    bounds1 = iter(pw_f.bounds)
    bounds2 = iter(pw_g.bounds)
    b1_next = next(bounds1)
    b2_next = next(bounds2)
    if b1_next < b2_next:
        new_bounds.append(b1_next)
        b1_next = next(bounds1)
        p1 = next(pieces1)
        p2 = Poly('0', x)
    elif b1_next > b2_next:
        new_bounds.append(b2_next)
        b2_next = next(bounds2)
        p1 = Poly('0', x)
        p2 = next(pieces2)
    else:
        new_bounds.append(b1_next)
        b1_next = next(bounds1)
        b2_next = next(bounds2)
        p1 = next(pieces1)
        p2 = next(pieces2)
    b1_flag = b2_flag = True
    while b1_flag or b2_flag:
        # new_polynomial_pieces.append(p1 + p2)
        if b1_next < b2_next:
            p, b = max_piece(x, p1, p2, new_bounds[-1], b1_next)
            new_polynomial_pieces.extend(p)
            new_bounds.extend(b)
            p1 = next(pieces1, Poly('0', x))
            # new_bounds.append(b1_next)
            try:
                b1_next = next(bounds1)
            except StopIteration:
                b1_next = float('inf')
                b1_flag = False
        elif b1_next > b2_next:
            p, b = max_piece(x, p1, p2, new_bounds[-1], b2_next)
            new_polynomial_pieces.extend(p)
            new_bounds.extend(b)
            p2 = next(pieces2, Poly('0', x))
            # new_bounds.append(b2_next)
            try:
                b2_next = next(bounds2)
            except StopIteration:
                b2_next = float('inf')
                b2_flag = False
        else:
            p, b = max_piece(x, p1, p2, new_bounds[-1], b1_next)
            new_polynomial_pieces.extend(p)
            new_bounds.extend(b)
            p1 = next(pieces1, Poly('0', x))
            p2 = next(pieces2, Poly('0', x))
            # new_bounds.append(b1_next)
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