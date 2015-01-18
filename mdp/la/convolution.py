import sympy
import sympy.abc
from sympy.functions.special.delta_functions import *
from sympy.polys import Poly
from la.piecewise import *

'''
The convolute_onepiece function is from
http://www.mare.ee/indrek/misc/convolution.pdf
slightly changed for our use
'''

# Convolute two "one-piece" functions. Arguments F and G
# are tuples in form (h(x), a_h, b_h), where h(x) is
# the function and [a_h, b_h) is the range where the functions
# are non-zero.


# # Two "flat" functions, uniform centered PDF-s
# F = (sympy.sympify(0.5), -1, 1)
# G = (sympy.sympify(0.05), -10, 10)
# print(convolute_onepiece(sympy.abc.x, F, G))

def convolute_onepiece(x, F, G):
    f, a_f, b_f = F
    g, a_g, b_g = G
    # make sure ranges are in order, swap values if necessary
    if b_f - a_f > b_g - a_g:
        f, a_f, b_f, g, a_g, b_g = g, a_g, b_g, f, a_f, b_f
    if a_f == b_f:
        return PiecewisePolynomial([Poly(f.subs(x, a_f) * g.subs(x, x - a_f), x)], [a_f + a_g, a_f + b_g])
    y = sympy.Dummy('y')
    i = sympy.integrate(f.subs(x, y) * g.subs(x, x - y), y)
    return PiecewisePolynomial([Poly(i.subs(y, x - a_g) - i.subs(y, a_f), x),
                                Poly(i.subs(y, b_f) - i.subs(y, a_f), x),
                                Poly(i.subs(y, b_f) - i.subs(y, x - b_g), x)],
                               [a_f + a_g,
                                b_f + a_g,
                                a_f + b_g,
                                b_f + b_g])


def convolute_piecewise(x, pw_f: PiecewisePolynomial, pw_g: PiecewisePolynomial):
    h = PiecewisePolynomial([Poly('0', x)], [pw_f.bounds[0] + pw_g.bounds[0], pw_f.bounds[-1] + pw_g.bounds[-1]])
    for i in range(0, pw_f.pieces):
        for j in range(0, pw_g.pieces):
            f = pw_f.polynomial_pieces[i]
            g = pw_g.polynomial_pieces[j]
            if f.is_zero or g.is_zero:
                continue
            f_piece = (f, pw_f.bounds[i], pw_f.bounds[i + 1])
            g_piece = (g, pw_g.bounds[j], pw_g.bounds[j + 1])
            piecewise_result = convolute_onepiece(x, f_piece, g_piece)
            h += piecewise_result
    return h