import sympy
import sympy.abc
from sympy.polys import Poly
from la.piecewise import *

'''
The convolute_onepiece function is from
http://www.mare.ee/indrek/misc/convolution.pdf
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
    y = sympy.Dummy('y')
    i = sympy.integrate(f.subs(x, y) * g.subs(x, x - y), y)
    return [
        (i.subs(y, x - a_g) - i.subs(y, a_f), a_f + a_g, b_f + b_g),
        (i.subs(y, b_f) - i.subs(y, a_f), b_f + a_g, a_f + b_g),
        (i.subs(y, b_f) - i.subs(y, x - b_g), a_f + b_g, b_f + a_g)
    ]

def merge_pieces_list(main_pieces, new_pieces):


def convolute_piecewise(pw_f: PiecewisePolynomial, pw_g: PiecewisePolynomial):
    x = sympy.sympify('x')
    H = sympy.sympify('0')
    c = []
    for i in range(0, pw_f.pieces):
        for j in range(0, pw_g.pieces):
            f = pw_f.polynomial_pieces[i]
            g = pw_g.polynomial_pieces[j]
            df = f.degree()
            if df == 0:
                f_i = sympy.sympify(f.coef[0])
            elif df == 1:
                f_i = sympy.sympify(f.coef[0] + f.coef[1] * x)
            else:
                raise ValueError('The piece must be constant or linear')
            dg = g.degree()
            if dg == 0:
                g_j = sympy.sympify(g.coef[0])
            elif dg == 1:
                g_j = sympy.sympify(g.coef[0] + g.coef[1] * x)
            else:
                raise ValueError('The piece must be constant or linear')
            if f_i == 0 or g_j == 0:
                continue
            f_piece = (f_i, pw_f.bounds[i], pw_f.bounds[i + 1])
            g_piece = (g_j, pw_g.bounds[i], pw_g.bounds[i + 1])
            convolute_result = convolute_onepiece(x, f_piece, g_piece)

