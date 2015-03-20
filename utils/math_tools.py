from exact_mdp.la.piecewise import *
from math import sqrt, log

LN2 = log(2)  # ln(2)


def norm_pdf_linear_approximation(mu, sigma, timespan):
    assert sigma > 0, 'sigma must be positive'
    delta = sqrt(2*sigma*sigma*LN2)
    h = 0.5/delta
    k = h/2/delta
    bound_left = -delta*2 + mu
    bound_right = delta*2 + mu
    new_polynomial_pieces = []
    new_bound = [timespan[0]]
    if bound_left >= timespan[1] or bound_right <= timespan[0]:
        return PiecewisePolynomial([Poly('0', x)], timespan[:])
    if bound_left > timespan[0]:
        new_polynomial_pieces.append(Poly('0', x))
        new_bound.append(bound_left)
    if timespan[0] < mu < timespan[1]:
        new_polynomial_pieces.append(Poly(k*x+h, x))
        new_polynomial_pieces.append(Poly(-k*x+h, x))
        new_bound.append(mu)
    elif mu <= timespan[0]:
        new_polynomial_pieces.append(Poly(-k*x+h, x))
    else:  # mu >= timespan[1]
        new_polynomial_pieces.append(Poly(k*x+h, x))
    if bound_right < timespan[1]:
        new_polynomial_pieces.append(Poly('0', x))
        new_bound.append(bound_right)
    new_bound.append(timespan[1])
    return PiecewisePolynomial(new_polynomial_pieces, new_bound)