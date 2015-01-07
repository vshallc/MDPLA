# import numpy as np
from numpy.polynomial import Polynomial as P
from mdp import *
from la.la import *
from la.piecewise import *
import numpy as np
import sympy
from sympy.polys import Poly


def main():
    #p1 = P([0, 1])
    #p2 = P([0, -1])
    #pp = PiecewisePolynomial([p1, p2], [-10, 0, 10])
    #print(pp(np.array([-1, 5])))

    state = [State('Home'),
             State('x2'),
             State('Work')]
    miu = [Outcome(state[0]),   # miu1
           Outcome(state[2]),   # miu2
           Outcome(state[1]),   # miu3
           Outcome(state[1]),   # miu4
           Outcome(state[2])]   # miu5
    likelihood = [PiecewisePolynomial([P([0]), P([1])],                                             # L1
                                      [7, 7 + 50 / 60, 14]),
                  PiecewisePolynomial([P([1]), P([0])],                                             # L2
                                      [7, 7 + 50 / 60, 14]),
                  PiecewisePolynomial([P([0]), P([-11, 1.5]), P([1]), P([15.5, -1.5]), P([0])],     # L3
                                      [7, 7 + 20 / 60, 8, 9 + 40 / 60, 10 + 20 / 60, 14]),
                  PiecewisePolynomial([P([1]), P([12, -1.5]), P([0]), P([-14.5, 1.5]), P([1])],     # L4
                                      [7, 7 + 20 / 60, 8, 9 + 40 / 60, 10 + 20 / 60, 14]),
                  PiecewisePolynomial([P([1])],                                                     # L5
                                      [7, 14])]
    # add actions to states
    state[0].add_action('taking train', miu[0], likelihood[0])      # Miss the 8am train
    state[0].add_action('taking train', miu[1], likelihood[1])      # Caught the 8am train
    state[0].add_action('driving', miu[2], likelihood[2])           # Highway - rush hour
    state[0].add_action('driving', miu[3], likelihood[3])           # Highway - off peak
    state[1].add_action('driving', miu[4], likelihood[4])           # Drive on backroad

    # assign value functions
    state[0].value_function = PiecewisePolynomial([P([float('-infinity')])], [7, 14])
    state[1].value_function = PiecewisePolynomial([P([float('-infinity')])], [7, 14])
    state[2].value_function = PiecewisePolynomial([P([1]), P([0])], [7, 11, 14])

    test_p = likelihood[2]
    for i in range(0, test_p.pieces):
        print(str(test_p.polynomial_pieces[i]) + ' ' + str(test_p.bounds[i:i+2]))
    print('constant')
    test_p_c = test_p.to_constant_function_approximation()
    for i in range(0, test_p_c.pieces):
        print(str(test_p_c.polynomial_pieces[i]) + ' ' + str(test_p_c.bounds[i:i+2]))
    print('pwc')
    test_p_pwc = test_p.to_pwc_function_approximation()
    for i in range(0, test_p_pwc.pieces):
        print(str(test_p_pwc.polynomial_pieces[i]) + ' ' + str(test_p_pwc.bounds[i:i+2]))


if __name__ == "__main__":
    main()