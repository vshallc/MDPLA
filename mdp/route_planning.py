from la.convolution import *
from la.la import *
from la.piecewise import *
from mdp import *
import sympy
import sympy.abc
from sympy.polys import Poly


def main():
    x = sympy.sympify('x')
    # p1 = P([0, 1])
    # p2 = P([0, -1])
    # pp = PiecewisePolynomial([p1, p2], [-10, 0, 10])
    # print(pp(np.array([-1, 5])))
    state = [State('Home'),
             State('x2'),
             State('Work')]
    miu = [(state[0], REL, PiecewisePolynomial([Poly('0', x),
                                                Poly('1', x),
                                                Poly('0', x)],
                                               [0, 1 / 6, 1 / 6, 7])),  # miu1
           (state[2], ABS, PiecewisePolynomial([Poly('0', x),
                                                Poly('16/9 * x - 16', x),
                                                Poly('-16/9 * x + 56/3', x),
                                                Poly('0', x)],
                                               [7, 9, 9.75, 10.5, 14])),  # miu2
           (state[1], REL, PiecewisePolynomial([Poly('0', x),
                                                Poly('x - 1', x),
                                                Poly('-x + 3', x),
                                                Poly('0', x)],
                                               [0, 1, 2, 3, 7])),  # miu3
           (state[1], REL, PiecewisePolynomial([Poly('0', x),
                                                Poly('x - 1/2', x),
                                                Poly('-x + 5/2', x),
                                                Poly('0', x)],
                                               [0, 0.5, 1.5, 2.5, 7])),  # miu4
           (state[2], REL, PiecewisePolynomial([Poly('1', x)],
                                               [1, 1]))]  # miu5
    likelihood = [PiecewisePolynomial([Poly('0', x),
                                       Poly('1', x)],  # L1
                                      [7, 7 + 50 / 60, 14]),
                  PiecewisePolynomial([Poly('1', x),
                                       Poly('0', x)],  # L2
                                      [7, 7 + 50 / 60, 14]),
                  PiecewisePolynomial([Poly('0', x),
                                       Poly('3/2 * x - 11', x),
                                       Poly('1', x),
                                       Poly('-3/2 * x + 31/2', x),
                                       Poly('0', x)],  # L3
                                      [7, 7 + 20 / 60, 8, 9 + 40 / 60, 10 + 20 / 60, 14]),
                  PiecewisePolynomial([Poly('1', x),
                                       Poly('-3/2 * x + 12', x),
                                       Poly('0', x),
                                       Poly('3/2 * x - 14.5', x),
                                       Poly('1', x)],  # L4
                                      [7, 7 + 20 / 60, 8, 9 + 40 / 60, 10 + 20 / 60, 14]),
                  PiecewisePolynomial([Poly('1', x)],  # L5
                                      [7, 14])]
    reward = {miu[0]: PiecewisePolynomial([Poly('0', x)], [7, 14]),
              miu[1]: PiecewisePolynomial([Poly('0', x)], [7, 14]),
              miu[2]: PiecewisePolynomial([Poly('0', x)], [7, 14]),
              miu[3]: PiecewisePolynomial([Poly('0', x)], [7, 14]),
              miu[4]: PiecewisePolynomial([Poly('0', x)], [7, 14])}
    # reward_start = [PiecewisePolynomial([Poly('0', x)], [7, 14]),
    # PiecewisePolynomial([Poly('0', x)], [7, 14]),
    #                 PiecewisePolynomial([Poly('0', x)], [7, 14]),
    #                 PiecewisePolynomial([Poly('0', x)], [7, 14]),
    #                 PiecewisePolynomial([Poly('0', x)], [7, 14])]
    # reward_arrival = [PiecewisePolynomial([Poly('0', x)], [7, 14]),
    #                   PiecewisePolynomial([Poly('1', x), Poly('-x + 12', x), Poly('0', x)], [7, 11, 12, 14]),
    #                   PiecewisePolynomial([Poly('0', x)], [7, 14]),
    #                   PiecewisePolynomial([Poly('0', x)], [7, 14]),
    #                   PiecewisePolynomial([Poly('0', x)], [7, 14])]
    # reward_duration = [PiecewisePolynomial([Poly('0', x)], [7, 14]),
    #                    PiecewisePolynomial([Poly('0', x)], [7, 14]),
    #                    PiecewisePolynomial([Poly('0', x)], [7, 14]),
    #                    PiecewisePolynomial([Poly('0', x)], [7, 14]),
    #                    PiecewisePolynomial([Poly('0', x)], [7, 14])]
    # add actions to states
    state[0].add_action('taking train', miu[0], likelihood[0])  # Miss the 8am train
    state[0].add_action('taking train', miu[1], likelihood[1])  # Caught the 8am train
    state[0].add_action('driving', miu[2], likelihood[2])  # Highway - rush hour
    state[0].add_action('driving', miu[3], likelihood[3])  # Highway - off peak
    state[1].add_action('driving', miu[4], likelihood[4])  # Drive on backroad

    # assign value functions
    # state[0].value_function = PiecewisePolynomial([Poly('0', x)], [7, 14])
    # state[1].value_function = PiecewisePolynomial([Poly('0', x)], [7, 14])
    # state[2].value_function = PiecewisePolynomial([Poly('0', x)], [7, 14])

    # for testing2
    # a = PiecewisePolynomial([Poly('0', x), Poly('1', x), Poly('1', x), Poly('0', x)],
    #                         [float('-inf'), -1, 0, 1, float('inf')])
    # b = PiecewisePolynomial([Poly('0', x), Poly('x + 1', x), Poly('-x + 1', x), Poly('0', x)],
    #                         [float('-inf'), -1, 0, 1, 100])
    # print('a')
    # for i in range(0, a.pieces):
    #     print(str(a.polynomial_pieces[i]) + ' ' + str(a.bounds[i:i + 2]))
    # a = 1.0 + a
    # print('a simplified')
    # for i in range(0, a.pieces):
    #     print(str(a.polynomial_pieces[i]) + ' ' + str(a.bounds[i:i + 2]))
    # aa = a(0)
    # print(aa)
    # test MDP
    mdp = MDP(state, miu, reward, state[0], {state[2]})
    u = value_iteration(mdp)
    for s in u:
        print(s, u[s])
    # c1 = mdp.V(None, x, a)
    # print('c1')
    # for i in range(0, c1.pieces):
    #     print(str(c1.polynomial_pieces[i]) + ' ' + str(c1.bounds[i:i + 2]))


if __name__ == "__main__":
    main()