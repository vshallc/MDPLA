import matplotlib.pyplot as plt
from la.la import *
from la.piecewise import *
from exact_mdp.mdp import *
import sympy
import sympy.abc
from sympy.polys import Poly


def init_mdp():
    state = [State('Home'),
             State('x2'),
             State('Work')]
    miu = {'miu1': (state[0], REL, PiecewisePolynomial([Poly('0', x),
                                                        Poly('1', x),
                                                        Poly('0', x)],
                                                       [0, 1 / 6, 1 / 6, 7])),  # miu1
           'miu2': (state[2], ABS, PiecewisePolynomial([Poly('0', x),
                                                        Poly('16/9 * x - 16', x),
                                                        Poly('-16/9 * x + 56/3', x),
                                                        Poly('0', x)],
                                                       [7, 9, 9.75, 10.5, 14])),  # miu2
           'miu3': (state[1], REL, PiecewisePolynomial([Poly('0', x),
                                                        Poly('x - 1', x),
                                                        Poly('-x + 3', x),
                                                        Poly('0', x)],
                                                       [0, 1, 2, 3, 7])),  # miu3
           'miu4': (state[1], REL, PiecewisePolynomial([Poly('0', x),
                                                        Poly('x - 1/2', x),
                                                        Poly('-x + 5/2', x),
                                                        Poly('0', x)],
                                                       [0, 0.5, 1.5, 2.5, 7])),  # miu4
           'miu5': (state[2], REL, PiecewisePolynomial([Poly('1', x)],
                                                       [1, 1]))}  # miu5
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
    reward = {'miu1': PiecewisePolynomial([Poly('0', x)], [7, 14]),
              'miu2': PiecewisePolynomial([Poly('0', x)], [7, 14]),
              'miu3': PiecewisePolynomial([Poly('0', x)], [7, 14]),
              'miu4': PiecewisePolynomial([Poly('0', x)], [7, 14]),
              'miu5': PiecewisePolynomial([Poly('0', x)], [7, 14])}
    # reward_start = [PiecewisePolynomial([Poly('0', x)], [7, 14]),
    # PiecewisePolynomial([Poly('0', x)], [7, 14]),
    # PiecewisePolynomial([Poly('0', x)], [7, 14]),
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
    state[0].add_action('taking train', 'miu1', likelihood[0])  # Miss the 8am train
    state[0].add_action('taking train', 'miu2', likelihood[1])  # Caught the 8am train
    state[0].add_action('driving', 'miu3', likelihood[2])  # Highway - rush hour
    state[0].add_action('driving', 'miu4', likelihood[3])  # Highway - off peak
    state[1].add_action('driving', 'miu5', likelihood[4])  # Drive on backroad
    # assign value functions
    # state[0].value_function = PiecewisePolynomial([Poly('0', x)], [7, 14])
    # state[1].value_function = PiecewisePolynomial([Poly('0', x)], [7, 14])
    # state[2].value_function = PiecewisePolynomial([Poly('0', x)], [7, 14])
    mdp = MDP(state, miu, reward, state[0],
              {state[2]: PiecewisePolynomial([Poly('1', x), Poly('-x + 12', x), Poly('0', x)], [7, 11, 12, 14])},
              [7, 14], lazy=1, pwc=1, lazy_error_tolerance=0.03)
    return mdp


def main():
    mdp = init_mdp()
    # test MDP
    u = mdp.value_iteration()
    for s in u:
        print(s, u[s])
    bd = u[mdp.states[0]].bounds
    print(bd)
    t = []
    c = 0
    while c < len(bd) - 1:
        stp = (bd[c + 1] - bd[c]) / 10.0
        for itv in np.arange(bd[c], bd[c + 1], stp):
            t.append(itv)
        c += 1
    t.append(bd[-1])
    v = [u[mdp.states[0]](tt) for tt in t]
    print(t, v)
    plt.plot(t, v)
    plt.show()


if __name__ == "__main__":
    main()