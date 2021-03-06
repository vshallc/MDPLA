import matplotlib.pyplot as plt
from exact_mdp.mdp import *
import time

def init_mdp():
    state = [State('Home'),
             State('x2'),
             State('Work')]
    miu = {'miu1': (state[0], REL, PiecewisePolynomial([P([1])], [1 / 6, 1 / 6])),  # miu1
           'miu2': (state[2], ABS, PiecewisePolynomial([P([-16, 16/9]), P([56/3, -16/9])], [9, 9.75, 10.5])),  # miu2
           'miu3': (state[1], REL, PiecewisePolynomial([P([-1, 1]), P([3, -1])], [1, 2, 3])),  # miu3
           'miu4': (state[1], REL, PiecewisePolynomial([P([-1/2, 1]), P([5/2, -1])], [0.5, 1.5, 2.5])),  # miu4
           'miu5': (state[2], REL, PiecewisePolynomial([P([1])], [1, 1]))}  # miu5
    likelihood = [PiecewisePolynomial([P([1])], [7 + 50 / 60, 14]),  # L1
                  PiecewisePolynomial([P([1])], [7, 7 + 50 / 60]),  # L2
                  PiecewisePolynomial([P([-11, 3/2]), P([1]), P([31/2, -3/2])],
                                      [7 + 20 / 60, 8, 9 + 40 / 60, 10 + 20 / 60]),  # L3
                  PiecewisePolynomial([P([1]), P([12, -3/2]), P([0]), P([-14.5, 3/2]), P([1])],
                                      [7, 7 + 20 / 60, 8, 9 + 40 / 60, 10 + 20 / 60, 14]),  # L4
                  PiecewisePolynomial([P([1])], [7, 14])]  # L5
    reward = {'miu1': PiecewisePolynomial([P([0])], [7, 14]),
              'miu2': PiecewisePolynomial([P([0])], [7, 14]),
              'miu3': PiecewisePolynomial([P([0])], [7, 14]),
              'miu4': PiecewisePolynomial([P([0])], [7, 14]),
              'miu5': PiecewisePolynomial([P([0])], [7, 14])}
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
              {state[2]: PiecewisePolynomial([P([1]), P([12, -1]), P([0])], [7, 11, 12, 14])},
              [7, 14], lazy=0, pwc=0, lazy_error_tolerance=0.03)
    return mdp


def main():
    start_time=time.time()
    for i in range(100):
        mdp = init_mdp()
        # test MDP
        u = mdp.value_iteration()
    print("--- %s seconds ---" % (time.time() - start_time))
    # for s in u:
    #     print(s, u[s])
    # bd = u[mdp.states[0]].bounds
    # print(bd)
    # t = []
    # c = 0
    # while c < len(bd) - 1:
    #     stp = (bd[c + 1] - bd[c]) / 2.0
    #     for itv in np.arange(bd[c], bd[c + 1], stp):
    #         t.append(itv)
    #     c += 1
    # t.append(bd[-1])
    # v = [u[mdp.states[0]](tt) for tt in t]
    # print(t, v)
    # plt.plot(t, v)
    # plt.show()


if __name__ == "__main__":
    main()