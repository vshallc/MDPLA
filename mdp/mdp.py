"""mdp.py

Define general MDP components

"""
import sympy
from la.convolution import *
# Constants
ABS = 0
REL = 1
# variable
x = sympy.sympify('x')
y = sympy.sympify('y')


def U_ABS_onepiece(t, P, V):
    p, a_p, b_p = P
    v, a_v, b_v = V
    a_max = max(a_p, a_v)
    b_min = min(b_p, b_v)
    if a_max > b_min:
        return PiecewisePolynomial([Poly('0', t)], [a_v, b_v])
    if a_p == b_p:
        return PiecewisePolynomial([Poly(v.subs(t, a_p) * p.subs(t, a_p), t)], [a_p, b_p])
    i = sympy.integrate(p * v, t)
    return PiecewisePolynomial([Poly(i.subs(t, b_min) - i.subs(t, a_max), t)], [a_max, b_min])


def U_ABS(t, P, V):
    h = PiecewisePolynomial([Poly('0', t)], [min(P.bounds[0], V.bounds[0]), max(P.bounds[-1], V.bounds[-1])])
    for i in range(0, P.pieces):
        for j in range(0, V.pieces):
            p = P.polynomial_pieces[i]
            v = V.polynomial_pieces[j]
            if p.is_zero or v.is_zero:
                continue
            p_piece = (p, P.bounds[i], P.bounds[i + 1])
            v_piece = (v, V.bounds[j], V.bounds[j + 1])
            piecewise_result = U_ABS_onepiece(t, p_piece, v_piece)
            h += piecewise_result
    return h


def U_REL(t, P, V):
    return convolute_piecewise(t, P, V)


class State(object):
    # _value_function = None
    __action = dict()

    def __init__(self, label):
        self.__label = label

    def __str__(self):
        return "(" + self.label + ")"

    def add_action(self, action, miu, likelihood):
        if action not in self.__action:
            self.__action[action] = dict()
        self.__action[action][miu] = likelihood

    @property
    def label(self):
        return self.__label

    @property
    def get_outcomes(self, action):
        return self.__action[action]

    @property
    def action_set(self):
        return self.__action.keys()

    @property
    def value_function(self):
        return self._value_function

    @value_function.setter
    def value_function(self, value_function):
        self._value_function = value_function


class Action(object):
    def __init__(self, label):
        self.__label = label


class MDP(object):
    def __init__(self, S, miu, R,
                 initial_state, terminal_state_set):
        self.__S = S  # State set
        # self.__A = A      # Action set
        self.__miu = miu
        # self.__R_s = R_s  # Reward of start time
        # self.__R_a = R_a  # Reward of arrival time
        # self.__R_d = R_d  # Reward of duration
        self.__R = R
        # self.__L = L
        self.__initial_state = initial_state
        self.__terminal_state_set = terminal_state_set
        # self.__current_state = initial_state

    # def reset(self):
    # self.__current_state = self.__initial_state

    def T(self, state, action):
        if state in self.__terminal_state_set:
            return {None: 1.0}
        else:
            return state.perform(action)

    # def R(self, i_miu, d):
    # return self.__R_s[i_miu] + self.__R_a[i_miu].subs(x, x + d) + self.__R_d[i_miu].subs(x, d)
    def R(self, miu):
        return self.__R[miu]

    def V(self, miu):
        return miu[0].value_function()

    def P(self, miu):
        return miu[2]

    def U(self, miu):
        if miu[1] == ABS:
            return self.R(miu) + U_ABS(x, self.P(miu), self.V(miu))
        elif miu[1] == REL:
            return self.R(miu) + U_REL(x, self.P(miu), self.V(miu))
        else:
            raise ValueError('The type of the miu time distribution function is wrong')

    def Q(self, s, a):
        q = PiecewisePolynomial([Poly('0', x)], s.value_function().bounds)
        outcomes = s.get_outcomes(a)
        for miu in outcomes:
            q += outcomes[miu] * self.U(miu)
        return q

    def V_bar(self, s: State):
        act_set = list(s.action_set)
        if len(act_set) == 1:
            return self.Q(s, act_set[0])
        else:
            best_pw = act_set[0]
            for a in act_set[1:]:
                best_pw = max_piecewise(best_pw, self.Q(s, a))
            return best_pw

    def V(self, s, t, v_bar):
        # This is only for piecewise linear function
        # v_bar = self.V_bar(s)
        new_bounds = v_bar.bounds.copy()
        new_polynomial_pieces = v_bar.polynomial_pieces.copy()
        min_v = v_bar(new_bounds[-1])
        for i in range(len(new_bounds) - 1, -1, -1):
            print(i)
            tmp = v_bar(new_bounds[i])
            if tmp < min_v:
                new_polynomial_pieces[i] = Poly(min_v, x)
            else:
                min_v = tmp
        return PiecewisePolynomial(new_polynomial_pieces, new_bounds)

    @property
    def initial_state(self):
        return self.__initial_state

    @property
    def terminal_state_set(self):
        return self.__terminal_state_set

