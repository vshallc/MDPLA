"""mdp.py

Define general MDP components

"""
import sympy
from exact_mdp.la.piecewise import *
# Constants
ABS = 0
REL = 1
# variable
# x = sympy.Symbol('x', real=True)
y = sympy.Symbol('y', real=True)


def U_ABS(t, P, V):
    # print('P: ', P)
    # print('V: ', V)
    h = P * V
    # print('h: ', h)
    i = [sympy.integrate(p.as_expr(), t) for p in h.polynomial_pieces]
    return PiecewisePolynomial([Poly(sum([i[j].subs(t, h.bounds[j + 1]) - i[j].subs(t, h.bounds[j])
                                          for j in range(len(h.polynomial_pieces))]), t)],
                               [min(P.bounds[0], V.bounds[0]), max(P.bounds[-1], V.bounds[-1])])


def U_REL(t, P, V):
    lower = V.bounds[0]
    upper = V.bounds[-1]
    h = PiecewisePolynomial([Poly('0', t)], [lower, upper])
    for i in range(0, P.pieces):
        for j in range(0, V.pieces):
            f = P.polynomial_pieces[i]
            g = V.polynomial_pieces[j]
            if f.is_zero or g.is_zero:
                continue
            f_piece = (f, P.bounds[i], P.bounds[i + 1])
            g_piece = (g, V.bounds[j], V.bounds[j + 1])
            piecewise_result = convolute_onepiece(t, f_piece, g_piece, lower, upper)
            h += piecewise_result
    return h


def convolute_onepiece(x, F, G, lower, upper):
    # This function is a modified version of the convolution function from
    # http://www.mare.ee/indrek/misc/convolution.pdf
    f, a_f, b_f = F
    g, a_g, b_g = G
    f = f.as_expr()
    g = g.as_expr()
    # special change for lazy approximation
    f = f.subs(x, -x)
    a_f, b_f = -b_f, -a_f
    # print('f: ', f, ' a_f: ', a_f, ' b_f: ', b_f)
    # print('g: ', g, ' a_g: ', a_g, ' b_g: ', b_g)
    # make sure ranges are in order, swap values if necessary
    if b_f - a_f > b_g - a_g:
        f, a_f, b_f, g, a_g, b_g = g, a_g, b_g, f, a_f, b_f
    bl = a_f + a_g
    bu = b_f + b_g
    if bl >= upper or bu <= lower:
        return PiecewisePolynomial([Poly('0', x)], [lower, upper])
    bl = lower if bl < lower else bl
    bu = upper if bu > upper else bu
    if a_f == b_f:
        return PiecewisePolynomial([Poly(f.subs(x, a_f) * g.subs(x, x - a_f), x)], [bl, bu])
    else:
        y = sympy.Dummy('y')
        i = sympy.integrate(f.subs(x, y) * g.subs(x, x - y), y)
        b = [bl]
        p = []
        b1 = b_f + a_g
        b2 = a_f + b_g
        b1 = lower if b1 < lower else upper if b1 > upper else b1
        b2 = lower if b2 < lower else upper if b2 > upper else b2
        if b[-1] < b1:
            b.append(b1)
            p.append(Poly(i.subs(y, x - a_g) - i.subs(y, a_f), x))
        if b[-1] < b2:
            b.append(b2)
            p.append(Poly(i.subs(y, b_f) - i.subs(y, a_f), x))
        if b[-1] < bu:
            b.append(bu)
            p.append(Poly(i.subs(y, b_f) - i.subs(y, x - b_g), x))
        return PiecewisePolynomial(p, b)


class State(object):
    # _value_function = None

    def __init__(self, label):
        self.__label = label
        self.actions = dict()

    def __str__(self):
        return "(" + self.label + ")"

    def add_action(self, action, miu, likelihood):
        if action not in self.actions:
            self.actions[action] = dict()
        self.actions[action][miu] = likelihood

    @property
    def label(self):
        return self.__label

    def get_outcomes(self, action):
        return self.actions[action]

    @property
    def action_set(self):
        return self.actions.keys()


class Action(object):
    def __init__(self, label):
        self.__label = label


class MDP(object):
    def __init__(self, states, mius, rewards,
                 initial_state, terminal_state_dict, time_horizon,
                 lazy=0, pwc=1, lazy_error_tolerance=0.1):
        self.states = states  # State set
        # self.__A = A      # Action set
        self.mius = mius
        # self.__R_s = R_s  # Reward of start time
        # self.__R_a = R_a  # Reward of arrival time
        # self.__R_d = R_d  # Reward of duration
        self.rewards = rewards
        self.__initial_state = initial_state
        self.__terminal_state_dict = terminal_state_dict
        self.__time_horizon = time_horizon
        self.__u1 = dict()
        self.__lazy = lazy
        self.__pwc = pwc
        self.__lazy_err_tol = lazy_error_tolerance
        self.reset_mdp()

    def reset_mdp(self):
        # self.__u1.clear()
        if self.__lazy:
            if self.__pwc:
                for m in self.mius:
                    self.mius[m] = (self.mius[m][0],
                                   self.mius[m][1],
                                   self.mius[m][2].to_pwc_function_approximation(self.__lazy_err_tol))
                # print('miu!!!')
                # for s in self.states:
                #     for a in s.action_set:
                #         outcomes = s.get_outcomes(a)
                #         for miu in outcomes:
                #             print(miu)
                # print('miu-')
                for m in self.rewards:
                    self.rewards[m] = self.rewards[m].to_pwc_function_approximation(self.__lazy_err_tol)
                for state in self.states:
                    for action in state.actions:
                        for miu in state.actions[action]:
                            state.actions[action][miu] = state.actions[action][miu].to_pwc_function_approximation(
                                self.__lazy_err_tol)
                    if state not in self.__terminal_state_dict:
                        self.__u1[state] = PiecewisePolynomial([Poly('0', x)], self.__time_horizon)
                    else:
                        self.__u1[state] = self.__terminal_state_dict[state].to_pwc_function_approximation(
                            self.__lazy_err_tol)
            else:
                for m in self.mius:
                    self.mius[m] = (self.mius[m][0],
                                   self.mius[m][1],
                                   self.mius[m][2].to_constant_function_approximation())
                for m in self.rewards:
                    self.rewards[m] = self.rewards[m].to_constant_function_approximation()
                for state in self.states:
                    for action in state.actions:
                        for miu in state.actions[action]:
                            state.actions[action][miu] = state.actions[action][miu].to_constant_function_approximation()
                    if state not in self.__terminal_state_dict:
                        self.__u1[state] = PiecewisePolynomial([Poly('0', x)], self.__time_horizon)
                    else:
                        self.__u1[state] = self.__terminal_state_dict[state].to_constant_function_approximation()
        else:
            for state in self.states:
                if state not in self.__terminal_state_dict:
                    self.__u1[state] = PiecewisePolynomial([Poly('0', x)], self.__time_horizon)
                else:
                    self.__u1[state] = self.__terminal_state_dict[state]

    @property
    def initial_state(self):
        return self.__initial_state

    @property
    def terminal_state_dict(self):
        return self.__terminal_state_dict

    def value_iteration(self):
        # V(x,t) = sup_{t'>=t}(\int^{t'}_t K(x,s) ds + V_bar(x,t'))
        u1 = self.__u1
        # r = self.rewards
        terminals = self.terminal_state_dict
        i = 0
        while True:
            u0 = u1.copy()
            stop_flag = True
            for s in self.states:
                if s in terminals:
                    continue
                print('state: ', s)
                u1[s] = self.state_value(s, u0)
                if u1[s] != u0[s]:
                    stop_flag = False
            i += 1
            print('iter',i)
            if stop_flag or i > 5:
                return u0

    def state_value(self, s: State, v):
        # This is only for piecewise linear function
        # compute v_bar
        act_set = list(s.action_set)
        if len(act_set) == 1:
            V_bar = self.q(s, act_set[0], self.rewards, v)
        else:
            best_pw = self.q(s, act_set[0], self.rewards, v)
            for a in act_set[1:]:
                best_pw = max_piecewise(x, best_pw, self.q(s, a, self.rewards, v))
            V_bar = best_pw
        # for dawdling
        # v_b = v_bar(s, self.rewards, v)
        new_bounds = V_bar.bounds.copy()
        new_polynomial_pieces = V_bar.polynomial_pieces.copy()
        min_v = V_bar(new_bounds[-1])
        count = len(new_bounds) - 2  # from the penultimate turning point
        while count > 0:
            print('count: ', count)
            tmp = V_bar(new_bounds[count])
            if tmp < min_v:
                # roots = sympy.solve(new_polynomial_pieces[count] - new_polynomial_pieces[count - 1], x)
                diff_p = new_polynomial_pieces[count] - new_polynomial_pieces[count - 1]
                roots = diff_p.real_roots()
                if roots:
                    print('bef r')
                    root = float(roots[0])
                    print('aft r')
                    if new_bounds[count - 1] < root < new_bounds[count + 1]:
                        new_bounds[count] = root
                        new_polynomial_pieces[count] = Poly(min_v, x)
                        count -= 1
                        continue
                del new_polynomial_pieces[count - 1]
                del new_bounds[count]
                count -= 1
            else:
                min_v = tmp
                count -= 1
        return PiecewisePolynomial(new_polynomial_pieces, new_bounds)

    def q(self, s, a, r, v):
        outcomes = s.get_outcomes(a)
        # print('q state: ', s, ' action: ', a)
        # for o in outcomes:
        # print('outcome: ', o[0], ' ', o[1], ' ', o[2])
        # for m in outcomes:
        # print('state: ', s, '-', m[0], 'abs/rel: ', m[1], 'prob: ', m[2], 'outcomes: ', outcomes[m])
        Q = PiecewisePolynomial([Poly('0', x)], self.__time_horizon)
        for miu in outcomes:
            # print('state: ', s, '-', self.mius[miu][0], 'abs/rel: ', self.mius[miu][1], 'prob: ', self.mius[miu][2], 'outcomes: ', outcomes[miu])
            if self.mius[miu][1] == ABS:
                U = r[miu] + U_ABS(x, self.mius[miu][2], v[self.mius[miu][0]])
            elif self.mius[miu][1] == REL:
                # print('r',r[miu])
                U = r[miu] + U_REL(x, self.mius[miu][2], v[self.mius[miu][0]])
            else:
                raise ValueError('The type of the miu time distribution function is wrong')
            Q += outcomes[miu] * U
            # print('m', self.mius[miu][2])
            # print('v', v[self.mius[miu][0]])
            # print('U', U)
            # print('Q', Q)
        if self.__lazy:
            if self.__pwc:
                Q = Q.to_pwc_function_approximation(self.__lazy_err_tol)
            else:
                Q = Q.to_constant_function_approximation()
        # return sum([outcomes[miu] * u(miu, r, v) for miu in outcomes])
        return Q

        # def v_bar(s: State, r, v):
        #     act_set = list(s.action_set)
        #     if len(act_set) == 1:
        #         # res = q(s, act_set[0], r, v)
        #         # print('act: ', act_set[0], ' ', res)
        #         # return res
        #         return q(s, act_set[0], r, v)
        #     else:
        #         best_pw = q(s, act_set[0], r, v)
        #         # print('act: ', act_set[0], ' ', best_pw)
        #         for a in act_set[1:]:
        #             best_pw = max_piecewise(x, best_pw, q(s, a, r, v))
        #             # print('act: ', a, ' ', best_pw)
        #         # print('best_pw: ', best_pw)
        #         return best_pw


        # def u(miu, r, v):
        #     if miu[1] == ABS:
        #         # res = r[miu] + U_ABS(x, miu[2], v[miu[0]])
        #         # print('ABS: ', res)
        #         # return res
        #         return r[miu] + U_ABS(x, miu[2], v[miu[0]])
        #     elif miu[1] == REL:
        #         # res = r[miu] + U_REL(x, miu[2], v[miu[0]])
        #         # print('REL: ', res)
        #         # return res
        #         return r[miu] + U_REL(x, miu[2], v[miu[0]])
        #     else:
        #         raise ValueError('The type of the miu time distribution function is wrong')