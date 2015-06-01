"""mdp.py

Define general MDP components

"""
import numpy.polynomial.polynomial
from exact_mdp.la.piecewise import *
# Constants
ABS = 0
REL = 1
# variable
# x = sympy.Symbol('x', real=True)
# y = sympy.Symbol('y', real=True)


def U_ABS(U, V):
    # print('P: ', P)
    # print('V: ', V)
    h = U * V
    # print('h: ', h)
    i = [p.integ() for p in h.polynomial_pieces]
    h = PiecewisePolynomial([
        P([sum([
            i[j](h.bounds[j + 1]) - i[j](h.bounds[j]) for j in range(len(h.polynomial_pieces))
        ])])
    ],
        [min(U.bounds[0], V.bounds[0]), max(U.bounds[-1], V.bounds[-1])])
    h.simplify()
    return h


def U_REL(U, V):
    print('urel u: ', U)
    print('urel v: ', V)
    lower = V.bounds[0]
    upper = V.bounds[-1]
    h = PiecewisePolynomial([P([0])], [lower, upper])
    for i in range(0, U.pieces):
        for j in range(0, V.pieces):
            f = U.polynomial_pieces[i]
            g = V.polynomial_pieces[j]
            if (f.degree() == 0 and f.coef[0] == 0) or (g.degree() == 0 and g.coef[0] == 0):
                print('zero')
                continue
            # print('U_REL', i, j, f, g)
            f_piece = (f, U.bounds[i], U.bounds[i + 1])
            g_piece = (g, V.bounds[j], V.bounds[j + 1])
            print('-----', i, j, '-----')
            # print('f: ', f)
            # print('g: ', g)
            piecewise_result = convolute_onepiece(f_piece, g_piece, lower, upper)
            print('conv: ', piecewise_result)
            h += piecewise_result
    h.simplify()
    return h


def convolute_onepiece(F, G, lower, upper):
    # This function is a modified version of the convolution function from
    # http://www.mare.ee/indrek/misc/convolution.pdf
    f, a_f, b_f = F
    g, a_g, b_g = G
    # special change for lazy approximation
    f = f(P([0, -1]))
    a_f, b_f = -b_f, -a_f
    print('f: ', f, ' a_f: ', a_f, ' b_f: ', b_f)
    print('g: ', g, ' a_g: ', a_g, ' b_g: ', b_g)
    # make sure ranges are in order, swap values if necessary
    if b_f - a_f > b_g - a_g:
        f, a_f, b_f, g, a_g, b_g = g, a_g, b_g, f, a_f, b_f
    bl = a_f + a_g
    bu = b_f + b_g
    if bl >= upper or bu <= lower:
        return PiecewisePolynomial([P([0])], [lower, upper])
    bl = lower if bl < lower else bl
    bu = upper if bu > upper else bu
    if a_f == b_f:
        return PiecewisePolynomial([f(a_f) * g(P([-a_f, 1]))], [bl, bu])
    else:
        cfy = f.coef  # f.subs(x, y)
        csub = P([1, -1])  # x-y where only y is seen as variable
        gda1 = g.degree() + 1
        cgxy = np.zeros((gda1, gda1))
        for i in range(gda1):
            cgxy[i][:i + 1] = (csub ** i).coef
        cgxy = (np.asarray([g.coef]).transpose() * cgxy).transpose()
        for i in range(1, gda1):  # g.subs(x, x - y)
            cgxy[i] = np.roll(cgxy[i], -i)
        fda1 = f.degree() + 1
        r, c = cgxy.shape
        fmg = np.zeros((r + fda1 - 1, c))
        for i in range(fda1):  # f.subs(x,y) * g.subs(x, x - y)
            fmg[i:i + r] = fmg[i:i + r] + cfy[i] * cgxy
        intfg = np.polynomial.polynomial.polyint(cgxy)  # integration
        b = [bl]
        p = []
        b1 = b_f + a_g
        b2 = a_f + b_g
        b1 = lower if b1 < lower else upper if b1 > upper else b1
        b2 = lower if b2 < lower else upper if b2 > upper else b2
        r, c = np.shape(intfg)
        if b[-1] < b1:
            b.append(b1)
            csub = P([-a_g, 1])
            intfgsub1 = P(intfg[0])
            intfgsub2 = P(intfg[0])
            for i in range(1, r):
                intfgsub1 += intfg[i] * csub ** i
                intfgsub2 + intfg[i] * a_f ** i
            p.append(intfgsub1 - intfgsub2)  # p.append(Poly(i.subs(y, x - a_g) - i.subs(y, a_f), x))
        if b[-1] < b2:
            b.append(b2)
            intfgsub1 = P(intfg[0])
            intfgsub2 = P(intfg[0])
            for i in range(1, r):
                intfgsub1 += intfg[i] * b_f ** i
                intfgsub2 + intfg[i] * a_f ** i
            p.append(intfgsub1 - intfgsub2)  # p.append(Poly(i.subs(y, b_f) - i.subs(y, a_f), x))
        if b[-1] < bu:
            b.append(bu)
            csub = P([-b_g, 1])
            intfgsub1 = P(intfg[0])
            intfgsub2 = P(intfg[0])
            for i in range(1, r):
                intfgsub1 += intfg[i] * b_f ** i
                intfgsub2 + intfg[i] * csub ** i
            p.append(intfgsub1 - intfgsub2)  # p.append(Poly(i.subs(y, b_f) - i.subs(y, x - b_g), x))
        return PiecewisePolynomial(p, b)


class State(object):
    # _value_function = None

    def __init__(self, label):
        self.label = label
        self.actions = dict()

    def __str__(self):
        return "(" + self.label + ")"

    def add_action(self, action, miu, likelihood):
        if action not in self.actions:
            self.actions[action] = dict()
        self.actions[action][miu] = likelihood

    def get_outcomes(self, action):
        return self.actions[action]

    @property
    def action_set(self):
        return self.actions.keys()


class Action(object):
    def __init__(self, label):
        self.label = label


class MDP(object):
    def __init__(self, states, mius, rewards,
                 initial_state, terminal_state_dict, time_horizon,
                 lazy=0, pwc=1, lazy_error_tolerance=0.1):
        self.states = states  # State set
        self.mius = mius
        self.rewards = rewards
        self.initial_state = initial_state
        self.terminal_state_dict = terminal_state_dict
        self.time_horizon = time_horizon
        self.u1 = dict()
        self.lazy = lazy
        self.pwc = pwc
        self.lazy_err_tol = lazy_error_tolerance
        self.reset_mdp()

    def reset_mdp(self):
        # self.__u1.clear()
        if self.lazy:
            if self.pwc:
                for m in self.mius:
                    self.mius[m] = (self.mius[m][0],
                                    self.mius[m][1],
                                    self.mius[m][2].to_pwc_function_approximation(self.lazy_err_tol))
                # print('miu!!!')
                # for s in self.states:
                # for a in s.action_set:
                # outcomes = s.get_outcomes(a)
                # for miu in outcomes:
                # print(miu)
                for m in self.rewards:
                    self.rewards[m] = self.rewards[m].to_pwc_function_approximation(self.lazy_err_tol)
                for state in self.states:
                    for action in state.actions:
                        for miu in state.actions[action]:
                            state.actions[action][miu] = state.actions[action][miu].to_pwc_function_approximation(
                                self.lazy_err_tol)
                    if state not in self.terminal_state_dict:
                        self.u1[state] = PiecewisePolynomial([P([0])], self.time_horizon)
                    else:
                        self.u1[state] = self.terminal_state_dict[state].to_pwc_function_approximation(
                            self.lazy_err_tol)
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
                    if state not in self.terminal_state_dict:
                        self.u1[state] = PiecewisePolynomial([P([0])], self.time_horizon)
                    else:
                        self.u1[state] = self.terminal_state_dict[state].to_constant_function_approximation()
        else:
            for state in self.states:
                if state not in self.terminal_state_dict:
                    self.u1[state] = PiecewisePolynomial([P([0])], self.time_horizon)
                    # print('debug:init', state, self.__u1[state])
                else:
                    self.u1[state] = self.terminal_state_dict[state]

    def value_iteration(self):
        # V(x,t) = sup_{t'>=t}(\int^{t'}_t K(x,s) ds + V_bar(x,t'))
        u1 = self.u1
        terminals = self.terminal_state_dict
        i = 0
        while True:
            print('========= iter =========: ', i)
            u0 = u1.copy()
            stop_flag = True
            for s in self.states:
                print('state: ', s)
                print('value pre: ', u0[s])
                if s in terminals:
                    continue
                u1[s] = self.state_value(s, u0)
                if stop_flag and u1[s] != u0[s]:
                    stop_flag = False
                print('value aft: ', u1[s])
            i += 1
            # print('iter',i)
            if stop_flag or i > 25:
                if i > 25:
                    print('i>25')
                return u0

    def state_value(self, s: State, v):
        # This is only for piecewise linear function
        # compute v_bar
        act_set = list(s.action_set)
        if len(act_set) == 1:
            V_bar = self.q(s, act_set[0], self.rewards, v)
        else:
            best_pw = self.q(s, act_set[0], self.rewards, v)
            print('===ACT', act_set[0])
            print('===BEST', best_pw)
            for a in act_set[1:]:
                qq = self.q(s, a, self.rewards, v)
                best_pw = max_piecewise(best_pw, qq)
                print('===ACT', a)
                print('===BEST', qq)
            best_pw.simplify()
            V_bar = best_pw
        # for dawdling
        # v_b = v_bar(s, self.rewards, v)
        new_bounds = V_bar.bounds.copy()
        new_polynomial_pieces = V_bar.polynomial_pieces.copy()
        min_v = V_bar(new_bounds[-1])
        count = len(new_bounds) - 2  # from the penultimate turning point
        while count > 0:
            # print('count: ', count)
            tmp = V_bar(new_bounds[count])
            if tmp < min_v:
                # roots = sympy.solve(new_polynomial_pieces[count] - new_polynomial_pieces[count - 1], x)
                diff_p = new_polynomial_pieces[count] - new_polynomial_pieces[count - 1]
                # print('diffp:', diff_p)
                # roots = diff_p.real_roots()
                roots = np.roots(diff_p.coef)
                roots = roots.real[abs(roots.imag) < 1e-5]
                print('roots:', roots)
                if len(roots):
                    root = float(roots[0])
                    if new_bounds[count - 1] < root < new_bounds[count + 1]:
                        new_bounds[count] = root
                        new_polynomial_pieces[count] = P(min_v)
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
        Q = PiecewisePolynomial([P([0])], self.time_horizon)
        for miu in outcomes:
            print('state: ', s, '-', self.mius[miu][0], 'abs/rel: ', self.mius[miu][1], 'prob: ', self.mius[miu][2], 'outcomes: ', outcomes[miu])
            if self.mius[miu][1] == ABS:
                U = r[miu] + U_ABS(self.mius[miu][2], v[self.mius[miu][0]])
            elif self.mius[miu][1] == REL:
                U = r[miu] + U_REL(self.mius[miu][2], v[self.mius[miu][0]])
            else:
                raise ValueError('The type of the miu time distribution function is wrong')
            print('===U', U)
            Q += outcomes[miu] * U
        if self.lazy:
            if self.pwc:
                Q = Q.to_pwc_function_approximation(self.lazy_err_tol)
            else:
                Q = Q.to_constant_function_approximation()
        # return sum([outcomes[miu] * u(miu, r, v) for miu in outcomes])
        return Q
