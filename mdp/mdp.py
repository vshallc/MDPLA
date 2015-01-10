"""mdp.py

Define general MDP components

"""

# Constants
ABS = 0
REL = 1

class State(object):
    #_value_function = None
    __action = dict()

    def __init__(self, label):
        self.__label = label

    def __str__(self):
        return "(" + self.label + ")"

    def add_action(self, action, outcome, likelihood):
        if action not in self.__action:
            self.__action[action] = dict()
        self.__action[action][outcome] = likelihood

    @property
    def label(self):
        return self.__label

    @property
    def value_function(self):
        return self._value_function

    @value_function.setter
    def value_function(self, value_function):
        self._value_function = value_function


class Action(object):

    def __init__(self, label):
        self.__label = label


class Outcome(object):

    def __init__(self, result_state):
        self.__result_state = result_state

    @property
    def result_state(self):
        return self.__result_state


class MDP(object):
    def __init__(self, S, A, miu, R_s, R_a, R_d,
                 initial_state, terminal_state_set):
        self.__S = S    # State set
        self.__A = A      # Action set
        self.__miu = miu
        self.__R_s = R_s    # Reward of start time
        self.__R_a = R_a    # Reward of arrival time
        self.__R_d = R_d    # Reward of duration
        self.__initial_state = initial_state
        self.__terminal_state_set = terminal_state_set
        # self.__current_state = initial_state

    # def reset(self):
    #    self.__current_state = self.__initial_state

    def T(self, state, action):
        if state in self.__terminal_state_set:
            return {None: 1.0}
        else:
            return state.perform(action)

    def R(self, i_miu, d):
        return self.__R_s[i_miu] + self.__R_a[i_miu] + self.__R_d[i_miu]

    def U(self, i_miu, t):
        if self.__miu[i_miu][1] == REL:
            pass
        elif self.__miu[i_miu] == ABS:
            pass
        else:
            raise ValueError('The type of the miu time distribution is wrong')

    @property
    def initialState(self):
        return self.__initial_state

    @property
    def terminalStateSet(self):
        return self.__terminal_state_set

