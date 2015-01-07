"""mdp.py

Define general MDP components

"""


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
    def __init__(self, state_space, action_set, initial_state, terminal_state_set):
        self.__state_space = state_space
        self.__action_set = action_set
        self.__initial_state = initial_state
        self.__terminal_state_set = terminal_state_set
        #self.__current_state = initial_state

    #def reset(self):
    #    self.__current_state = self.__initial_state

    def transition(self, state, action):
        if state in self.__terminal_state_set:
            return {None: 1.0}
        else:
            return state.perform(action)

    @property
    def initialState(self):
        return self.__initial_state

    @property
    def terminalStateSet(self):
        return self.__terminal_state_set

