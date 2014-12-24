"""mdp.py

Define general MDP components

"""


class State(object):
    __label = None
    __state = None

    def __init__(self, label, state):
        self.__label = label
        self.__state = state

    def __str__(self):
        return "(" + self.label + "," + self.state + ")"

    @property
    def label(self):
        return self.__label

    @property
    def state(self):
        return self.__state


class Action(object):
    pass


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

