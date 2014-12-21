"""mdp.py

Define general MDP components

"""


class State(object):
    id = 1

    def __init__(self, id, state):
        self.id = id
        self.state = state

    def __str__(self):
        return "(" + self.id + "," + self.state + ")"

