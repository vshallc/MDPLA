class State(object):
    def __init__(self, label, time):
        self.__label = label
        self.__time = time
        self.__act_dict = {}

    def __str__(self):
        return '(' + self.__label + ',' + self.time + ')'

    def add_action(self, action, outcomes):
        self.__act_dict[action] = outcomes

    def print_detail(self):
        print(str(self))
        for a in self.__act_dict:
            act = self.__act_dict[a]
            # print(len(act[0]))
            for i in range(len(act)):
                # print(str(a) + ':' + str(i))
                print(str(a) + ':' + str(act[i][0]) + ' ' + str(act[i][1]))

    @property
    def label(self):
        return self.__label
    
    @property
    def time(self):
        return self.__time

    @property
    def act_dict(self):
        return self.__act_dict


class MDP(object):
    def __init__(self, states, rewards, initial, terminals):
        self.__states = states
        self.__rewards = rewards
        self.__initial = initial
        self.__terminals = terminals

    @property
    def states(self):
        return self.__states

    @property
    def rewards(self):
        return self.__rewards

    @property
    def initial(self):
        return self.__initial

    @property
    def terminals(self):
        return self.__terminals


def value_iteration(mdp):
    u1 = dict([(s, 0) for s in mdp.states])
    r = mdp.rewards
    terminals = mdp.terminals
    while True:
        u = u1.copy()
        delta = 0.0
        for s in mdp.states:
            # print(s, 'size', len(s.act_dict))
            if s in terminals:
                continue
            u1[s] = max([sum([p * (r[s][s1] + u[s1]) for (p, s1) in s.act_dict[a]]) for a in s.act_dict])
            delta = max(delta, abs(u1[s] - u[s]))
        if delta == 0:
            return u