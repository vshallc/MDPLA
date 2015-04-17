import itertools as it
import exact_mdp.mdp as emdp
import random
from utils.math_tools import *
from exact_mdp.la.piecewise import *

# 1 unit is about 6 min


def vrp2mdp(vrp):
    states = dict()
    mius = dict()
    rewards = dict()
    terminal_states = dict()
    rmap = vrp.road_map
    tasks = vrp.tasks
    rng = range(len(tasks))
    lrng = range(len(tasks) + 1)
    row_max = len(rmap)
    col_max = len(rmap[0])
    for l in lrng:
        all_combs = it.combinations(rng, l)
        for comb in all_combs:
            comb = list(comb)
            for r in range(row_max):
                for c in range(col_max):
                    label = make_label(rmap[r][c], tasks, comb)
                    states[label] = emdp.State(label)
    for key in states:
        print(key)
    for l in lrng:
        all_combs = it.combinations(rng, l)
        for comb in all_combs:
            comb = list(comb)
            for r in range(row_max):
                for c in range(col_max):
                    node_from = rmap[r][c]
                    label_from = make_label(node_from, tasks, comb)
                    state_from = states[label_from]
                    # move
                    for node_to in node_from.neighbours:
                        label_to = make_label(node_to, tasks, comb)
                        state_to = states[label_to]
                        link_label = label_from + '-' + label_to
                        for traffic in node_from.neighbours[node_to]:
                            miu_label = link_label + ':' + str(traffic.label)
                            mius[miu_label] = (state_to, emdp.REL, traffic.distribution)
                            rewards[miu_label] = PiecewisePolynomial([Poly('0', x)], node_from.timespan)
                            state_from.add_action('move:' + link_label, miu_label, traffic.likelihood)
                    for t in comb:
                        if tasks[t].location == r * col_max + c:
                            comb_to = comb.copy()
                            print(comb_to)
                            comb_to.remove(t)
                            print(comb_to)
                            label_to = make_label(node_from, tasks, comb_to)
                            state_to = states[label_to]
                            link_label = label_from + '-' + label_to
                            # execute_success
                            miu_label = link_label + ':success'
                            mius[miu_label] = (state_to, emdp.REL, tasks[t].distribution_success)
                            rewards[miu_label] = tasks[t].reward_success
                            state_from.add_action('execute:' + link_label, miu_label, tasks[t].likelihood_success)
                            # execute_failure
                            miu_label = link_label + ':failure'
                            mius[miu_label] = (state_to, emdp.REL, tasks[t].distribution_failure)
                            rewards[miu_label] = tasks[t].reward_failure
                            state_from.add_action('execute:' + link_label, miu_label, tasks[t].likelihood_failure)
                            # abandon
                            miu_label = link_label + ':penalty'
                            mius[miu_label] = (state_to, emdp.REL, tasks[t].distribution_penalty)
                            rewards[miu_label] = tasks[t].reward_penalty
                            state_from.add_action('abandon:' + link_label, miu_label, tasks[t].likelihood_penalty)
    initial_label = make_label(vrp.company, tasks, [i for i in rng])
    initial_state = states[initial_label]
    terminal_label = make_label(vrp.company, tasks, [])
    terminal_states[states[terminal_label]] = vrp.terminate_function
    state_list=[states[key] for key in states]
    return emdp.MDP(state_list, mius, rewards,
                    initial_state, terminal_states, vrp.timespan,
                    lazy=1, pwc=1, lazy_error_tolerance=0.1)


def make_label(node, tasks, task_comb):
    # return str(node.label)+str([tasks[i] for i in task_comb])
    return str(node.label) + str(task_comb)


def random_vrp(row, col, task_num, timespan=None, terminate_function=None):
    if timespan is None:
        timespan = [0, 40]
        terminate_function = PiecewisePolynomial([Poly('0', x), Poly(-10 * x + 200, x), Poly(float('-10000'), x)],
                                                 [0, 20, 30, 40])
    company_row = random.randint(0, row - 1)
    company_col = random.randint(0, col - 1)
    return VRP(random_map(row, col, timespan),
               random_tasks(task_num, row, col, timespan),
               timespan, terminate_function,
               company_row, company_col)


def random_node(label, timespan,
                avg_travel_cost_rush=5, avg_travel_cost_rush_error=2,
                avg_travel_cost_off=2, avg_travel_cost_off_error=1,
                avg_rush_length=7, avg_rush_length_error=3, avg_rush_span_margin=4):
    mu_rush = avg_travel_cost_rush + random.random() * avg_travel_cost_rush_error * 2 - avg_travel_cost_rush_error
    mu_off = avg_travel_cost_off + random.random() * avg_travel_cost_off_error * 2 - avg_travel_cost_off_error
    sigma_rush, sigma_off = 2 ** (random.random() * 2 - 1), 2 ** (random.random() * 2 - 1)
    if sigma_rush < sigma_off:
        sigma_rush, sigma_off = sigma_off, sigma_rush
    rush_length = avg_rush_length + random.random() * avg_rush_length_error * 2 - avg_rush_length_error
    rush_span_margin = avg_rush_span_margin * (2 ** (random.random() * 2 - 1))
    rush_span_begin = timespan[0] + random.random() * (timespan[1] - timespan[0] - 2 * rush_span_margin - rush_length)
    rush_span = [rush_span_begin,
                 rush_span_begin + rush_span_margin,
                 rush_span_begin + rush_span_margin + rush_length,
                 rush_span_begin + rush_span_margin * 2 + rush_length]
    return Node(label, mu_rush, mu_off, sigma_rush, sigma_off, rush_span, timespan)


def random_map(row, col, timespan):
    rand_map = []
    for r in range(row):
        rand_map.append([])
        for c in range(col):
            label = str(r * col + c)
            rand_map[r].append(random_node(label, timespan))
    for r in range(row - 1):
        for c in range(col):
            link_two_nodes(rand_map[r][c], rand_map[r + 1][c])
    for r in range(row):
        for c in range(col - 1):
            link_two_nodes(rand_map[r][c], rand_map[r][c + 1])
    return rand_map


def random_task(label, row, col, timespan):
    r = int(random.random() * row)
    c = int(random.random() * col)
    location = r * col + c
    # deadline = random.random() * (timespan[1] - timespan[0]) + timespan[0]
    reward = random.randint(1, 3)
    window_left = random.random() * (timespan[1] - timespan[0]) + timespan[0]
    window_right = random.random() * (timespan[1] - timespan[0]) + timespan[0]
    if window_left > window_right:
        window_left, window_right = window_right, window_left
    # reward_function = PiecewisePolynomial([Poly(reward, x), Poly(float('-inf'), x)],
    # [timespan[0], deadline, timespan[1]])
    # penalty = PiecewisePolynomial([Poly(1, x)], timespan)
    # time_cost_distribution = PiecewisePolynomial([Poly('1', x)], [0.5, 1.5])
    return Task(label, location, reward, 1, [0.5, 1.5], timespan, [window_left, window_right])


def random_tasks(task_num, row, col, timespan=None):
    return [random_task(str(i), row, col, timespan) for i in range(task_num)]


def link_two_nodes(node1, node2):
    node1.add_neighbour(node2)
    node2.add_neighbour(node1)


def travel_outcomes(from_node, to_node):
    # result = dict()
    timespan = from_node.timespan
    rush_span = (np.asarray(from_node.rush_span) + np.asarray(to_node.rush_span)) / 2
    k1 = 1 / (rush_span[1] - rush_span[0])
    k2 = 1 / (rush_span[3] - rush_span[2])
    # rush hour
    # likelihood
    l1 = Poly('0', x)
    l2 = Poly(sympy.sympify(k1 * (x - rush_span[0]), x))
    l3 = Poly('1', x)
    l4 = Poly(sympy.sympify(-k2 * (x - rush_span[3]), x))
    l5 = Poly('0', x)
    pw_rush = [l1, l2, l3, l4, l5]
    bd_rush = timespan[:]
    bd_rush[1:1] = rush_span
    likelihood_rush = PiecewisePolynomial(pw_rush, bd_rush)
    # distribution
    mu_rush = (from_node.seed_mu_rush + to_node.seed_mu_rush) / 2
    sigma_rush = math.sqrt(from_node.seed_sigma_rush * to_node.seed_sigma_rush)
    distribution_rush = norm_pdf_linear_approximation(mu_rush, sigma_rush, from_node.timespan)
    traffic_rush = Traffic('rush', likelihood_rush, distribution_rush)
    # off peak
    # likelihood
    l1 = Poly('1', x)
    l2 = Poly(sympy.sympify(-k1 * (x - rush_span[1]), x))
    l3 = Poly('0', x)
    l4 = Poly(sympy.sympify(k2 * (x - rush_span[2]), x))
    l5 = Poly('1', x)
    pw_off = [l1, l2, l3, l4, l5]
    bd_off = timespan[:]
    bd_off[1:1] = rush_span
    likelihood_off = PiecewisePolynomial(pw_off, bd_off)
    # distribution
    mu_off = (from_node.seed_mu_off + to_node.seed_mu_off) / 2
    sigma_off = math.sqrt(from_node.seed_sigma_off * to_node.seed_sigma_off)
    distribution_off = norm_pdf_linear_approximation(mu_off, sigma_off, from_node.timespan)
    traffic_off = Traffic('off', likelihood_off, distribution_off)
    return [traffic_rush, traffic_off]


class Node(object):
    def __init__(self, label, seed_mu_rush, seed_mu_off, seed_sigma_rush, seed_sigma_off, seed_rush_span, timespan):
        self.neighbours = dict()
        self.label = label
        self.seed_mu_rush = seed_mu_rush
        self.seed_mu_off = seed_mu_off
        self.seed_sigma_rush = seed_sigma_rush
        self.seed_sigma_off = seed_sigma_off
        self.rush_span = seed_rush_span
        self.timespan = timespan

    def add_neighbour(self, neighbour):
        self.neighbours[neighbour] = travel_outcomes(self, neighbour)


class Traffic(object):
    def __init__(self, label, likelihood, distribution):
        self.label = label
        self.likelihood = likelihood
        self.distribution = distribution


class Task(object):
    def __init__(self, label, location, reward, penalty, timecostspan, timespan, time_window):
        self.label = label
        self.location = location
        # self.reward = reward
        self.penalty = penalty
        # self.time_cost = time_cost
        self.reward_success = PiecewisePolynomial([Poly(reward, x)], timespan)
        self.reward_failure = PiecewisePolynomial([Poly('0', x)], timespan)
        bounds = timespan.copy()
        bounds[1:1] = time_window
        self.likelihood_success = PiecewisePolynomial([Poly('0', x), Poly('1', x), Poly('0', x)], bounds.copy())
        self.likelihood_failure = PiecewisePolynomial([Poly('1', x), Poly('0', x), Poly('1', x)], bounds.copy())
        self.distribution_success = PiecewisePolynomial([Poly(1 / (timecostspan[1] - timecostspan[0]), x)],
                                                        timecostspan)
        self.distribution_failure = PiecewisePolynomial([Poly(1 / (timecostspan[1] - timecostspan[0]), x)],
                                                        timecostspan)
        # penalty
        self.reward_penalty = PiecewisePolynomial([Poly(penalty, x)], timespan)
        self.likelihood_penalty = PiecewisePolynomial([Poly('1', x)], timespan)
        self.distribution_penalty = PiecewisePolynomial([Poly('0', x)], [0, 0])


class VRP(object):
    def __init__(self, road_map, tasks, timespan, terminate_function, company_row, company_col):
        self.road_map = road_map
        self.tasks = tasks
        self.timespan = timespan
        self.terminate_function = terminate_function
        self.company = road_map[company_row][company_col]
