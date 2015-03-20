import random
from math_tools import *
from exact_mdp.la.piecewise import *

# 1 unit is about 6 min


def link_two_nodes(node1, node2):
    node1.add_neighbours(node2)
    node2.add_neighbours(node1)


def random_node(label, timespan=None,
                avg_travel_cost_rush=5, avg_travel_cost_rush_error=2,
                avg_travel_cost_off=2, avg_travel_cost_off_error=1,
                avg_rush_length=7, avg_rush_length_error=3, avg_rush_span_margin=4):
    if timespan is None:
        timespan = [0, 40]
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


def random_map(row, col):
    rand_map = []
    for r in range(row):
        rand_map.append([])
        for c in range(col):
            label = str(r*col+c)
            rand_map[r].append(random_node(label))
    for r in range(row-1):
        for c in range(col-1):
            link_two_nodes(rand_map[r][c], rand_map[r][c+1])
            link_two_nodes(rand_map[r][c], rand_map[r+1][c])
    return rand_map


def traffic_likelihood(from_node, to_node):
    timespan = from_node.timespan
    rush_span = (np.asarray(from_node.rush__span) + np.asarray(to_node.rush_span)) / 2
    k1 = 1 / (rush_span[1] - rush_span[0])
    k2 = 1 / (rush_span[3] - rush_span[2])
    # rush hour
    l1 = Poly('0', x)
    l2 = Poly(sympy.sympify(k1 * (x - rush_span[0]), x))
    l3 = Poly('1', x)
    l4 = Poly(sympy.sympify(-k2 * (x - rush_span[3]), x))
    l5 = Poly('0', x)
    pw_rush = [l1, l2, l3, l4, l5]
    bd_rush = timespan[:]
    bd_rush[1:1] = rush_span
    # off peak
    l1 = Poly('1', x)
    l2 = Poly(sympy.sympify(-k1 * (x - rush_span[1]), x))
    l3 = Poly('0', x)
    l4 = Poly(sympy.sympify(k2 * (x - rush_span[2]), x))
    l5 = Poly('1', x)
    pw_off = [l1, l2, l3, l4, l5]
    bd_off = timespan[:]
    bd_off[1:1] = rush_span
    return PiecewisePolynomial(pw_rush, bd_rush), PiecewisePolynomial(pw_off, bd_off)


def travel_outcomes(from_node, to_node):
    outcomes = dict()
    # traffic_rush = from_node.label + '-' + to_node.label + '_rush'
    # traffic_off = from_node.label + '-' + to_node.label + '_off'
    mu = (from_node.seed_mu_rush + to_node.seed_mu_rush) / 2
    sigma = math.sqrt(from_node.seed_sigma_rush * to_node.seed_sigma_rush)
    pw = norm_pdf_linear_approximation(mu, sigma, from_node.timespan)
    outcomes['rush'] = pw
    mu = (from_node.seed_mu_off + to_node.seed_mu_off) / 2
    sigma = math.sqrt(from_node.seed_sigma_off * to_node.seed_sigma_off)
    pw = norm_pdf_linear_approximation(mu, sigma, from_node.timespan)
    outcomes['off'] = pw
    return outcomes


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
        self.neighbours[neighbour] = (traffic_likelihood(self, neighbour), travel_outcomes(self, neighbour))


class Map(object):
    def __init__(self, road_map, task_set):
        self.road_map = road_map
        self.task_set = task_set