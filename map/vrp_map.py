from math_tools import *
from exact_mdp.la.piecewise import *


def link_two_nodes(node1, node2, travel_cost1, travel_cost2):
    node1.add_neighbours(node2, travel_cost1)
    node2.add_neighbours(node1, travel_cost2)


def random_rush_likelihood(timespan, rush_span):
    k1 = 1/(rush_span[1]-rush_span[0])
    k2 = 1/(rush_span[3]-rush_span[2])
    # rush hour
    l1 = Poly('0', x)
    l2 = Poly(sympy.sympify(k1*(x-rush_span[0]), x))
    l3 = Poly('1', x)
    l4 = Poly(sympy.sympify(-k2*(x-rush_span[3]), x))
    l5 = Poly('0', x)
    pw_rush = [l1, l2, l3, l4, l5]
    bd_rush = timespan.copy()
    bd_rush[1:1] = rush_span
    # off peak
    l1 = Poly('1', x)
    l2 = Poly(sympy.sympify(-k1*(x-rush_span[1]), x))
    l3 = Poly('0', x)
    l4 = Poly(sympy.sympify(k2*(x-rush_span[2]), x))
    l5 = Poly('1', x)
    pw_off = [l1, l2, l3, l4, l5]
    bd_off = timespan.copy()
    bd_off[1:1] = rush_span
    return PiecewisePolynomial(pw_rush, bd_rush), PiecewisePolynomial(pw_off, bd_off)


def travel_outcomes(from_node, to_node):
    outcomes = dict()
    # traffic_rush = from_node.label + '-' + to_node.label + '_rush'
    # traffic_off = from_node.label + '-' + to_node.label + '_off'
    mu = (from_node.seed_mu_rush + to_node.seed_mu_rush)/2
    sigma = math.sqrt(from_node.seed_sigma_rush * to_node.seed_sigma_rush)
    pw = norm_pdf_linear_approximation(mu, sigma, from_node.timespan)
    outcomes['rush'] = pw
    mu = (from_node.seed_mu_off + to_node.seed_mu_off)/2
    sigma = math.sqrt(from_node.seed_sigma_off * to_node.seed_sigma_off)
    pw = norm_pdf_linear_approximation(mu, sigma, from_node.timespan)
    outcomes['off'] = pw
    return outcomes

# def travel_cost_distribution(seed_mu, seed_sigma, timespan):
#     return math_tools.norm_pdf_linear_approximation(seed_mu, seed_sigma, timespan)


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


class Map(object):
    def __init__(self, road_map, task_set):
        self.road_map = road_map
        self.task_set = task_set