
def link_two_nodes(node1, node2, travel_cost1, travel_cost2):
    node1.add_neighbours(node2, travel_cost1)
    node2.add_neighbours(node1, travel_cost2)


class Node(object):
    def __init__(self):
        self.neighbours = dict()

    def add_neighbour(self, neighbour, travel_cost):
        self.neighbours[neighbour] = travel_cost

class Map(object):
    def __init__(self, road_map, task_set):
        self.road_map = road_map
        self.task_set = task_set