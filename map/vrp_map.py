
def link_two_nodes(node1, node2):
    node1.add_neighbours(node2)
    node2.add_neighbours(node1)


class Node(object):
    def __init__(self):
        self.neighbours = []

    def add_neighbour(self, neighbour):
        self.neighbours.append(neighbour)

class Map(object):
    def __init__(self, road_map, task_set):
        self.road_map = road_map
        self.task_set = task_set