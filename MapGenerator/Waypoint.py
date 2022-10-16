import numpy as np
import json


class Waypoint:
    LOOKUP = {}

    def __init__(self, ID, position, neighbors, *args, **kwargs):
        self.id = ID
        self.position = np.array(position)
        self.neighbors = neighbors
        self.forward_neighbors = set()
        self.backward_neighbors = set()
        Waypoint.LOOKUP[ID] = self
        self.direction = np.array([0, 0])

    def calculate_direction(self):
        direction = np.array([node.position - self.position for node in self.forward_neighbors])
        direction = np.mean(direction, axis=0)
        self.direction = direction

    def add_node_connection(self, waypoint):
        self.forward_neighbors.add(waypoint)
        waypoint.backward_neighbors.add(self)


def create_waypoint_queue(path):
    with open(path, 'r') as file:
        data = json.load(file)

    waypoint_ids = {}
    for id, info in data.items():
        info['id'] = int(id)
        waypoint_ids[int(id)] = Waypoint(**info)

    for waypoint in waypoint_ids.values():
        for ID in waypoint.neighbors:
            neighbor = waypoint_ids[ID]
            waypoint.add_node_connection(neighbor)

        waypoint.calculate_direction()

    return list(waypoint_ids.values())
