import numpy as np
import cv2
import matplotlib.pyplot as plt
from MapGenerator import Waypoint, create_waypoint_queue
from shapely.geometry import Polygon, Point


class _Polygon:
    def __init__(self, shape):
        self.polygon = Polygon(shape)
        self.waypoints = set()

    def register_point(self, waypoint):
        point = Point(waypoint.position)
        if self.polygon.contains(point):
            self.waypoints.add(waypoint)


class PolygonHelper:
    def __init__(self, polygon_file: str, waypoint_file: str):
        pass
