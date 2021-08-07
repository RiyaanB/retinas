import numpy as np
from representation import *
from pose import *
import apriltag

MIN_POINTS = 4
MIN_REPRESENTATION = 1


def __is_hashable__(x):
    return not (('__hash__' not in dir(x)) or (x.__hash__ is None))


def __is_iterable__(x):
    return not (('__iter__' not in dir(x)) or (x.__iter__ is None))


class RetinaBody:

    def __init__(self, name="Body", point_dict=None, pose=((0, 0, 0), (0, 0, 0)), representation=None):
        self.name = name
        if point_dict is None:
            point_dict = {}
        self.point_dict = point_dict
        if representation is None:
            self.representation = Representation(name)
            self.representation.add_point_dict(point_dict)
        else:
            self.representation = representation

        self.pose = Pose(pose)


class RetinaCamera:

    def __init__(self, streamer, get_view=None, pose=((0, 0, 0), (0, 0, 0)), representation=None):
        self.streamer = streamer
        if representation is None:
            representation = AxisCamera(streamer.name)
        self.representation = representation
        self.pose = Pose(pose)
        self.get_view = get_view
