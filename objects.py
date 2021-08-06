import numpy as np
from representation import *
from pose import *

MIN_POINTS = 4
MIN_REPRESENTATION = 1


def __is_hashable__(x):
    return not (('__hash__' not in dir(x)) or (x.__hash__ is None))


def __is_iterable__(x):
    return not (('__iter__' not in dir(x)) or (x.__iter__ is None))


class Body:

    def __init__(self, labels, points, representation=None, pose=None):
        assert __is_iterable__(labels)
        assert isinstance(points, np.matrix) or isinstance(points, np.ndarray)
        assert points.shape[0] == 3
        assert points.shape[1] >= MIN_POINTS
        assert len(labels) == points.shape[1]
        for label in labels:
            assert __is_hashable__(label)
        assert representation is None or isinstance(representation, Representation)

        self.labels = labels
        self.points = points
        self.representation = representation

        if pose is None:
            pose = Pose(0, 0, 0, 0, 0, 0)
        self.pose = pose


class Camera:

    def __init__(self, streamer, representation=None, pose=None):
        self.streamer = streamer
        self.representation = representation
        self.pose = pose

    def get_view(self):
        pass
