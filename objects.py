import numpy as np

MIN_POINTS = 4
MIN_REPRESENTATION = 1


def __is_hashable__(x):
    return not (('__hash__' not in dir(x)) or (x.__hash__ is None))


def __is_iterable__(x):
    return not (('__iter__' not in dir(x)) or (x.__iter__ is None))


class Representation:

    def __init__(self, segments, colors):
        for segment in segments:
            assert len(segment) == 2
            assert len(segment[0]) == 3
            assert len(segment[1]) == 3
        for color in colors:
            assert len(color) == 3

        self.segments = segments
        self.colors = colors


class Body:

    def __init__(self, labels, points, representation=None, rvec=None, tvec=None):
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

        if rvec is None:
            self.rvec = [0, 0, 0]
        if tvec is None:
            self.tvec = [0, 0, 0]

        assert len(rvec) == 3
        for r in rvec:
            assert float(r)
        assert len(tvec) == 3
        for t in tvec:
            assert float(t)
        self.rvec = rvec
        self.tvec = tvec


class Camera:

    def __init__(self, streamer):
        self.streamer = streamer

    def get_view(self):
        pass
