import cv2
import numpy as np

BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
YELLOW = (255, 255, 0)
MAGENTA = (255, 0, 255)
CYAN = (0, 255, 255)

DEFAULT_THICKNESS = 1


class Representation:

    def __init__(self, name):
        self.name = name
        self.points = set()
        self.lines = dict()
        self.dots = dict()

    def add_point(self, point):
        if point not in self.points:
            self.points.add(point)

    def add_line(self, start, end, color=RED, thickness=DEFAULT_THICKNESS):
        line_key = (start[0], start[1], start[2], end[0], end[1], end[2])
        self.add_point(start)
        self.add_point(end)
        self.lines[line_key] = color, thickness

    def add_dot(self, point, color=RED, thickness=DEFAULT_THICKNESS):
        self.add_point(point)
        self.dots[point] = color, thickness

    def add_representation(self, representation):
        self.points.update(representation.points)
        for line_key, details in representation.lines.items():
            self.lines[line_key] = details
        for point, details in representation.dots.items():
            self.lines[point] = details

    def draw(self, frame, K, D, rvec, tvec):
        object_points_list = list(self.points)

        object_points_map = {}
        for i in range(len(object_points_list)):
            object_points_map[object_points_list[i]] = i

        object_points = np.float32(object_points_list)
        frame_points = cv2.projectPoints(object_points, rvec, tvec, K, D)

        for line in self.lines:
            object_start = line[0], line[1], line[2]
            object_end = line[3], line[4], line[5]

            frame_start = frame_points[object_points_map[object_start]].astype(int).ravel()
            frame_end = frame_points[object_points_map[object_end]].astype(int).ravel()

            color, thickness = self.lines[line]

            frame = cv2.line(frame, frame_start, frame_end, color, thickness)

        for dot in self.dots:
            object_point = dot
            frame_point = frame_points[object_points_map[object_point]].astype(int).ravel()
            color, thickness = self.dots[dot]
            frame = cv2.circle(frame, frame_point, radius=0, color=color, thickness=thickness)


class AxisArrow(Representation):

    def __init__(self, name="Arrow", color=RED, thickness=DEFAULT_THICKNESS, axis=2, axis_start=-0.05, axis_end=0.05,
                 size=0.2):
        super(AxisArrow, self).__init__(name)
        size = (axis_end - axis_start) * size  # size is percentage
        if axis == 0:
            self.add_line((axis_start, 0, 0), (axis_end, 0, 0), color, thickness)
            self.add_line((axis_end - size, -size, 0), (axis_end, 0, 0), color, thickness)
            self.add_line((size, 0, axis_end - size), (axis_end, 0, 0), color, thickness)
        if axis == 1:
            self.add_line((0, axis_start, 0), (0, axis_end, 0), color, thickness)
            self.add_line((0, axis_end - size, -size), (0, axis_end, 0), color, thickness)
            self.add_line((0, axis_end - size, size), (0, axis_end, 0), color, thickness)
        if axis == 2:
            self.add_line((0, 0, axis_start), (0, 0, axis_end), color, thickness)
            self.add_line((0, -size, axis_end - size), (0, 0, axis_end), color, thickness)
            self.add_line((0, size, axis_end - size), (0, 0, axis_end), color, thickness)


class Axes(Representation):

    def __init__(self, name="Axes", length=0.1, color=RED, thickness=DEFAULT_THICKNESS):
        super(Axes, self).__init__(name)
        if isinstance(color[0], int):
            color = [color, color, color]
        self.add_representation(AxisArrow("X", color[0], thickness, 0, -length / 2, length / 2, 0.2))
        self.add_representation(AxisArrow("Y", color[1], thickness, 1, -length / 2, length / 2, 0.2))
        self.add_representation(AxisArrow("Z", color[2], thickness, 2, -length / 2, length / 2, 0.2))


class AxisRectangle(Representation):

    def __init__(self, name="Rectangle", color=RED, thickness=DEFAULT_THICKNESS, axis=2, top_left=(-0.1, 0.1, 0),
                 height=0.05, width=0.02):
        super(AxisRectangle, self).__init__(name)
        assert axis in (0, 1, 2)
        if axis == 0:
            top_right = top_left[0], top_left[1] - width, top_left[2]
            bot_left = top_left[0], top_left[1], top_left[2] - height
            bot_right = top_left[0], top_left[1] - width, top_left[2] - height
        elif axis == 1:
            top_right = top_left[0] + width, top_left[1], top_left[2]
            bot_left = top_left[0], top_left[1], top_left[2] - height
            bot_right = top_left[0] + width, top_left[1], top_left[2] - height
        else:
            top_right = top_left[0] + width, top_left[1], top_left[2]
            bot_left = top_left[0], top_left[1] - height, top_left[2]
            bot_right = top_left[0] + width, top_left[1] - height, top_left[2]

        if isinstance(color[0], int):
            color = [color, color, color, color]

        self.add_line(top_left, top_right, color[0], thickness)
        self.add_line(top_right, bot_right, color[1], thickness)
        self.add_line(bot_right, bot_left, color[2], thickness)
        self.add_line(bot_left, top_left, color[3], thickness)


class AxisSquare(AxisRectangle):

    def __init__(self, name="Square", color=RED, thickness=DEFAULT_THICKNESS, axis=2, top_left=(-0.1, 0.1, 0),
                 side=0.05):
        super(AxisSquare, self).__init__(name, color, thickness, axis, top_left, side, side)
