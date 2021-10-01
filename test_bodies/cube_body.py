import numpy as np

from objects import RetinaBody
from pose import Pose

TAG_SIDE_LENGTH = 0.059
CUBE_SIDE_LENGTH = 0.075
SQUARE_BOUNDARY_LENGTH = (CUBE_SIDE_LENGTH - TAG_SIDE_LENGTH) / 2
TAG_HALF_LENGTH = TAG_SIDE_LENGTH / 2
CUBE_HALF_LENGTH = CUBE_SIDE_LENGTH / 2

TAG_POINTS = np.array([
    [-TAG_HALF_LENGTH, -TAG_HALF_LENGTH, 0, 1],
    [-TAG_HALF_LENGTH,  TAG_HALF_LENGTH, 0, 1],
    [ TAG_HALF_LENGTH,  TAG_HALF_LENGTH, 0, 1],
    [ TAG_HALF_LENGTH, -TAG_HALF_LENGTH, 0 ,1],
])



PI = 3.1415926546

tag0_pose = Pose(PI / 2, 0, 0, 0, -CUBE_HALF_LENGTH, 0)
tag1_pose = Pose(0, 0, 0, 0, 0, CUBE_HALF_LENGTH)
tag2_pose = Pose(-PI / 2, 0, 0, 0, CUBE_HALF_LENGTH, 0)
tag3_pose = Pose(PI, 0, 0, 0, 0, -CUBE_HALF_LENGTH)
tag4_pose = Pose(0,0,0,-CUBE_HALF_LENGTH,0,0) @ Pose(0, 0, -PI / 2, 0, 0, 0) @ Pose(PI/2, 0, 0, 0, 0, 0)
tag5_pose = Pose(0,0,0, CUBE_HALF_LENGTH,0,0) @ Pose(0, 0,  PI / 2, 0,  0, 0) @ Pose(PI/2, 0, 0, 0, 0, 0)


def get_cube_point_dict(x):
    pd = dict()

    pd[x+0, 0], pd[x+0, 1], pd[x+0, 2], pd[x+0, 3] = (tag0_pose.matrix @ TAG_POINTS.T).T[:4, :3]
    pd[x+1, 0], pd[x+1, 1], pd[x+1, 2], pd[x+1, 3] = (tag1_pose.matrix @ TAG_POINTS.T).T[:4, :3]
    pd[x+2, 0], pd[x+2, 1], pd[x+2, 2], pd[x+2, 3] = (tag2_pose.matrix @ TAG_POINTS.T).T[:4, :3]
    pd[x+3, 0], pd[x+3, 1], pd[x+3, 2], pd[x+3, 3] = (tag3_pose.matrix @ TAG_POINTS.T).T[:4, :3]
    pd[x+4, 0], pd[x+4, 1], pd[x+4, 2], pd[x+4, 3] = (tag4_pose.matrix @ TAG_POINTS.T).T[:4, :3]
    pd[x+5, 0], pd[x+5, 1], pd[x+5, 2], pd[x+5, 3] = (tag5_pose.matrix @ TAG_POINTS.T).T[:4, :3]

    return pd


CUBE_0_POINT_DICT = get_cube_point_dict(0)
CUBE_1_POINT_DICT = get_cube_point_dict(6)

cube0_body = RetinaBody("Cube0", CUBE_0_POINT_DICT)
cube1_body = RetinaBody("Cube1", CUBE_1_POINT_DICT)
