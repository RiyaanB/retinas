import Retinas.pose as pose
import numpy as np
import math

from Retinas.objects import RetinaBody

Pose = pose.Pose

CORNER_TAG_LENGTH   = 0.055    # meters
CORNER_FULL_LENGTH  = 0.069    # meters
WORLD_X_LENGTH      = 0.540    # meters
WORLD_Y_LENGTH      = 0.844    # meters
TAG_Z_HEIGHT        = 0.007    # meters


Z_HEIGHT = TAG_Z_HEIGHT
angle45 = 3.1415926536/4

WX = WORLD_X_LENGTH
WY = WORLD_Y_LENGTH
CH = CORNER_FULL_LENGTH/2
TAG_HALF_LENGTH = CORNER_TAG_LENGTH/2
XY_OFFSET = (CORNER_FULL_LENGTH/math.sqrt(2)) / 2
XY_HEIGHT = Z_HEIGHT + XY_OFFSET

TAG_POINTS = np.array([
    [-TAG_HALF_LENGTH,  TAG_HALF_LENGTH, 0, 1],
    [ TAG_HALF_LENGTH,  TAG_HALF_LENGTH, 0, 1],
    [ TAG_HALF_LENGTH, -TAG_HALF_LENGTH, 0, 1],
    [-TAG_HALF_LENGTH, -TAG_HALF_LENGTH, 0 ,1],
])

C1_Z = Pose((0,0,0), np.array([[0], [0], [Z_HEIGHT]]))
C1_X = Pose((0,angle45,0), np.array([[-CH-XY_OFFSET], [0], [XY_HEIGHT]]))
C1_Y = Pose((-angle45,0,0), np.array([[0], [-CH-XY_OFFSET], [XY_HEIGHT]]))

C2_Z = Pose((0,0,0), np.array([[0], [WY], [Z_HEIGHT]]))
C2_X = Pose((0,angle45,0), np.array([[-CH-XY_OFFSET], [WY], [XY_HEIGHT]]))
C2_Y = Pose((angle45,0,0), np.array([[0], [WY+CH+XY_OFFSET], [XY_HEIGHT]]))

C3_Z = Pose((0,0,0), np.array([[WX], [WY], [Z_HEIGHT]]))
C3_X = Pose((0,-angle45,0), np.array([[WX+CH+XY_OFFSET], [WY], [XY_HEIGHT]]))
C3_Y = Pose((angle45,0,0), np.array([[WX], [WY+CH+XY_OFFSET], [XY_HEIGHT]]))

C4_Z = Pose((0,0,0), np.array([[WX], [0], [Z_HEIGHT]]))
C4_X = Pose((0,-angle45,0), np.array([[WX+CH+XY_OFFSET], [0], [XY_HEIGHT]]))
C4_Y = Pose((-angle45,0,0), np.array([[WX], [-CH-XY_OFFSET], [XY_HEIGHT]]))


x = 575

pd = dict()

pd[x + 0, 0], pd[x + 0, 1], pd[x + 0, 2], pd[x + 0, 3] = (C4_Z.matrix @ TAG_POINTS.T).T[:4, :3]
pd[x + 1, 0], pd[x + 1, 1], pd[x + 1, 2], pd[x + 1, 3] = (C4_X.matrix @ TAG_POINTS.T).T[:4, :3]
pd[x + 2, 0], pd[x + 2, 1], pd[x + 2, 2], pd[x + 2, 3] = (C4_Y.matrix @ TAG_POINTS.T).T[:4, :3]
pd[x + 3, 0], pd[x + 3, 1], pd[x + 3, 2], pd[x + 3, 3] = (C3_Z.matrix @ TAG_POINTS.T).T[:4, :3]
pd[x + 4, 0], pd[x + 4, 1], pd[x + 4, 2], pd[x + 4, 3] = (C3_Y.matrix @ TAG_POINTS.T).T[:4, :3]
pd[x + 5, 0], pd[x + 5, 1], pd[x + 5, 2], pd[x + 5, 3] = (C3_X.matrix @ TAG_POINTS.T).T[:4, :3]
pd[x + 6, 0], pd[x + 6, 1], pd[x + 6, 2], pd[x + 6, 3] = (C2_Z.matrix @ TAG_POINTS.T).T[:4, :3]
pd[x + 7, 0], pd[x + 7, 1], pd[x + 7, 2], pd[x + 7, 3] = (C2_X.matrix @ TAG_POINTS.T).T[:4, :3]
pd[x + 8, 0], pd[x + 8, 1], pd[x + 8, 2], pd[x + 8, 3] = (C2_Y.matrix @ TAG_POINTS.T).T[:4, :3]
pd[x + 9, 0], pd[x + 9, 1], pd[x + 9, 2], pd[x + 9, 3] = (C1_Y.matrix @ TAG_POINTS.T).T[:4, :3]
pd[x +10, 0], pd[x +10, 1], pd[x +10, 2], pd[x +10, 3] = (C1_X.matrix @ TAG_POINTS.T).T[:4, :3]
pd[x +11, 0], pd[x +11, 1], pd[x +11, 2], pd[x +11, 3] = (C1_Z.matrix @ TAG_POINTS.T).T[:4, :3]

world_body = RetinaBody("World", pd)
