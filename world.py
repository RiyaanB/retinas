import cv2
import numpy as np
from pose import *
from representation import *


DEFAULT_CAMERA_POSITION = (0, -1, 1.)
DEFAULT_CAMERA_TARGET = (0., 0., 0.)
DEFAULT_WORLD_CAM_POSE = get_cam_pose(DEFAULT_CAMERA_POSITION, DEFAULT_CAMERA_TARGET)

bodies = []
cameras = []

