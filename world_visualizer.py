import apriltag
import cv2
import camera_tracking.camera_streamer as cs
import time
import numpy as np
import math

####################################################################

# LINK AND CORNER CONFIGURATIONS:


import tagconfigs.link_config.config_12_all_corners as LC
import tagconfigs.world_config.config_12_45degrees_all_corners as WC

# This Dict maps from Link TAG_NUMBER to Robot coordinates
LINK_TAG_DICT = LC.LINK_TAG_DICT
# This Dict maps from Corner TAG_ID to World Pose
CORNER_POSE_DICT = WC.CORNER_POSE_DICT
# This Dict maps from Corner TAG_ID to the Tag's name
CORNER_NAME_DICT = WC.CORNER_NAME_DICT


# These numbers hold side lengths of Link and Corner tags in meters
LINK_TAG_LENGTH = LC.LINK_TAG_LENGTH
CORNER_TAG_LENGTH = WC.CORNER_TAG_LENGTH

# GET_TAG_NUMBER is a function which maps from TAG_ID to TAG_NUMBER on that link
GET_TAG_NUMBER = LC.get_tag_number
# GET_LINK_ID is a function which maps from TAG_ID to LINK_ID
GET_LINK_ID = LC.get_link_id

####################################################################

CAMERA_LOCATION = np.array([[WC.WX/2, -1.3, 0.8]])
CAMERA_DESTINATION = np.array([[WC.WX/2, 0, 0.2]])

################################

# GRAM-SCHMIDT PROCESS

CAMERA_DIRECTION_VECTOR = CAMERA_DESTINATION - CAMERA_LOCATION
CAMERA_DIRECTION_UNIT_VECTOR = CAMERA_DIRECTION_VECTOR / np.linalg.norm(CAMERA_DIRECTION_VECTOR)
UP_VECTOR = np.array([[0,1,0]])
CAMERA_RIGHT = np.cross(UP_VECTOR, CAMERA_DIRECTION_UNIT_VECTOR)
CAMERA_RIGHT /= np.linalg.norm(CAMERA_RIGHT)
CAMERA_UP = np.cross(CAMERA_DIRECTION_UNIT_VECTOR, CAMERA_RIGHT)
ROT = np.zeros((4,4))
ROT[0, :3] = CAMERA_RIGHT
ROT[1, :3] = CAMERA_UP
ROT[2, :3] = CAMERA_DIRECTION_UNIT_VECTOR
ROT[:3, 3:] = CAMERA_LOCATION.T
ROT[3, :] = [0, 0, 0, 1]
ROT = ROT @ np.array([[ -1, 0,  0, 0],
                   [0, -1,  0, 0],
                   [0,  0,  1, 0],
                   [0, 0, 0, 1]])

################################

####################################################################

C1_COLORS = [(255, 255, 0), (0, 255, 255), (255, 255, 255)]
C2_COLORS = [(255, 0, 0), (0, 255, 255), (255, 255, 255)]
C3_COLORS = [(0, 255, 0), (0, 255, 255), (255, 255, 255)]
C4_COLORS = [(0, 0, 255), (0, 255, 255), (255, 255, 255)]


def get_RT_from_4x4(M):
    return M[:3, :3], M[:3, 3:]

def get_4x4_from_vectors(rvec, tvec):
    R = cv2.Rodrigues(rvec)[0]
    T = tvec
    M = np.zeros((4, 4))
    M[:3, :3] = R
    M[:3, 3:] = T
    M[3, :] = [0, 0, 0, 1]

    return M

def get_vectors_from_4x4(M):
    R, T = get_RT_from_4x4(M)
    return cv2.Rodrigues(R)[0], T


def draw_axis(frame, transformation, colors, K, D):
    points = np.float32([[0.03, 0, 0], [0, 0.03, 0], [0, 0, 0.03], [0, 0, 0]])

    R, T = get_RT_from_4x4(transformation)

    (x, y, z, o), _ = cv2.projectPoints(points, R, T, K, D)

    o = (int(o[0][0]), int(o[0][1]))
    x = (int(x[0][0]), int(x[0][1]))
    y = (int(y[0][0]), int(y[0][1]))
    z = (int(z[0][0]), int(z[0][1]))


    frame = cv2.line(frame, o, x, colors[0], 2)
    frame = cv2.line(frame, o, y, colors[1], 2)
    frame = cv2.line(frame, o, z, colors[2], 2)

    return frame


####################################################################
K = np.array([
            [954.0874994019651, 0                , 660.572082940535  ],
            [0                , 949.9159862376827, 329.78814306885795],
            [0                , 0                , 1                 ]
        ])

D = 0

cv2.namedWindow("WORLD")
# cv2.startWindowThread()

img_counter = 0

top_cTw = np.array([
        [-1, 0, 0, 0.5],
        [0, 1, 0, -0.25],
        [0,  0, 1, 2],
        [0, 0, 0, 1],
    ])

print(get_vectors_from_4x4(top_cTw))

default_cTw = ROT

print(get_vectors_from_4x4(default_cTw))

while True:
    frame = np.zeros(WC.CAMERA_DIMS)

    for corner in CORNER_POSE_DICT:

        corner_pose = CORNER_POSE_DICT[corner]
        rTc = np.linalg.inv(default_cTw) @ corner_pose
        if corner in (584, 585, 586):
            color = C1_COLORS
        elif corner in (581, 582, 583):
            color = C2_COLORS
        elif corner in (578, 579, 580):
            color = C3_COLORS
        else:
            color = C4_COLORS

        draw_axis(frame, rTc, color, K, D)

    k = cv2.waitKey(1)
    if k % 256 == 27:
        # ESC pressed
        print("Escape hit, closing...")
        break
    elif k % 256 == 32:
        # SPACE pressed
        pass

    # frame = cv2.flip(frame, 0)
    cv2.imshow("WORLD", frame)

for a in range(5):
    cv2.destroyAllWindows()
exit()
