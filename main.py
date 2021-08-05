import numpy as np
import cv2



def get_camera_observation(i, C):
    """
    :param i: which Camera Streamer Object to take observation from
    :param C:
    :return:
    """

def get_poses(C, B, tag_to_body_and_num):
    """
    :param C: List of Camera Streamer Objects from which observations will be retrieved, of length J
    :param B: List of Rigid Bodies of length I (including world)
    :param tag_to_body_and_num: a function mapping tag_id and tag_corner to body_id, point_id
    :return:
    """

    I = len(B)

    assert type(tag_to_body_and_num) == type(get_poses)

    f