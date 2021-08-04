import cv2
import numpy as np
import math


def delta(v1, v2):
    return np.sum(v1 - v2)


def homography_method(H, K, approx_T=None):
    num, Rs, Ts, Ns = cv2.decomposeHomographyMat(H, K)

    count = 0
    valid = []

    for a in range(num):
        if Ts[a][2][0] < 0 or np.dot(np.array([0,0,1]), Ns[a]) > 0:
            # print(a)
            # print(cv2.Rodrigues(Rs[a])[0])
            # print(Ts[a])
            valid.append(a)
            count += 1

    # print(count)

    if len(valid) < 1:
        return False, None, None


    min_solution = valid[0]
    for a in valid:
        if delta(Ts[a], approx_T) < delta(min_solution, approx_T):
            min_solution = a

    R_vector = cv2.Rodrigues(Rs[-1])[0]
    T_vector = Ts[-1]
    return True, R_vector, T_vector