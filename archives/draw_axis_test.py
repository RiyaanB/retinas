import numpy as np
import cv2


def draw_axis(img, R, t, K):
    # unit is mm
    # rotV, _ = cv2.Rodrigues(R)

    # in meters
    points = np.float32([[0.03,0,0],[0,0.03,0],[0,0,0.03],[0,0,0]])

    (x,y,z,o), _ = cv2.projectPoints(points, R, t, K, (0, 0, 0, 0))

    o = (int(o[0][0]), int(o[0][1]))
    x = (int(x[0][0]), int(x[0][1]))
    y = (int(y[0][0]), int(y[0][1]))
    z = (int(z[0][0]), int(z[0][1]))

    img = cv2.line(img, o, x, (0,0,0), 3)
    img = cv2.line(img, o, y, (0,255,0), 3)
    img = cv2.line(img, o, z, (255,255,255), 3)

    # (cX, cY) = (int(r.center[0]), int(r.center[1]))
    # cv2.circle(frame, (cX, cY), 5, (0, 0, 255), -1)
    #
    # np.trunc(R)
    #
    # R *= 180/math.pi
    #
    # print(R)

    # cv2.putText(img, "{:.3}".format(R[0][0]), (x[0], x[1] - 15),
    #             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    # cv2.putText(img, "{:.3}".format(R[1][0]), (y[0], y[1] - 15),
    #             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    # cv2.putText(img, "{:.3}".format(R[2][0]), (z[0], z[1] - 15),
    #             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    return img