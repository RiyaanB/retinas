import apriltag
import cv2
import camera_tracking.camera_streamer as cs
import time
import camera_tracking.pose as pose
import numpy as np
from numpy.linalg import inv
import math
np.set_printoptions(suppress=True)
np.set_printoptions(precision=3)


K = np.array([
        [
            954.0874994019651,
            0,
            660.572082940535
        ],
        [
            0,
            949.9159862376827,
            329.78814306885795
        ],
        [
            0,
            0,
            1
        ]
    ])

D = np.array([0.036998502838994515, -0.13207581435883872, -0.000045055253893522236, -0.007745497656725853, 0.11519181871308336])

detector = apriltag.Detector()

streamer = cs.WebcamStreamer(1, cs.mac_K)
# streamer = cs.RemoteStreamer(cs.URL)

cv2.namedWindow("test")

w_T_c = None


def draw_tvec(img, T, o):

    t_cm = T * 100

    cv2.putText(img, "{:.3}, {:.3}, {:.3}".format(t_cm[0][0], t_cm[1][0], t_cm[2][0]), (o[0], o[1] - 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (128, 128, 0), 2)


def draw_axis(img, R, t, K):
    # unit is mm
    # rotV, _ = cv2.Rodrigues(R)

    # in meters
    points = np.float32([[0.03,0,0],[0,0.03,0],[0,0,0.03],[0,0,0]])

    r_matrix = cv2.Rodrigues(R)[0]

    z_vec = np.float64([[0], [0], [1]])

    # print(r_matrix)
    # print(z_vec)

    normal = (r_matrix @ z_vec).flatten()

    # print(normal)

    flip = 0

    if np.dot(normal, np.array([[0],[0],[1]])) < 0:
        # R[0][0] += math.pi
        # R[1][0] += math.pi
        flip = 1

    (x,y,z,o), _ = cv2.projectPoints(points, R, t, K, (0, 0, 0, 0))

    o = (int(o[0][0]), int(o[0][1]))
    x = (int(x[0][0]), int(x[0][1]))
    y = (int(y[0][0]), int(y[0][1]))
    z = (int(z[0][0]), int(z[0][1]))

    img = cv2.line(img, o, x, (0,0,0), 3)
    img = cv2.line(img, o, y, (0,255,0), 3)
    img = cv2.line(img, o, z, (255,255,255*flip), 3)

    # (cX, cY) = (int(r.center[0]), int(r.center[1]))
    # cv2.circle(frame, (cX, cY), 5, (0, 0, 255), -1)

    np.trunc(R)

    R *= 180/math.pi

    print(R)

    cv2.putText(frame, "{:.3}".format(R[0][0]), (x[0], x[1] - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    cv2.putText(frame, "{:.3}".format(R[1][0]), (y[0], y[1] - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    cv2.putText(frame, "{:.3}".format(R[2][0]), (z[0], z[1] - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # draw_tvec(frame, t, o)

    return img


clock = time.time()

while True:

    ret, frame = streamer.read()
    # print(frame.shape)
    if not ret:
        print("failed to grab frame")
        break

    k = cv2.waitKey(1)
    if k % 256 == 27:
        # ESC pressed
        print("Escape hit, closing...")
        break

    grayscale_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    results = detector.detect(grayscale_frame)

    for r in results:
        # extract the bounding box (x, y)-coordinates for the AprilTag
        # and convert each of the (x, y)-coordinate pairs to integers
        (ptA, ptB, ptC, ptD) = r.corners
        ptB = (int(ptB[0]), int(ptB[1]))
        ptC = (int(ptC[0]), int(ptC[1]))
        ptD = (int(ptD[0]), int(ptD[1]))
        ptA = (int(ptA[0]), int(ptA[1]))
        # draw the bounding box of the AprilTag detection
        cv2.line(frame, ptA, ptB, (0, 255, 0), 2) # 0 0 255
        cv2.line(frame, ptB, ptC, (0, 255, 0), 2)
        cv2.line(frame, ptC, ptD, (0, 255, 0), 2)
        cv2.line(frame, ptD, ptA, (0, 255, 0), 2) # 255 0 0
        # draw the center (x, y)-coordinates of the AprilTag
        (cX, cY) = (int(r.center[0]), int(r.center[1]))
        cv2.circle(frame, (cX, cY), 5, (0, 0, 255), -1)
        # draw the tag family on the image
        tagFamily = r.tag_family.decode("utf-8")
        tagID = r.tag_id

        cv2.putText(frame, str(r.tag_id), (ptA[0], ptA[1] - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        ret, rvec, tvec = pose.square_pnp_method(0.032, r.corners, K, 0)
        ret_wrong, rvec2, tvec_wrong = pose.homography_method(r.homography, K, approx_T=tvec)

        # r_matrix = cv2.Rodrigues(rvec)[0]
        #
        # print(r_matrix.shape)

        # print(rvec.shape)

        draw_tvec(frame, tvec, (cX, cY))
        draw_axis(frame, rvec2, tvec, cs.mac_K)


        print("########################")

        # print(rvec)

        print(tvec)

        # print(r.homography)

    cv2.imshow("test", frame)
    print(1/(time.time() - clock))
    clock = time.time()

cv2.destroyAllWindows()
for i in range(1, 5):
    cv2.waitKey(1)

streamer.close()
exit()
