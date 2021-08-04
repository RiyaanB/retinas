import apriltag
import cv2
import camera_tracking.camera_streamer as cs
import time
import camera_tracking.pose as pose
import numpy as np
from numpy.linalg import inv
import math

np.set_printoptions(suppress=True)
detector = apriltag.Detector()

mac_camera_matrix = [
        [954.0874994019651, 0                , 660.572082940535  ],
        [0                , 949.9159862376827, 329.78814306885795],
        [0                , 0                , 1                 ]
]

mac_camera_params = [954.0874994019651, 949.9159862376827, 660.572082940535, 329.78814306885795]

phone_camera_matrix = [
    [1.13e3, 0  , 1.07e3],
    [0  , 1.14e3, 733],
    [0  , 0  , 1]
]

phone_camera_params = [248, 249, 174, 151]

tag_size = .32
object_points = np.array(
    [[-tag_size/2, tag_size/2, 0],
     [tag_size/2, tag_size/2, 0],
     [tag_size/2, -tag_size/2, 0],
     [-tag_size/2, -tag_size/2, 0]]
)

detector = apriltag.Detector()

streamer = cs.WebcamStreamer(1, cs.mac_K)
# streamer = cs.RemoteStreamer(cs.URL)
time.sleep(2)

cv2.namedWindow("test")
# cv2.startWindowThread()

img_counter = 0

K = np.array(mac_camera_matrix)
# K = np.array(phone_camera_matrix)

D = np.array([0.036998502838994515, -0.13207581435883872, -0.000045055253893522236, -0.007745497656725853, 0.11519181871308336])
# D = np.array([0.334, -7.09, 0.0165, 0.0134])
# D = np.array([0, 0, 0, 0])

c_T_w = None
w_T_c = None

def rotation_matrix(axis, theta):
    """
    Return the rotation matrix associated with counterclockwise rotation about
    the given axis by theta radians.
    """
    axis = np.asarray(axis)
    axis = axis / math.sqrt(np.dot(axis, axis))
    a = math.cos(theta / 2.0)
    b, c, d = -axis * math.sin(theta / 2.0)
    aa, bb, cc, dd = a * a, b * b, c * c, d * d
    bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
    return np.array([[aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac)],
                     [2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab)],
                     [2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc]])


while True:
    ret, frame = streamer.read()
    if not ret:
        print("failed to grab frame")
        break

    k = cv2.waitKey(1)
    if k % 256 == 27:
        # ESC pressed
        print("Escape hit, closing...")
        break
    elif k % 256 == 32:
        # SPACE pressed
        pass

    # cv2.imwrite("live_image.png", frame)
    # img = cv2.imread('live_image.png', cv2.IMREAD_GRAYSCALE)

    grayscale_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    results = detector.detect(grayscale_frame)
    # print(results)

    # print(grayscale_frame.shape)

    for r in results:

        # extract the bounding box (x, y)-coordinates for the AprilTag
        # and convert each of the (x, y)-coordinate pairs to integers
        (ptA, ptB, ptC, ptD) = r.corners
        ptB = (int(ptB[0]), int(ptB[1]))
        ptC = (int(ptC[0]), int(ptC[1]))
        ptD = (int(ptD[0]), int(ptD[1]))
        ptA = (int(ptA[0]), int(ptA[1]))
        # draw the bounding box of the AprilTag detection
        cv2.line(frame, ptA, ptB, (0, 255, 0), 2)
        cv2.line(frame, ptB, ptC, (0, 255, 0), 2)
        cv2.line(frame, ptC, ptD, (0, 255, 0), 2)
        cv2.line(frame, ptD, ptA, (0, 255, 0), 2)
        # draw the center (x, y)-coordinates of the AprilTag
        (cX, cY) = (int(r.center[0]), int(r.center[1]))
        cv2.circle(frame, (cX, cY), 5, (0, 0, 255), -1)
        # draw the tag family on the image
        tagFamily = r.tag_family.decode("utf-8")
        tagID = r.tag_id
        # tagName = link_dict[tagID]
        # cv2.putText(frame, tagName, (ptA[0], ptA[1] - 15),
        #             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        cv2.putText(frame, str(r.tag_id), (ptA[0], ptA[1] - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        flag, rvec, tvec = cv2.solvePnP(object_points, r.corners, K, np.zeros([0,0,0,0]), flags=cv2.SOLVEPNP_IPPE_SQUARE)

        print("################################################")
        # print(rvec)

        r_x = rotation_matrix([1, 0, 0], rvec[0][0])
        r_y = rotation_matrix([0, 1, 0], rvec[1][0])
        r_z = rotation_matrix([0, 0, 1], rvec[2][0])

        R = r_z @ r_y
        R = R @ r_x

        if r.tag_id == 2:

            c_T_w = np.concatenate((np.concatenate((R, tvec), axis=1), np.array([[0, 0, 0, 1]])), axis=0)
            w_T_c = inv(c_T_w)
        elif w_T_c is None:
            pass
        else:
            c_T_r = np.concatenate((np.concatenate((R, tvec), axis=1), np.array([[0, 0, 0, 1]])), axis=0)
            r = w_T_c @ c_T_r

            # for row in range(0,3):
            #     for col in range(0,3):
            #         r[row][col] *= 180/3.1415926536

            print(r)



        #print(tvec)
        # print(tvec[2][0])

    time.sleep(1)
    # print("\n\n\n################################################")
    cv2.imshow("test", frame)

cv2.destroyAllWindows()
for i in range(1, 5):
    cv2.waitKey(1)

streamer.close()
exit()
