import apriltag
import cv2
import camera_tracking.camera_streamer as cs
import time
import camera_tracking.pose as pose
import numpy as np
from numpy.linalg import inv
import math


detector = apriltag.Detector()

mac_camera_matrix = [
        [954.0874994019651, 0                , 660.572082940535  ],
        [0                , 949.9159862376827, 329.78814306885795],
        [0                , 0                , 1                 ]
]

mac_camera_params = [954.0874994019651, 949.9159862376827, 660.572082940535, 329.78814306885795]

phone_camera_matrix = [
    [248, 0  , 174],
    [0  , 249, 151],
    [0  , 0  , 1]
]

phone_camera_params = [248, 249, 174, 151]

object_points = np.array([[-0.079999998, 0.079999998, 0],
 [0.079999998, 0.079999998, 0],
 [0.079999998, -0.079999998, 0],
 [-0.079999998, -0.079999998, 0]])

detector = apriltag.Detector()

streamer = cs.WebcamStreamer()
# streamer = cs.RemoteStreamer(cs.URL)
time.sleep(2)

cv2.namedWindow("test")
# cv2.startWindowThread()

img_counter = 0

K = np.array(mac_camera_matrix)

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

    for r in results:
        # extract the bounding box (x, y)-coordinates for the AprilTag
        # and convert each of the (x, y)-coordinate pairs to integers
        (ptA, ptB, ptC, ptD) = r.corners
        ptB = (int(ptB[0]), int(ptB[1]))
        ptC = (int(ptC[0]), int(ptC[1]))
        ptD = (int(ptD[0]), int(ptD[1]))
        ptA = (int(ptA[0]), int(ptA[1]))
        # draw the bounding box of the AprilTag detection
        cv2.line(frame, ptA, ptB, (0, 0, 255), 2)
        cv2.line(frame, ptB, ptC, (0, 255, 0), 2)
        cv2.line(frame, ptC, ptD, (0, 255, 0), 2)
        cv2.line(frame, ptD, ptA, (255, 0, 0), 2)
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

        H = r.homography

        # print(H)

        num, Rs, Ts, Ns = cv2.decomposeHomographyMat(H, K)

        # max_num = 0
        # for a in range(1, num):
        #     if Ts[a][2][0] > Ts[max_num][2][0]:
        #         max_num = a
        # # print(Rs[a])

        R_vector = cv2.Rodrigues(Rs[-1])[0]
        print(R_vector)
        # print(Ts[a])
        # print(Ns[a])

        # print("Vector((", end="")
        # for t in Ts[a]:
        #     print(t[0], end=", ")
        # print("), (", end="")
        # for n in Ns[a]:
        #     print(n[0]+t[0], end=", ")
        # print("))")

        print("################################################")
        # print("[INFO] tag family: {}".format(tagID))
    print("\n\n\n################################################")
    cv2.imshow("test", frame)

cv2.destroyAllWindows()
for i in range(1, 5):
    cv2.waitKey(1)

streamer.close()
exit()
