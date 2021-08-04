import apriltag
import cv2
import camera_tracking.camera_streamer as cs
import time
import numpy as np
from numpy.linalg import inv
import math

detector = apriltag.Detector()

streamer = cs.WebcamStreamer(0, cs.mac_K)
# streamer = cs.RemoteStreamer(cs.URL)
time.sleep(2)

cv2.namedWindow("test")
# cv2.startWindowThread()

img_counter = 0

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
    print(results)

    for r in results:
        # extract the bounding box (x, y)-coordinates for the AprilTag
        # and convert each of the (x, y)-coordinate pairs to integers
        (ptA, ptB, ptC, ptD) = r.corners
        ptB = (int(ptB[0]), int(ptB[1]))
        ptC = (int(ptC[0]), int(ptC[1]))
        ptD = (int(ptD[0]), int(ptD[1]))
        ptA = (int(ptA[0]), int(ptA[1]))
        # draw the bounding box of the AprilTag detection
        cv2.line(frame, ptA, ptB, (255, 0, 0), 2)
        cv2.line(frame, ptB, ptC, (0, 255, 0), 2)
        cv2.line(frame, ptC, ptD, (255, 255, 255), 2)
        cv2.line(frame, ptD, ptA, (255, 255, 255), 2)
        # draw the center (x, y)-coordinates of the AprilTag
        (cX, cY) = (int(r.center[0]), int(r.center[1]))
        cv2.circle(frame, (cX, cY), 5, (0, 0, 255), -1)
        # draw the tag family on the image
        tagFamily = r.tag_family.decode("utf-8")
        tagID = str(r.tag_id)
        cv2.putText(frame, tagID+"A", (ptA[0], ptA[1] - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2.putText(frame, "B", (ptB[0], ptB[1] - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2.putText(frame, "C", (ptC[0], ptC[1] - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        print("[INFO] tag family: {}".format(tagID))

    print(frame.shape)
    cv2.imshow("test", frame)

cv2.destroyAllWindows()
for i in range(1, 5):
    cv2.waitKey(1)

streamer.close()
exit()