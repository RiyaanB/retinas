# Import required modules
import cv2
import numpy as np
import os
import glob
from utils import camera_streamer as cs

NUM_IMAGES = 10
streamer = cs.WebcamStreamer('rtsp://192.168.0.226:554', cs.oneplus_8t_K)

# Define the dimensions of checkerboard
CHECKERBOARD = (5, 7)


# stop the iteration when specified
# accuracy, epsilon, is reached or
# specified number of iterations are completed.
criteria = (cv2.TERM_CRITERIA_EPS +
            cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)


# Vector for 3D points
threedpoints = []

# Vector for 2D points
twodpoints = []


# 3D points real world coordinates
objectp3d = np.zeros((1, CHECKERBOARD[0]
                    * CHECKERBOARD[1],
                    3), np.float32)
objectp3d[0, :, :2] = np.mgrid[0:CHECKERBOARD[0],
                            0:CHECKERBOARD[1]].T.reshape(-1, 2)
prev_img_shape = None

images = []
img = None

filenames = glob.glob('calibration/*.png')
for filename in filenames:
    images.append(cv2.imread(filename))

print("Read", len(images), "images from files")

while len(images) < NUM_IMAGES:

    if streamer.ret:
        img = streamer.img
    
    if img is not None:
        cv2.imshow("image", img)

        k = cv2.waitKey(1)

        if k%256 == 27:
            # ESC pressed
            print("Escape hit, closing...")
            break
        elif k%256 == 32:
            # SPACE pressed
            images.append(img)
            cv2.imwrite(f'calibration/pic{len(images)}.png', img)
            print("Snapshot", len(images), "taken.")

print("Proceeding with", len(images), "calibration images...")

grayColors = []

for image in images:
    grayColor = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    grayColors.append(grayColor)

    # Find the chess board corners
    # If desired number of corners are
    # found in the image then ret = true
    ret, corners = cv2.findChessboardCorners(
                    grayColor, CHECKERBOARD,
                    cv2.CALIB_CB_ADAPTIVE_THRESH
                    + cv2.CALIB_CB_FAST_CHECK +
                    cv2.CALIB_CB_NORMALIZE_IMAGE)
    print(ret)
    # If desired number of corners can be detected then,
    # refine the pixel coordinates and display
    # them on the images of checker board
    if ret == True:
        threedpoints.append(objectp3d)

        # Refining pixel coordinates
        # for given 2d points.
        corners2 = cv2.cornerSubPix(
            grayColor, corners, (11, 11), (-1, -1), criteria)

        twodpoints.append(corners2)

        # Draw and display the corners
        image = cv2.drawChessboardCorners(image,
                                        CHECKERBOARD,
                                        corners2, ret)

    # cv2.imshow('img', image)
    # cv2.waitKey(0)

cv2.destroyAllWindows()

# h, w = image.shape[:2]


# Perform camera calibration by
# passing the value of above found out 3D points (threedpoints)
# and its corresponding pixel coordinates of the
# detected corners (twodpoints)
print(threedpoints)
print(twodpoints)
ret, matrix, distortion, r_vecs, t_vecs = cv2.calibrateCamera(
    threedpoints, twodpoints, grayColors[0].shape[::-1], None, None)


# Displaying required output
print(" Camera matrix:")
print(matrix)

print("\n Distortion coefficient:")
print(distortion)

print("\n Rotation Vectors:")
print(r_vecs)

print("\n Translation Vectors:")
print(t_vecs)
