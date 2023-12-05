import cv2
import numpy as np
import glob
import apriltag

detector = apriltag.Detector(apriltag.DetectorOptions(
            families='tag36h11',
            border=1,
            nthreads=8,
            quad_decimate=1.0,
            quad_blur=0.0,
            refine_edges=True,
            refine_decode=False,
            refine_pose=False,
            debug=False,
            quad_contours=True
            ))

images = []

filenames = glob.glob('lighting_calibration/*.HEIC')
for filename in filenames:
    frame = cv2.imread(filename)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    images.append(hsv)

for image in images:
	grayscale_frame = cv2.inRange(image, (0,0,0), (180, 255, 255))
	print(grayscale_frame.dtype)
	results = detector.detect(grayscale_frame)
	print(len(results))
	cv2.imshow('img', grayscale_frame)
	cv2.waitKey(0)
