import cv2
import numpy as np
import os
import glob
from utils import camera_streamer as cs
import apriltag

NUM_IMAGES = -1

detector = apriltag.Detector()

images = []
filenames = glob.glob('lighting_calibration/*')

for filename in filenames:
    images.append(cv2.imread(filename))

print("Read", len(images), "images from files")

total = 0

def scale(img, scale_percent_or_dim):
    if type(scale_percent_or_dim) == tuple:
        dim = scale_percent_or_dim
    else:
        width = int(img.shape[1] * scale_percent_or_dim / 100)
        height = int(img.shape[0] * scale_percent_or_dim / 100)
        dim = (width, height)
    resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
    return resized

def pipeline(rgb):
    smallrgb = scale(rgb, 50)
    smallrgbblurred = cv2.GaussianBlur(smallrgb, (17, 17), 0)
    smallhsvblurred = cv2.cvtColor(smallrgbblurred, cv2.COLOR_BGR2HSV)
    smallgray = cv2.cvtColor(smallrgb, cv2.COLOR_BGR2GRAY)
    # smallgrayblurred = cv2.GaussianBlur(smallgray,(5,5),0)
    # _ , smallgrayblurred = cv2.threshold(smallgrayblurred, 125, 255, cv2.THRESH_BINARY)

    # gray = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)
    # _ , gray = cv2.threshold(gray, 125, 255, cv2.THRESH_BINARY)
    # gray = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    #                                       cv2.THRESH_BINARY, 199, 5)
    # _, gray = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # print(np.mean(grayblurred))
    # hsv *= (grayblurred > np.mean(grayblurred))[:,:,None]
    MIN = (0, 0, 166)
    MAX = (180, 142, 255)
    out = cv2.inRange(smallhsvblurred, MIN, MAX)
    kernel = np.ones((5, 5), np.uint8)
    out = cv2.erode(out, kernel, iterations=12)
    out = cv2.dilate(out, kernel, iterations=25)
    cv2.imshow("Frame", out * smallgray)
    cv2.imshow("Original", smallrgb)
    cv2.waitKey(0)
    return scale(out, (rgb.shape[1], rgb.shape[0])) * cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)

for rgb in images:
    out = pipeline(rgb)
    try:
        results = detector.detect(out)
    except:
        print("Contour overflow!")
    total += len(results)

print(total/len(images))