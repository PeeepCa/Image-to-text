import cv2
from PIL import Image
import pytesseract
import argparse
import os
import numpy as np

image = cv2.imread("dummy.jpg")

dilated_img = cv2.dilate(image[:,:,1], np.ones((7, 7), np.uint8))
bg_img = cv2.medianBlur(dilated_img, 21)

diff_img = 255 - cv2.absdiff(image[:,:,1], bg_img)

norm_img = cv2.normalize(diff_img, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
cv2.imshow('norm_img', cv2.resize(norm_img, (0, 0), fx = 0.5, fy = 0.5))

th = cv2.threshold(norm_img, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
cv2.imshow('th', cv2.resize(th, (0, 0), fx = 0.5, fy = 0.5))

filename = "{}.png".format(os.getpid())
cv2.imwrite(filename,th)

text = pytesseract.image_to_string(Image.open(filename), lang='ces')
print(text)
os.remove(filename)
