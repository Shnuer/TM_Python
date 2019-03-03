import cv2
import numpy as np
from matplotlib import pyplot as plt


def operation(frame,example,meth):
	w, h, l = example.shape
	frame_met = frame.copy()
	method = eval(meth)
	res = cv2.matchTemplate(frame_met,example,method)

	min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)	

	if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
		top_left = min_loc
	else:
		top_left = max_loc
	bottom_right = (top_left[0] + w, top_left[1] + h)
	cv2.rectangle(frame,top_left, bottom_right, 255, 2)
	return frame








sample = cv2.imread("sample.png")
sample2=sample.copy()
example = cv2.imread("example.png")
cap = cv2.VideoCapture(0)
while 1:
	
	_, frame = cap.read()
	
	methods = ['cv2.TM_CCOEFF', 'cv2.TM_CCOEFF_NORMED', 'cv2.TM_CCORR', 'cv2.TM_CCORR_NORMED', 'cv2.TM_SQDIFF', 'cv2.TM_SQDIFF_NORMED']

	cv2.imshow('TM_CCOEFF',operation(frame,example,methods[0]))
	cv2.imshow('TM_CCOEFF_NORMED',operation(frame,example,methods[1]))
	cv2.imshow('TM_CCORR',operation(frame,example,methods[2]))
	cv2.imshow('TM_CCORR_NORMED',operation(frame,example,methods[3]))
	# cv2.imshow('TM_SQDIFF',operation(frame,example,methods[4]))
	# cv2.imshow('TM_SQDIFF_NORMED',operation(frame,example,methods[5]))

	k = cv2.waitKey(5) & 0xFF
	if k == 27:
		break

cv2.destroyAllWindows()