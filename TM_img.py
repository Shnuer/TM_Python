import cv2
import numpy as np
from matplotlib import pyplot as plt

sample = cv2.imread("sample.png")
sample2=sample.copy()
example = cv2.imread("example.png")

w, h, l = example.shape

# w, h = example.shape[::-1]


methods = ['cv2.TM_CCOEFF', 'cv2.TM_CCOEFF_NORMED', 'cv2.TM_CCORR', 'cv2.TM_CCORR_NORMED', 'cv2.TM_SQDIFF', 'cv2.TM_SQDIFF_NORMED']

for meth in methods:
	sample = sample2.copy()
	method = eval(meth)

	res = cv2.matchTemplate(sample,example,method)
	min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

# 1
	if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
		top_left = min_loc
	else:
		top_left = max_loc
	bottom_right = (top_left[0] + w, top_left[1] + h)

	cv2.rectangle(sample,top_left, bottom_right, 255, 2)

	plt.subplot(121),plt.imshow(res,cmap = 'gray')
	plt.title('Matching Result'), plt.xticks([]), plt.yticks([])
	plt.subplot(122),plt.imshow(sample,cmap = 'gray')
	plt.title('Detected Point'), plt.xticks([]), plt.yticks([])
	plt.suptitle(meth)

	plt.show()