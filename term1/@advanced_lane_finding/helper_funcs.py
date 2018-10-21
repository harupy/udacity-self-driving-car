import os
import  shutil
import glob
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
import cv2


def empty_dir(dir_path):
	for file in os.listdir(dir_path):
		fpath = os.path.join(dir_path, file)
		try:
			if os.path.isfile(fpath):
				os.unlink(fpath)
			elif os.path.isdir(fpath): shutil.rmtree(fpath)
		except Exception as e:
			print(e)


def show_img(img):
	if isinstance(img, str):
		img = cv2.imread(img)

	plt.imshow(img[:, :, ::-1])
	plt.show()


def bgr2gray(img):
	return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def bgr2hls(img):
	return cv2.cvtColor(img, cv2.COLOR_BGR2HLS)


def canny(img, low_threshold, high_threshold):
	return cv2.Canny(img, low_threshold, high_threshold)


def gaussian_blur(img, kernel_size):
	return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)


def mask_roi(img, vertices):
	mask = np.zeros_like(img)

	if len(img.shape) > 2:
		channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
		ignore_mask_color = (255,) * channel_count
	else:
		ignore_mask_color = 255

	cv2.fillPoly(mask, vertices, ignore_mask_color)
	masked_image = cv2.bitwise_and(img, mask)
	return masked_image


def draw_lines(img, lines, color=(255, 0, 0), thickness=2):
	for line in lines:
		for x1, y1, x2, y2 in line:
			cv2.line(img, (x1, y1), (x2, y2), color, thickness)

	
def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
	lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]),
							minLineLength=min_line_len, maxLineGap=max_line_gap)
	line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
	draw_lines(line_img, lines)
	return line_img


def weighted_img(img, initial_img, alpha=0.8, beta=1., gamma=0.):
	return cv2.addWeighted(initial_img, alpha, img, beta, gamma)


def calc_roi_vertices(img):
	h, w = img.shape[:2]
	vertices = np.array([
		[int(0.4 * w), int(0.65 * h)],  # top left
		[int(0.6 * w), int(0.65 * h)],  # top right
		[int(0.9 * w), h - 1],  # bottom right
		[int(0.1 * w), h - 1],  # bottom left
	])
	return vertices.reshape((-1, 4, 2))


def calc_warp_dst(src, img_shape):
	h, w = img_shape[:2]
	margin_x = 300
	margin_y = 0
	dst = np.array([
		[margin_x, margin_y],
		[w - margin_x, margin_y],
		[w - margin_x, h - 1],
		[margin_x, h - 1]], dtype="float32")
	return dst



def abs_sobel_thresh(img, orient='x', sobel_kernel=3, thresh=(0, 255)):
	# Convert to grayscale
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	# Apply x or y gradient with the OpenCV Sobel() function
	# and take the absolute value
	if orient == 'x':
		abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 1, 0))
	if orient == 'y':
		abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 0, 1))
	# Rescale back to 8 bit integer
	scaled_sobel = np.uint8(255 * abs_sobel / np.max(abs_sobel))
	# Create a copy and apply the threshold
	grad_binary = np.zeros_like(scaled_sobel)
	# Here I'm using inclusive (>=, <=) thresholds, but exclusive is ok too
	grad_binary[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])] = 1

	return grad_binary


def mag_thresh(img, sobel_kernel=3, thresh=(0, 255)):
	# Convert to grayscale
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	# Take both Sobel x and y gradients
	sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
	sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
	# Calculate the gradient magnitude
	gradmag = np.sqrt(sobelx**2 + sobely**2)
	# Rescale to 8 bit
	scale_factor = np.max(gradmag) / 255
	gradmag = (gradmag / scale_factor).astype(np.uint8)
	# Create a binary image of ones where threshold is met, zeros otherwise
	mag_binary = np.zeros_like(gradmag)
	mag_binary[(gradmag >= thresh[0]) & (gradmag <= thresh[1])] = 1

	# Return the binary image
	return mag_binary


def dir_threshold(img, sobel_kernel=3, thresh=(0, np.pi / 2)):
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	# Calculate the x and y gradients
	sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
	sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
	# Take the absolute value of the gradient direction,
	# apply a threshold, and create a binary image result
	absgraddir = np.arctan2(np.absolute(sobely), np.absolute(sobelx))
	dir_binary = np.zeros_like(absgraddir)
	dir_binary[(absgraddir >= thresh[0]) & (absgraddir <= thresh[1])] = 1
	return dir_binary


def hls_thresh(img, lower, upper):
	hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
	return cv2.inRange(hls, lower, upper)


def extract_white_yellow(img):
	img_hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
	# white color mask
	lower = np.uint8([0, 200, 0])
	upper = np.uint8([255, 255, 255])
	white_mask = cv2.inRange(img_hls, lower, upper)
	# yellow color mask
	lower = np.uint8([10, 0, 100])
	upper = np.uint8([40, 255, 255])
	yellow_mask = cv2.inRange(img_hls, lower, upper)
	# combine masks
	mask = cv2.bitwise_or(white_mask, yellow_mask).astype(np.uint8)
	return cv2.bitwise_and(img, img, mask=mask)
		

def calc_angles(lines):
	x1, y1, x2, y2 = lines.squeeze().T
	return np.rad2deg(np.arctan2(y2 - y1, x2 - x1))


def calc_mid_points(lines):
	x1, y1, x2, y2 = lines.squeeze().T
	return (x1 + x2) / 2, (y1 + y2) / 2


def remove_outliers(lines, img):
	h, w = img.shape[:2]
	angles = calc_angles(lines)
	mx, my = calc_mid_points(lines)
	idx = (np.abs(angles) > 25) & (np.abs(angles) < 70) & \
		  (((mx < (w / 2)) & (angles < 0)) | ((mx > w / 2) & (angles > 0))) & \
		  (my > h / 2)
	return lines[idx]


def separate_lines(lines):
	angles = calc_angles(lines)
	left_lines = lines[angles < 0]
	right_lines = lines[angles > 0]
	return left_lines, right_lines


def fit_lines(lines, y1, y2):
	x, y = lines.squeeze().reshape(-1, 2).T
	p = np.polyfit(y, x, 1)
	x1 = np.polyval(p, y1)
	x2 = np.polyval(p, y2)
	return np.array([[[x1, y1, x2, y2]]]).astype(int)


def calc_hist(img):
	# Grab only the bottom half of the image
	# Lane lines are likely to be mostly vertical nearest to the car
	bottom_half = img[img.shape[0] // 2:, :]
	# Sum across image pixels vertically - make sure to set an `axis`
	# i.e. the highest areas of vertical lines should be larger values
	hist = np.sum(bottom_half, axis=0)

	return hist


def pipeline_P1(img):
	"""
	The pipeplie I implemented in the 1st project
	"""
	# save_path = os.path.join('video_frames', datetime.now().strftime("%Y%d%H%M%S%f") + '.jpg')
	if len(img.shape) == 3:
		img_wy = extract_white_yellow(img)
		img_gray = bgr2gray(img_wy)
	else:
		img_gray = img.copy()
	kernel_size = 5
	low_thresh = 150
	high_thresh = 200
	rho = 1
	theta = np.pi / 180
	threshold = 25
	min_line_len = 8
	max_line_gap = 5
	roi_vertices = calc_roi_vertices(img)
	h, w = img.shape[:2]
	img_blur = gaussian_blur(img_gray, kernel_size=kernel_size)
	img_canny = canny(img_blur, low_thresh, high_thresh)
	img_roi = mask_roi(img_canny, roi_vertices)
	lines = cv2.HoughLinesP(img_roi, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)
	lines = remove_outliers(lines, img)
	left_lines, right_lines = separate_lines(lines)
	left_fit_line = fit_lines(left_lines, roi_vertices[0][0][1], h)
	right_fit_line = fit_lines(right_lines, roi_vertices[0][0][1], h)
	line_img = np.zeros_like(img)
	draw_lines(line_img, left_fit_line, color=(0, 0, 255), thickness=10)
	draw_lines(line_img, right_fit_line, color=(255, 0, 0), thickness=10)
	img_final = weighted_img(line_img, img, alpha=0.95)
	draw_lines(img_final, left_lines, (0, 0, 255), 1)
	draw_lines(img_final, right_lines, (255, 0, 0), 1)
	cv2.polylines(img_final, roi_vertices, True, (0, 255, 0), 1, 4)
	return img_final, left_fit_line.squeeze(), right_fit_line.squeeze()



if __name__ == '__main__':
	pass