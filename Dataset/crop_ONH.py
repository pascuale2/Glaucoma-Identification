# import the necessary packages
from imutils import contours
from skimage import measure
import numpy as np
import argparse
import imutils
import cv2
from google.colab.patches import cv2_imshow
import PIL
import os

def cropONH(imageName):
	left_bias = 128
	output_dim = 512
	def calc_gt_bounds(msk_path):
		gt = PIL.Image.open(msk_path)
		mw, mh = gt.size

		gt = 255 - np.array(gt)
		gt_T = gt.T

		for i in range(gt.shape[0]):
			if (127 in gt[i]) and (0 in gt[i]):
				h1 = i
				break
		for i in range(gt.shape[0]):
			if (127 in gt[-i]) and (0 in gt[-i]):
				h2 = mh - i
				break
		for i in range(gt_T.shape[0]):
			if (127 in gt_T[i]) and (0 in gt_T[i]):
				w1 = i
				break
		for i in range(gt_T.shape[0]):
			if (127 in gt_T[-i]) and (0 in gt_T[-i]):
				w2 = mw - i
				break
		return h1, h2, w1, w2
	
	data_set = {"n":"Training","g":"Training", "V":"Validation", "T":"Testing"}
	
	folderName= data_set[imageName[0]]
	folder_path = "drive/Shared drives/Capstone Summer 2020/Data/Original/" + folderName 
	
	img_path = folder_path + "/Images/" + imageName + ".jpg"
	mask_path = folder_path + "/Masks/" + imageName + ".bmp"
	
	y1, y2, x1, x2 = calc_gt_bounds(mask_path)
	image = PIL.Image.open(img_path)
	
	im_w, im_h = image.size
	var = round(0.15 * im_w)
	curr_threshold = starting_threshold = 250

	while True:
		# load the image, convert it to grayscale, and blur it
		image = cv2.imread(img_path)
  
		gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
		blur = cv2.bilateralFilter(gray,9,75,75)
		median=cv2.medianBlur(gray,5)

		# threshold the image to reveal light regions in the blurred image
		thresh = cv2.threshold(median, curr_threshold, 255, cv2.THRESH_BINARY)[1]

		# perform a series of erosions and dilations to remove any small blobs of noise from the thresholded image
		thresh = cv2.erode(thresh, None, iterations=2)
		thresh = cv2.dilate(thresh, None, iterations=4)

		# perform a connected component analysis on the thresholded image, then initialize a mask to store only the "large" components
		labels = measure.label(thresh, connectivity=2, background=0)
		mask = np.zeros(thresh.shape, dtype="uint8")

		largest_blob = 0
		# loop over the unique components
		for label in np.unique(labels):
			# if this is the background label, ignore it
			if label == 0:
				continue
			# otherwise, construct the label mask and count the number of pixels 
			labelMask = np.zeros(thresh.shape, dtype="uint8")
			labelMask[labels == label] = 255
			numPixels = cv2.countNonZero(labelMask)

			# if the number of pixels in the component is sufficiently large, then add it to our mask of "large blobs"
			if numPixels > largest_blob:
				largest_blob = numPixels 
				mask = labelMask


		# find the contours in the mask, then sort them from left to right
		cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
		cnts = imutils.grab_contours(cnts)
	
		#If there is nothing found for the image
		if cnts == []:
			curr_threshold -= 10
			
			continue
		print("Decreased threshold by: {}".format(starting_threshold - curr_threshold))
		cnts = contours.sort_contours(cnts)[0]

		# loop over the contours
		for (i, c) in enumerate(cnts):
			(x, y, w, h) = cv2.boundingRect(c)

			center = (round(x+(w/2)), round(y+(h/2)))
			cv2.putText(image, "O", center, cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
   
			box_radius = output_dim//2
			tl_pt = [center[0]-box_radius-left_bias, center[1]-box_radius-left_bias]
			br_pt = [center[0]+box_radius, center[1]+box_radius]

			#Check if TL point is out of bounds
			if tl_pt[0] < 0:
				neg = tl_pt[0] * -1
				tl_pt[0] = 0
				br_pt[0] += neg

			if tl_pt[1] < 0:
				neg = tl_pt[1] * -1
				tl_pt[1] = 0
				br_pt[1] += neg

			#Check if BR point is out of bounds
			if br_pt[0] > im_w:
				pos = im_w-br_pt[0]
				br_pt[0] = im_w
				tl_pt[0] -= pos
			if br_pt[1] > im_h:
				pos = im_h-br_pt[1]
				br_pt[1] = im_h
				tl_pt[1] -= pos
	
			cv2.rectangle(image,tuple(tl_pt) , tuple(br_pt), (255,0,0), 8)
			break
		
		# Checks if Ground Truth Bounds are within Mask Bounds
		if not ((x1 >= tl_pt[0]) and (x2 <= br_pt[0]) and (y1 >= tl_pt[1]) and (y2 <= br_pt[1])):
			print("-"*50)
			print("GT Bounds aren't within Mask Bounds for the IMAGE: {}".format(imageName))
			print("-"*50)
   
		print("Wid: ", br_pt[0]-tl_pt[0], "\tHei:", br_pt[1]-tl_pt[1])
		
		# Opens a new image without the rectangles and stuff
		orig_image = cv2.imread(img_path)
		gt = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
		# Crops the 'image/ground truth' to the mask bounds and dsiplays the cropped 'image/ground truth'
		cropped_image = orig_image[tl_pt[1]:br_pt[1],tl_pt[0]:br_pt[0]]
		cropped_gt = gt[tl_pt[1]:br_pt[1],tl_pt[0]:br_pt[0]]
		
		# print(np.asarray(cropped_gt).shape)
		c_img_path = "drive/Shared drives/Capstone Summer 2020/Data/Testing/Images/" + imageName + ".jpg"
		c_msk_path = "drive/Shared drives/Capstone Summer 2020/Data/Testing/Masks/" + imageName + ".bmp"

		# SAVES THE CROPPED IMAGES AND MASKS
		cv2.imwrite(c_img_path, cropped_image)
		cv2.imwrite(c_msk_path, cropped_gt)
		# cv2_imshow(cropped_gt)
		break
