import numpy as np
import cv2
import glob
from skimage.feature import hog
import pickle
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from tqdm import tqdm 
from settings import *


def get_hog_features(img, orient, pix_per_cell, cell_per_block, vis=False, feature_vec=True):
	"""
	Return HOG features and visualization (optionally)
	"""
	if vis == True:
		features, hog_image = hog(img, orientations=orient, pixels_per_cell=(pix_per_cell, pix_per_cell),
			cells_per_block=(cell_per_block, cell_per_block), transform_sqrt=False,
			visualise=True, feature_vector=False)
		return features, hog_image
	else:
		features = hog(img, orientations=orient, pixels_per_cell=(pix_per_cell, pix_per_cell),
			cells_per_block=(cell_per_block, cell_per_block), transform_sqrt=False,
			visualise=False, feature_vector=feature_vec)
		return features


# Define a function to extract features from a list of images
# Have this function call bin_spatial() and color_hist()
def extract_features(imgs, color_space, orient, 
	pix_per_cell, cell_per_block, hog_channel, hog_feat=True):

	# Create a list to append feature vectors to
	features = []

	# Iterate through the list of images
	for file in tqdm(imgs):

		file_features = []
		feature_image = mpimg.imread(file)
		if len(feature_image.shape) == 2:
			feature_image = cv2.cvtColor(feature_image, cv2.COLOR_GRAY2BGR) 
		if hog_feat == True:
		# Call get_hog_features() with vis=False, feature_vec=True
			if hog_channel == 'ALL':
				hog_features = []
				for channel in range(feature_image.shape[2]):
					hog_features.append(get_hog_features(feature_image[:,:,channel],
						orient, pix_per_cell, cell_per_block,
						vis=False, feature_vec=True))
				hog_features = np.ravel(hog_features)
			else:
				hog_features = get_hog_features(feature_image[:,:,hog_channel], orient,
					pix_per_cell, cell_per_block, vis=False, feature_vec=True)
			# Append the new feature vector to the features list
			file_features.append(hog_features)
		features.append(np.concatenate(file_features))
	# Return list of feature vectors
	return features


# Define a function to extract features from a single image window
# This function is very similar to extract_features()
# just for a single image rather than list of images
def single_img_features(img, color_space='RGB', orient=9,
	pix_per_cell=8, cell_per_block=2, hog_channel=0,
	hog_feat=True):
	#1) Define an empty list to receive features
	img_features = []

	feature_image = np.copy(img)
	
	# Compute HOG features if flag is set
	if hog_feat == True:
		if hog_channel == 'ALL':
			hog_features = []
			for channel in range(feature_image.shape[2]):
				hog_features.extend(get_hog_features(feature_image[:,:,channel],
									orient, pix_per_cell, cell_per_block,
									vis=False, feature_vec=True))
		else:
			hog_features = get_hog_features(feature_image[:,:,hog_channel], orient,
						pix_per_cell, cell_per_block, vis=False, feature_vec=True)
		#8) Append features to list
		img_features.append(hog_features)

	#9) Return concatenated array of features
	return np.concatenate(img_features)


if __name__ == '__main__':
  
	# Read in some MNIST image examples
	eight_image = mpimg.imread('example_images/example_eight.jpg')
	two_image = mpimg.imread('example_images/example_two.jpg')

	# Plot the examples
	fig = plt.figure(figsize=(18,14))
	plt.subplot(121) 
	plt.imshow(eight_image)
	plt.title('Example Eight Image')
	plt.subplot(122)
	plt.imshow(two_image)
	plt.title('Example Two Image')

	# Extract hog features from these two images
	eight_bgr = cv2.cvtColor(eight_image, cv2.COLOR_GRAY2BGR)
	two_bgr = cv2.cvtColor(two_image, cv2.COLOR_GRAY2BGR)

	# Define HOG parameters
	orient = 9
	pix_per_cell = 2
	cell_per_block = 1

	features = []
	hog_images = []

	# Call our function with vis=True to see an image output
	for i in range(eight_bgr.shape[2]):
		_, hog_image = get_hog_features(eight_bgr[:,:,i], orient, 
			pix_per_cell, cell_per_block, 
			vis=True, feature_vec=False)
		hog_images.append(hog_image)
		_, hog_image = get_hog_features(two_bgr[:,:,i], orient, 
			pix_per_cell, cell_per_block, 
			vis=True, feature_vec=False)
		hog_images.append(hog_image)


	# Plot the examples
	fig = plt.figure(figsize=(18,14))
	plt.subplot(341) 
	plt.imshow(eight_bgr[:,:,0], cmap='gray')
	plt.title('Eight > B in BGR')
	plt.subplot(342)
	plt.imshow(hog_images[0], cmap='gray')
	plt.title('Eight > B in BGR > HOG')
	plt.subplot(343)
	plt.imshow(two_bgr[:,:,0], cmap='gray')
	plt.title('Two > B in BGR')
	plt.subplot(344)
	plt.imshow(hog_images[1], cmap='gray')
	plt.title('Two > B in BGR > HOG')
	plt.subplot(345)
	plt.imshow(eight_bgr[:,:,1], cmap='gray')
	plt.title('Eight > G in BGR')
	plt.subplot(346)
	plt.imshow(hog_images[2], cmap='gray')
	plt.title('Eight > G in BGR > HOG')
	plt.subplot(347)
	plt.imshow(two_bgr[:,:,1], cmap='gray')
	plt.title('Two > G in BGR')
	plt.subplot(348)
	plt.imshow(hog_images[3], cmap='gray')
	plt.title('Two > G in BGR > HOG')
	plt.subplot(349)
	plt.imshow(eight_bgr[:,:,2], cmap='gray')
	plt.title('Eight > R in BGR')
	plt.subplot(3,4,10)
	plt.imshow(hog_images[4], cmap='gray')
	plt.title('Eight > R in BGR > HOG')
	plt.subplot(3,4,11)
	plt.imshow(two_bgr[:,:,2], cmap='gray')
	plt.title('Two > R in BGR')
	plt.subplot(3,4,12)
	plt.imshow(hog_images[5], cmap='gray')
	plt.title('Eight > R in BGR > HOG')
	fig.tight_layout()

	# Non-digit image
	image = mpimg.imread('example_images/non_digit.jpg')


	if len(image.shape) == 2:
		image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
	if image.shape[2] == 1:
		image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

	for i in range(image.shape[2]):
		_, hog_image = get_hog_features(image[:,:,i], orient,
			pix_per_cell, cell_per_block,
			vis=True, feature_vec=False)
		hog_images.append(hog_image)

	# Plot the example
	fig = plt.figure(figsize=(18,14))
	plt.subplot(321) 
	plt.imshow(image[:,:,0], cmap='gray')
	plt.title('Non-digit > B in BGR')
	plt.subplot(342)
	plt.imshow(hog_images[6], cmap='gray')
	plt.title('Non-digit > B in BGR > HOG')
	plt.subplot(323) 
	plt.imshow(image[:,:,1], cmap='gray')
	plt.title('Non-digit > G in BGR')
	plt.subplot(344)
	plt.imshow(hog_images[7], cmap='gray')
	plt.title('Non-digit > G in BGR > HOG')
	plt.subplot(325) 
	plt.imshow(image[:,:,2], cmap='gray')
	plt.title('Non-digit > R in BGR')
	plt.subplot(346)
	plt.imshow(hog_images[8], cmap='gray')
	plt.title('Non-digit > R in BGR')
	fig.tight_layout()


