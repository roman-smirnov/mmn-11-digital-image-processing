#################################################################
# Digital Image Processing
# MMN 11
# Roman Smirnov
# 29/10/2017
# Python version 3.5
#################################################################

import cv2
import numpy as np

ORIGINAL_IMAGE_PATH = "original.jpg"  # load this image
OUTPUT_IMG_NAME = "Output Image"  # name of output image


def flip_img_horizontal(src_img):
    """ mirror the image across the y axis - the impl is not done in place"""
    height, width = src_img.shape[0] - 1, src_img.shape[1] - 1
    out_img = np.zeros((height, width), np.uint8)  # create a new black greyscale img with same dimension
    dx, dy = np.meshgrid(np.arange(width), np.arange(height))  # create indices array for vectorized transformation
    sx, sy = width - dx, dy  # define the horizontal flip transformation
    sx, sy = sx.round().astype(int), sy.round().astype(int)  # get nearest neighbour source pixel
    out_img[dy, dx] = src_img[sy, sx]  # perform the transformations in parallel
    return out_img


# load the img
img = cv2.imread(ORIGINAL_IMAGE_PATH, cv2.IMREAD_GRAYSCALE)
# flip the img
output_img = flip_img_horizontal(img)
# display the output image
cv2.imshow(OUTPUT_IMG_NAME, output_img)
# wait for a key press
cv2.waitKey(0)
