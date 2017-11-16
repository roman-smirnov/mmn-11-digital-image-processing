import cv2
import numpy as np
import matplotlib.pyplot as plt

ORIGINAL_IMAGE_PATH = "images/original.jpg"  # load this image
OUTPUT_IMAGE_PATH = "images/output.jpg"  # write output to this file


def flip_horizontal(old_img):
    """ flip the image over the y axis """
    rows = old_img.shape[0]
    cols = old_img.shape[1]
    new_img = np.zeros((rows, cols), np.uint8)  # create a black grayscale image having same size as original
    for c in range(1, cols + 1):  # iterate over the image columns (+1 because range not inclusive function)
        new_img[:, c - 1:c] = old_img[:, cols - c:cols - c + 1]  # copy each col in old img to its mirror loc in new img
    return new_img


def scale_down(old_img, scale_factor):
    """ scale the resolution of the image down by a factor """
    rows = int(old_img.shape[0] / scale_factor)
    cols = int(old_img.shape[1] / scale_factor)
    new_img = np.zeros((rows, cols), np.uint8)  # create a black grayscale image of scaled down size
    for r in range(0, rows):
        for c in range(0, cols):
            new_img[r, c] = old_img[r * scale_factor, c * scale_factor]  # copy only pixels in steps of scale_factor
    return new_img


def scale_up(old_img, scale_factor):
    """ scale the resolution of the image up by the given factor """
    rows = old_img.shape[0] * scale_factor
    cols = old_img.shape[1] * scale_factor
    new_img = np.zeros((rows, cols), np.uint8)  # create a black grayscale image of scaled up size
    for r in range(0, rows):  # iterate and duplicate alternating pixels to upscale resolution
        for c in range(0, cols):
            new_img[r, c] = old_img[int(r / scale_factor), int(c / scale_factor)]  # copy the pixel from original
    return new_img


def rotate_by_90(old_img):
    """ rotate image by 90 degrees clockwise by simple transposition """
    rows = old_img.shape[0]
    cols = old_img.shape[1]
    new_img = np.zeros((cols, rows), np.uint8)  # create a black grayscale image of inverted size
    for r in range(0, rows):
        for c in range(0, cols):
            new_img[c, r] = old_img[rows - r - 1, cols - c - 1]  # switch between row and column for each pixel
    return new_img


# WARNING! currently implemented for up to 90 deg rotation
# TODO! re-implement with array operations from numpy to get x1k+ improvement in performance
# TODO! use rotated corner positions to calc new image dimensions instead of dividing to cases
def rotate_by_arbitrary(old_img, angle_deg):
    """ rotate image by some arbitrary degree clockwise by bilinear interpolation """
    old_cols = old_img.shape[1]
    old_rows = old_img.shape[0]
    new_cols, new_rows = get_new_dimensions(angle_deg, old_cols, old_rows)
    new_img = np.zeros((new_rows, new_cols), np.uint8)  # create a black grayscale image of appropriate size
    center_x = new_cols / 2
    center_y = new_rows / 2
    rot_coefs = get_rotation_coefficients(angle_deg, center_x, center_y)
    rows_offset = (new_rows - old_rows) / 2
    cols_offset = (new_cols - old_cols) / 2
    for r in range(0, new_rows):
        for c in range(0, new_cols):
            x, y = affine_transformation(c-center_x, r-center_y, *rot_coefs)
            x -= cols_offset
            y -= rows_offset
            if 0 <= x < old_cols and 0 <= y < old_rows:
                new_img[r, c] = bilinear_interpolation(old_img, x, y)
    return new_img


def get_new_dimensions(angle_deg, old_width, old_height):
    angle_rad = np.deg2rad(angle_deg)
    sine = np.sin(angle_rad)
    cosine = np.cos(angle_rad)
    new_width = int(old_width * cosine + old_height * sine)
    new_height = int(old_width * sine + old_height * cosine)
    return new_width, new_height


def get_rotation_coefficients(angle_deg, center_x=0.0, center_y=0.0):
    angle_rad = np.deg2rad(angle_deg)
    return np.cos(angle_rad), -np.sin(angle_rad), np.sin(angle_rad), np.cos(angle_rad), center_x, center_y


def affine_transformation(x, y, a, b, c, d, e, f):
    """ perform an affine 2d transformation using the given matrix coefficients"""
    return x * a + y * b + e, x * c + y * d + f


def linear_interpolation(x, x_0, x_1, y_0, y_1):
    """ return the linearly interpolated value between two points """
    return y_0+(x-x_0)*(y_1-y_0)/(x_1 - x_0)


def bilinear_interpolation(img, x, y):
    """ perform bilinear interpolation for a value between 4 points and return the result """
    cols, rows = img.shape[1], img.shape[0]  # image bounds
    x_0, x_1, y_0, y_1 = int(x), int(x)+1, int(y), int(y)+1
    result = img[y_0, x_0]  # nearest neighbour value to be return in case interp not possible
    if 0 <= x_1 < cols and 0 <= y_1 < rows:  # check pixel is within image bound
        interp_val_a = linear_interpolation(x, x_0, x_1, int(img[y_0, x_0]), int(img[y_0, x_1]))
        interp_val_b = linear_interpolation(x, x_0, x_1, int(img[y_1, x_0]), int(img[y_1, x_1]))
        result = linear_interpolation(y, y_0, y_1, interp_val_a, interp_val_b)
    return result


original_img = cv2.imread(ORIGINAL_IMAGE_PATH, cv2.IMREAD_GRAYSCALE)
print("original image size: " + str(original_img.shape))

# output_img = flip_horizontal(original_img)
# output_img = scale_down(original_img, 2)
# output_img = scale_up(original_img, 2)
# output_img = scale_up( scale_down(original_img, 2), 2)  # downscale than upscale
# output_img = rotate_by_90(original_img)
output_img = rotate_by_arbitrary(original_img, 30)

print("output image size: " + str(output_img.shape))

# cv2.imshow("output", output_img)  # Show image
# cv2.waitKey(0)  # display until key pressed
cv2.imwrite(OUTPUT_IMAGE_PATH, output_img)  # write output image to file

# alternate way to display output image
# plt.imshow(result_img, cmap='gray')
# plt.show()
