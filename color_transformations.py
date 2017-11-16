import cv2
import numpy as np

ORIGINAL_IMAGE_PATH = "images/original.jpg"  # load this image
OUTPUT_IMAGE_PATH = "images/output.jpg"  # write output to this file


def perform_thresholding(image, threshold):
    return np.vectorize(lambda x: 0 if x < threshold else 255)(image)


def scale_image(old_img, scale_factor):
    scaled_height, scaled_width = ((x * scale_factor) for x in old_img.shape)
    new_img = np.zeros((scaled_height, scaled_width), np.uint8)
    dx, dy = np.meshgrid(np.arange(scaled_width - 1), np.arange(scaled_height - 1))
    sx, sy = dx / scale_factor, dy / scale_factor
    sx, sy = sx.round().astype(int), sy.round().astype(int)  # nearest neighbour
    new_img[dy, dx] = old_img[sy, sx]
    return new_img


def scale_and_halftone(old_img, scale_factor):
    scaled_height, scaled_width = ((x * scale_factor) for x in old_img.shape)
    new_img = scale_image(old_img, scale_factor)
    for x in range(0, scaled_width):
        for y in range(0, scaled_height):
            halftone = get_halftone(new_img[y, x])
            if x % scale_factor == 0 and y % scale_factor == 0:  # zeroth pixel
                new_img[y, x] = halftone[0]
            elif x % scale_factor == 0 and y % scale_factor != 0:  # first pixel
                new_img[y, x] = halftone[1]
            elif x % scale_factor != 0 and y % scale_factor == 0:  # 2nd pixel
                new_img[y, x] = halftone[2]
            else:
                new_img[y, x] = halftone[3]
    return new_img


# this function is wrong - TODO: don't code at 3 am
def scale_and_halftone_via_error_diffusion(old_img, scale_factor):
    scaled_height, scaled_width = ((x * scale_factor) for x in old_img.shape)
    new_img = scale_image(old_img, scale_factor)
    for x in range(0, scaled_width):
        for y in range(0, scaled_height):
            error = new_img[y, x]
            halftone = get_halftone(new_img[y, x])
            if x % scale_factor == 0 and y % scale_factor == 0:  # zeroth pixel
                new_img[y, x] = halftone[0]
            elif x % scale_factor == 0 and y % scale_factor != 0:  # first pixel
                new_img[y, x] = halftone[1]
            elif x % scale_factor != 0 and y % scale_factor == 0:  # 2nd pixel
                new_img[y, x] = halftone[2]
            else:
                new_img[y, x] = halftone[3]

            error = int((int(error) - int(new_img[y, x])) / 8)
            if x < scaled_width - 1 and y < scaled_height - 1:
                new_img[y + 1, x] += error
                new_img[y, x + 1] += error
                new_img[y + 1, x + 1] += error
    return new_img


def dither(src_img):
    height, width = src_img.shape[0] - 1, src_img.shape[1] - 1
    dither_matrix = np.asarray([51, 153, 204, 100], np.uint8)
    dither_matrix = np.resize(dither_matrix, (height + 1, width + 1))
    src_img = np.where(src_img < dither_matrix, 0, 255)
    return src_img


# binarization algorithm
def floyd_steinberg(src_img):
    height, width = src_img.shape[0] - 1, src_img.shape[1] - 1
    src_img = src_img.astype(int)

    for y in range(0, height):
        for x in range(0, width):
            intensity = src_img[y, x]
            src_img[y, x] = 0 if intensity < 128 else 255
            # diffuse the error
            error = intensity - src_img[y, x]
            src_img[y + 1, x] += (3*error / 8)
            src_img[y, x + 1] += (3*error / 8)
            src_img[y + 1, x + 1] += (error / 4)

    return src_img


img = cv2.imread(ORIGINAL_IMAGE_PATH, cv2.IMREAD_GRAYSCALE)
# output_img = perform_thresholding(img, 255 / 2)
# output_img = scale_and_halftone(img, 2)

output_img = img
for i in range(1, 10):
    output_img = floyd_steinberg(output_img)

# output_img = scale_and_halftone_via_error_diffusion(output_img, 2)

cv2.imwrite(OUTPUT_IMAGE_PATH, output_img)  # write output image to file
