import cv2
import numpy as np
from skimage.viewer import ImageViewer


def remove_rows(image, rows, cols):
    newrows = int(rows / 2)
    newimg = np.zeros((newrows, cols), np.uint8)
    for r in range(1, newrows + 1):
        newimg[r - 1:r, :] = image[r * 2 - 1:r * 2, :]
    return newimg


img = cv2.imread('pirate.jpg', cv2.IMREAD_GRAYSCALE)
print(img.shape)
img = remove_rows(img, img.shape[0], img.shape[1])

viewer = ImageViewer(img)
viewer.show()
