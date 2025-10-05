import cv2
import numpy as np
import argparse

image_path = '../../assets/'

def main():
    # gives a numpy array, grayscale b/c only want the text
    # image.shape is (height, width, 3 BGR channels)
    # but with using grayscale is just (height, width)
    print('a')
    image = cv2.imread(image_path + "receipt2.jpg", cv2.IMREAD_GRAYSCALE)
    image = normalize_image(image)
    cv2.imwrite(image_path + 'cleaned.jpg', image)
    return


# do this to increase contrast, which should be useful for text vs paper
def normalize_image(image):
    # make the destination buffer
    normalized_image = np.zeros((image.shape[0], image.shape[1]))

    # to maximize contrast for OCR
    image = cv2.normalize(image, normalized_image, 0, 255, cv2.NORM_MINMAX)
    return image


if __name__ == '__main__':
    main()

