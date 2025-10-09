import cv2
import numpy as np
from rembg import remove
import pytesseract

from matplotlib import pyplot as plt
import argparse

image_path = '../../assets/'


def main():
    # gives a numpy array, grayscale b/c only want the text
    # image.shape is (height, width, 3 BGR channels)
    # but with using grayscale is just (height, width)
    img = cv2.imread(image_path + "receipt.jpg", cv2.IMREAD_GRAYSCALE)
    # img = cv2.imread(image_path + "receipt3.jpg")

    # img = process_image(img)
    # img = remove(img)
    img = reduce_shadow(img)

    print(pytesseract.image_to_string(img))
    # img = remove(img)

    # img = f(img)
    # img = adaptive_threshold(img)
    # img = process_image(img)
    # img = remove_bg(img)
    # img = deskew_image(img)
    # img = deskew_image(img)
    # img = normalize_image(img)
    # img = threshold_image(img)
    cv2.imwrite(image_path + 'gaussian9.jpg', img)
    return

def f(img):
    h, w = img.shape[:2]
    # threshold on white
    # Define lower and uppper limits
    lower = np.array([200, 200, 200])
    upper = np.array([255, 255, 255])
    thresh = cv2.inRange(img, lower, upper)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (20, 20))
    morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    mask = 255 - morph
    result = cv2.bitwise_and(img, img, mask=mask)
    return result
def reduce_blur(img):
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# definitely has some room for improvement
def reduce_shadow(img):
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # estimate and flatten lighting, large sigma will have more smoothing
    # goal's just to approx the bg lighting

    # so this will just blur the image
    blur = cv2.GaussianBlur(img, (0, 0), sigmaX=15, sigmaY=15)

    # this will normalize the image, and this is what we want since
    # blurring will greatly change the sharp content (e.g. text, has outlines)
    # so then we can figure out what's been changed a lot and what hasn't,
    # which helps to distinguish b/w bg and text, although it's not perfect
    divide = cv2.divide(img, blur, scale=255)

    # otsu's method is used here because we've effective split the foreground and background,
    # giving us the bimodal lighting that otsu's needs to work properly
    thresh = cv2.threshold(divide, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

    # kernel for connecting broken letters
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

    # closing = dilation then erosion
    # gaussian 8 = 1,3
    # 1, 3 seems to be best, can always go back though
    dilated = cv2.dilate(thresh, kernel, iterations=1)
    closed = cv2.erode(dilated, kernel, iterations=3)
    return closed

    # mask = np.zeros(divide.shape[:2], np.uint8)
    # bgdModel = np.zeros((1, 65), np.float64)
    # fgdModel = np.zeros((1, 65), np.float64)
    # rect = (x, y, w, h)  # bounding box around receipt
    # cv2.grabCut(divide, mask, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)
    # mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
    # segmented = divide * mask2[:, :, np.newaxis]


# do this to increase contrast, which should be useful for text vs paper
def normalize_image(img):
    # make the destination buffer
    normalized_image = np.zeros((img.shape[0], img.shape[1]))

    # to maximize contrast for OCR
    image = cv2.normalize(img, normalized_image, 0, 255, cv2.NORM_MINMAX)
    return image


def process_image(img):
        height, width = img.shape[:2]
        margin = 0.05  # 5% margin
        x = int(width * margin)
        y = int(height * margin)
        rect = (x, y, width - 2 * x, height - 2 * y)

        # Create mask and models

        mask = np.zeros(img.shape[:2], np.uint8)
        bgdModel = np.zeros((1, 65), np.float64)
        fgdModel = np.zeros((1, 65), np.float64)

        # Apply GrabCut

        cv2.grabCut(img, mask, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)

        # Create binary mask

        mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype("uint8")

        # Convert to RGBA and apply mask to alpha channel

        output_rgba = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
        output_rgba[:, :, 3] = mask2 * 255  # 0 for background, 255 for foreground
        return output_rgba

        # Save as PNG to preserve transparency

def threshold_image(img):
    # don't really care about the threshold value
    _, binary = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binary

def remove_bg(img):
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    kernel = np.ones((3, 3), np.uint8)
    cleaned = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
    return cleaned

# fixing rotated/tilted text
def deskew_image(img):
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    inverted = cv2.bitwise_not(img)
    # return inverted
    coords = np.column_stack(np.where(inverted > 0))
    angle = cv2.minAreaRect(coords)
    print(angle)
    angle = angle[-1]
    print(angle)
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle
    (h, w) = inverted.shape
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    deskewed = cv2.warpAffine(inverted, M, (w, h),
                              flags=cv2.INTER_CUBIC,
                              borderMode=cv2.BORDER_REPLICATE)

    return deskewed


def adaptive_threshold(img):
    # we don't want to use otsu's because that's primarily if there
    # are two distinct different lightings
    # we should be using adaptive thresholding rather than simple,
    # as our intent is to remove shaoows
    # text suppression

    # clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    # enhanced = clahe.apply(img)
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    img = cv2.medianBlur(img, 5)
    ret, th1 = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
    th2 = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)
    th3 = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                               cv2.THRESH_BINARY, 11, 2)
    titles = ['Original Image', 'Global Thresholding (v = 127)',
              'Adaptive Mean Thresholding', 'Adaptive Gaussian Thresholding']
    images = [img, th1, th2, th3]
    # for i in range(4):
    #     plt.subplot(2, 2, i + 1), plt.imshow(images[i], 'gray')
    #     plt.title(titles[i])
    #     plt.xticks([]), plt.yticks([])
    # plt.show()
    # kernel = np.ones((2, 2), np.uint8)
    # cleaned = cv2.morphologyEx(th3, cv2.MORPH_CLOSE, kernel)
    return th3
    # dilated_img = cv22.dilate(img, np.ones((7, 7), np.uint8))
    # bg_img = cv22.medianBlur(dilated_img, 21)
    #
    # diff_img = 255 - cv22.absdiff(img, bg_img)
    # norm_img = diff_img.copy()  # Needed for 3.x compatibility
    # cv22.normalize(diff_img, norm_img, alpha=0, beta=255, norm_type=cv22.NORM_MINMAX, dtype=cv22.cv2_8UC1)
    # _, thr_img = cv22.threshold(norm_img, 255, 0, cv22.THRESH_TRUNC)
    # cv22.normalize(thr_img, thr_img, alpha=0, beta=255, norm_type=cv22.NORM_MINMAX, dtype=cv22.cv2_8UC1)
    # kernel = np.ones((2, 2), np.uint8)
    # closed = cv22.morphologyEx(thr_img, cv22.MORPH_CLOSE, kernel)
    # return closed


if __name__ == '__main__':
    main()
