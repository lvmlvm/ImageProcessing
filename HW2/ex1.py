import cv2
import matplotlib.pyplot as plt
import numpy as np
import math
import os


def read_img(img_path):
    """
        Read grayscale image
        Inputs:
        img_path: str: image path
        Returns:
        img: cv2 image
    """
    return cv2.imread(img_path, 0)


def padding_img(img, filter_size=3):
    """
        Conveniently, np has a padding functions np.pad() that we can utilize immediately.
        The "edge" mode essentially works as Replicate padding.
    """
    padding = int(filter_size / 2)
    return np.pad(img, (padding, padding), mode='edge')

def mean_filter(img, filter_size=3):
    # Retrieves the image and filter dimensions
    height, width = img.shape[:2]
    padding = int(filter_size / 2)

    # Intermediate variables for modifying the image
    smooth_img = np.zeros((height, width), dtype=np.uint8)
    pad_img = padding_img(img, filter_size)

    # The mean filter itself
    for i in range(padding, height - padding):
        for j in range(padding, width - padding):
            block = pad_img[i - padding : i + padding + 1, j - padding : j + padding + 1]
            mean = np.mean(block)
            smooth_img[i - padding, j - padding] = int(mean)

    return smooth_img



def median_filter(img, filter_size=3):
    height, width = img.shape[:2]
    smooth_img = np.zeros((height, width), dtype=np.uint8)

    padding = int(filter_size / 2)
    pad_img = padding_img(img, filter_size)

    for i in range(padding, height - padding):
        for j in range(padding, width - padding):
            block = pad_img[i - padding : i + padding + 1, j - padding : j + padding + 1]
            mean = np.median(block)
            smooth_img[i - padding, j - padding] = int(mean)

    return smooth_img


def psnr(gt_img, smooth_img):
    gt_img = gt_img.astype(np.float32)
    smooth_img = smooth_img.astype(np.float32)

    print(gt_img.shape, smooth_img.shape)

    mse = np.mean(np.square(gt_img - smooth_img))
    max_val = 255.0

    psnr = 10 * np.log10((max_val ** 2) / mse)

    return psnr


def show_res(before_img, after_img):
    """
        Show the original image and the corresponding smooth image
        Inputs:
            before_img: cv2: image before smoothing
            after_img: cv2: corresponding smoothed image
        Return:
            None
    """
    plt.figure(figsize=(12, 9))
    plt.subplot(1, 2, 1)
    plt.imshow(before_img, cmap='gray')
    plt.title('Before')

    plt.subplot(1, 2, 2)
    plt.imshow(after_img, cmap='gray')
    plt.title('After')
    plt.show()


if __name__ == '__main__':
    img_noise = "ex1_images/noise.png" # <- need to specify the path to the noise image
    img_gt = "" # <- need to specify the path to the gt image
    img = read_img(img_noise)
    filter_size = 3

    # Mean filter
    mean_smoothed_img = mean_filter(img, filter_size)
    show_res(img, mean_smoothed_img)
    print('PSNR score of mean filter: ', psnr(img, mean_smoothed_img))

    # Median filter
    median_smoothed_img = median_filter(img, filter_size)
    show_res(img, median_smoothed_img)
    print('PSNR score of median filter: ', psnr(img, median_smoothed_img))

