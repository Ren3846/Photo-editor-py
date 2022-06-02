import random

import PIL.Image
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple
from beetle_algo import get_image_segments
from knn_filter import knn_filter

IMAGE_PATH = '/Users/ren3846/Downloads/LAB_21/image/2022-03-04 23.43.00.jpg'
RGB_MAX_VALUE = 255


def brightness(rgb: np.ndarray) -> int:
    return 0.3 * rgb[0] + 0.59 * rgb[1] + 0.11 * rgb[2]


def task_1(image_obj: PIL.Image):
    image_obj.show()


def task_2(brightness_array: np.ndarray):
    print("Матриця яскравості")
    print(brightness_array)


def task_3(image_array: np.ndarray, brightness_array: np.ndarray):
    figure, axs = plt.subplots(4, figsize=(13, 13))
    axs[0].hist(image_array[:, :, 0].flatten(), color="red", rwidth=0.9)
    axs[0].set_title("Червоний колір")
    axs[1].hist(image_array[:, :, 1].flatten(), color="green", rwidth=0.9)
    axs[1].set_title("Зелений колір")
    axs[2].hist(image_array[:, :, 2].flatten(), color="blue", rwidth=0.9)
    axs[2].set_title("Синій колір")
    axs[3].hist(brightness_array.flatten(), color="gray", rwidth=0.9)
    axs[3].set_title("Градація сірого")
    figure.show()


def show_grey_image(brightness_array: np.ndarray):
    grey_array = np.zeros(shape=(brightness_array.shape[0], brightness_array.shape[1], 3))
    grey_array[:, :, 0] = brightness_array
    grey_array[:, :, 1] = brightness_array
    grey_array[:, :, 2] = brightness_array

    grey_image = Image.fromarray(grey_array.astype('uint8'), 'RGB')
    grey_image.show()


def get_black_white_color(rgb: tuple) -> int:
    return 255 if sum(rgb) > RGB_MAX_VALUE * 3 / 2 else 0


def get_black_white_image(image_array: np.ndarray) -> np.ndarray:
    black_white_array = np.zeros(shape=(image_array.shape[0], image_array.shape[1], 3))
    black_white_values = np.apply_along_axis(get_black_white_color, 2, image_array)
    black_white_array[:, :, 0] = black_white_values
    black_white_array[:, :, 1] = black_white_values
    black_white_array[:, :, 2] = black_white_values

    return black_white_array


def task_4(image_array: np.ndarray, brightness_array: np.ndarray, black_white_array: np.ndarray):
    show_grey_image(brightness_array)

    black_white_image = Image.fromarray(black_white_array.astype('uint8'), 'RGB')
    black_white_image.show()

    show_negative_image(image_array)


def show_negative_image(image_array: np.ndarray):
    negative_image_array = RGB_MAX_VALUE - image_array

    negative_image = Image.fromarray(negative_image_array.astype('uint8'), 'RGB')
    negative_image.show()


def show_marked_pixels_on_image(marked_segments: List[List[Tuple[int, int]]], image_array: np.ndarray):
    image_array_with_path = image_array.copy()
    for segment in marked_segments:
        color = [random.choice([0, RGB_MAX_VALUE]), random.choice([0, RGB_MAX_VALUE]),
                 random.choice([0, RGB_MAX_VALUE])]
        for coordinates in segment:
            image_array_with_path[coordinates[0], coordinates[1]] = color

    image_with_path = Image.fromarray(image_array_with_path.astype('uint8'), 'RGB')
    image_with_path.show()


def task_5_beetle_algorythm(black_white_array: np.ndarray):
    marked_pixels_list = get_image_segments(black_white_array)
    show_marked_pixels_on_image(marked_pixels_list, image_array=image_matrix)


def task_5_knn_filter(brightness_array: np.ndarray):
    brightness_matrix_filtered = knn_filter(brightness_array)
    show_grey_image(brightness_matrix_filtered)


if __name__ == '__main__':
    image = Image.open(IMAGE_PATH)
    image_matrix = np.array(image)
    brightness_matrix = np.apply_along_axis(brightness, 2, image_matrix)
    black_white_matrix = get_black_white_image(image_matrix)

    #task_1(image)
    #task_2(brightness_matrix)
    #task_3(image_matrix, brightness_matrix)
    #task_4(image_matrix, brightness_matrix, black_white_matrix)
    #task_5_beetle_algorythm(black_white_matrix)
    #task_5_knn_filter(brightness_matrix)

    plt.show()
