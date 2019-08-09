import os
import random
from typing import AnyStr, Any

import tensorflow as tf
from tensorflow.contrib import slim


def str2bool(x: AnyStr) -> bool:
    return x.lower() in ('true')


def augmentation(image: Any, augment_size: Any) -> Any:
    seed = random.randint(0, 2 ** 31 - 1)
    image_shape_orig = tf.shape(image)
    image = tf.image.random_flip_left_right(image, seed=seed)
    image = tf.image.resize_images(image, [augment_size, augment_size])
    image = tf.random_crop(image, image_shape_orig, seed=seed)

    return image


class ImageData:
    def __init__(self, load_size, channels, augment_flag):
        self.load_size = load_size
        self.channels = channels
        self.augment_flag = augment_flag

    def image_processing(self, filename) -> Any:
        x = tf.read_file(filename=filename)
        x_decode = tf.image.decode_jpeg(x, channels=self.channels)
        img = tf.image.resize_images(x_decode, [self.load_size, self.load_size])
        img = tf.cast(img, tf.float32) / 127.5 - 1

        if self.augment_flag:
            augment_size = self.load_size + (30 if self.load_size == 256 else 15)
            p = random.random()
            if p > 0.5:
                img = augmentation(img, augment_size)

        return img


def check_folder(directory: str) -> str:
    if not os.path.exists(directory):
        os.makedirs(directory)
    return directory


def show_all_variables():
    model_vars = tf.trainable_variables()
    slim.model_analyzer.analyze_vars(model_vars, print_info=True)
