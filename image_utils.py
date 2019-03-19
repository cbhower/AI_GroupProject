from PIL import Image, ImageOps
from os import listdir
from os.path import isfile, join, isdir
import numpy as np


def get_top_level_dir(path):
    return [name for name in listdir(path) if isdir(join(path, name))]


def walk_dir(path):
    return [f for f in listdir(path) if isfile(join(path, f))]


def rescale_image(raw_image, new_width, new_height, resize_mode=Image.ANTIALIAS):
    image = Image.open(raw_image)
    img = image.resize((new_width, new_height), resize_mode)
    return img


def load_data():
    dir_path = 'test-set'
    image_dir = get_top_level_dir(dir_path)
    images = []
    labels = []
    i = 0
    for dir in image_dir:
        label_buf = [0, 0, 0]
        file_path = dir_path + '/' + dir + '/images/'
        files = walk_dir(file_path)
        for file in files:
            label_buf[i] = 1
            image_path = file_path + file
            rescaled_image = rescale_image(image_path, 224, 224)
            #rescaled_image = convert_color(rescaled_image, 'L')
            buf = pil_to_nparray(rescaled_image)
            buf /= 255.
            images.append(buf)
            labels.append(label_buf)
        i += 1

    return images, labels


def convert_color(pil_image, mode):
    return pil_image.convert(mode)


def pil_to_nparray(pil_image):
    pil_image.load()
    return np.asarray(pil_image, dtype='float32')


if __name__ == '__main__':
    load_data()