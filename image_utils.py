from PIL import Image, ImageOps
from os import listdir
from os.path import isfile, join, isdir


def get_top_level_dir(path):
    return [name for name in listdir(path) if isdir(join(path, name))]

def walk_dir(path):
    return [f for f in listdir(path) if isfile(join(path, f))]

def rescale_image(raw_image, size):
    image = Image.open(raw_image)
    raw_width, raw_height = image.size
    image = image.resize((raw_width//3,raw_height//3), resample = Image.BILINEAR)

    width, height = image.size
    delta_w = size - width
    delta_h = size - height
    padding = (delta_w//2, delta_h//2, delta_w-(delta_w//2), delta_h-(delta_h//2))
    new_im = ImageOps.expand(image, padding)

    return new_im

def load_data():
    dir_path = 'test-set'
    image_dir = get_top_level_dir(dir_path)
    images = []
    for dir in image_dir:
        file_path = dir_path + '/' + dir + '/images/'
        files = walk_dir(file_path)
        for file in files:
            image_path = file_path + file
            rescaled_image = rescale_image(image_path, 256)
            images.append(rescaled_image)
    print(images[0].show())

if __name__ == '__main__':
    load_data()