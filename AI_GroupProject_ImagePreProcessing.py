# -*- coding: utf-8 -*-
"""
Created on Sun Mar 17 22:13:16 2019

@author: Christian
"""
import PIL
from PIL import Image, ImageOps
import numpy as np

# Load image and downsample to 1/3 of the original size
image = Image.open('C:\\Users\\Christian\\Documents\\imagenet-example\\images\\brown_bear.png')
width, height = image.size
image = image.resize((width//3,height//3), resample = PIL.Image.BILINEAR)

# Get dimensions
width, height = image.size
image.size

# Pad image with all black border
new_size = 250
delta_w = new_size - width
delta_h = new_size - height
padding = (delta_w//2, delta_h//2, delta_w-(delta_w//2), delta_h-(delta_h//2))
new_im = ImageOps.expand(image, padding)

# Show new image for verification
new_im.show()
new_im.size


# crop the borders and only return the center square of the image
# may not need this function 
#left = (width - 200)/2
#top = (height - 200)/2
#right = (width + 200)/2
#bottom = (height + 200)/2
#
#image = image.crop((left, top, right, bottom))
#image.size

