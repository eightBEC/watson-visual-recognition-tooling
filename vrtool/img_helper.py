# coding: utf-8

# Copyright 2018-2020 Jan Forster
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
import cv2
import math
from os.path import splitext

'''
Provides helper functions for image duplication
'''
# Rotate the image without cropping


def rotate_img_full(img, angle):
    width = img.shape[1]
    height = img.shape[0]
    rangle = np.deg2rad(angle)

    # calculate the size of the container
    container_width = (height*abs(np.sin(rangle)) + abs(width*np.cos(rangle)))
    container_height = (height*abs(np.cos(rangle)) + abs(width*np.sin(rangle)))
    container_center = (container_width*0.5, container_height*0.5)

    # obtain rotation matrix
    R = cv2.getRotationMatrix2D(container_center, angle, 1.0)

    # calculate translation so that full image is contained in container
    # shift to outer_center: x' = x + t
    # rotate image: x'' = Rx' = Rx + Rt

    t = np.array([(container_width-width)*0.5, (container_height-height)*0.5])
    Rt = np.dot(R[:, :2], t)

    # We only have to add Rt to the translation part of the matrix R to obtain the final Euclidean transformation
    R[:, 2] += Rt

    return cv2.warpAffine(img, R, (int(math.ceil(container_width)), int(math.ceil(container_height))), flags=cv2.INTER_LANCZOS4)

# Crop image according to (width, height)


def crop_img(img, width, height):
    img_w = img.shape[1]
    img_h = img.shape[0]

    if(width > img_w):
        width = img_w

    if(height > img_h):
        height = img_h

    w1 = int((img_w - width) * 0.5)
    w2 = int((img_w + width) * 0.5)
    h1 = int((img_h - height) * 0.5)
    h2 = int((img_h + height) * 0.5)

    return img[h1:h2, w1:w2]

# Finds the width & height of the maximal rectangle in the rotated image


def largest_rotated_rect(width, height, angle):
    rangle = np.deg2rad(angle)
    quadrant = int(math.floor(rangle / (math.pi / 2))) & 3
    sign_alpha = rangle if ((quadrant & 1) == 0) else math.pi - rangle
    alpha = (sign_alpha % math.pi + math.pi) % math.pi

    bb_w = width * math.cos(alpha) + height * math.sin(alpha)
    bb_h = width * math.sin(alpha) + height * math.cos(alpha)

    gamma = math.atan2(bb_w, bb_w) if (
        width < height) else math.atan2(bb_w, bb_w)

    delta = math.pi - alpha - gamma

    length = height if (width < height) else width

    d = length * math.cos(alpha)
    a = d * math.sin(alpha) / math.sin(delta)

    y = a * math.cos(gamma)
    x = y * math.tan(gamma)

    return bb_w - 2 * x, bb_h - 2 * y

# Duplicates image

def _rotate_img(img, angle, base, extension):
    rot_img = rotate_img_full(img, angle)
    new_width, new_height = largest_rotated_rect(
        img.shape[1], img.shape[0], -angle)
    crop_rot_img = crop_img(rot_img, new_width, new_height)
    return crop_rot_img

def rotate_left(img, angle, base, extension):
    rotation_result = _rotate_img(img, -angle, base, extension)

    dir_rot_ccw_img = base + "_rot_ccw" + \
        str(angle).replace(".", "_") + extension

    cv2.imwrite(dir_rot_ccw_img, rotation_result)
   

def rotate_right(img, angle, base, extension):
    rotation_result = _rotate_img(img, angle, base, extension)

    dir_rot_ccw_img = base + "_rot_cw" + \
        str(angle).replace(".", "_") + extension

    cv2.imwrite(dir_rot_ccw_img, rotation_result)

def flip_image(img, base, extension):
    mirror_img = cv2.flip(img, 1)
    dir_mirror_img = base + "_mirror" + extension
    cv2.imwrite(dir_mirror_img, mirror_img)
  

# Duplicates all images in 'class_images'
def duplicate_class(class_images, rotation_angle):
    for image in class_images:
        base, extension = splitext(image)
        img = cv2.imread(image)
        if isinstance(rotation_angle, list):
            for angle in rotation_angle:
                rotate_left(img, angle, base, extension)
                rotate_right(img, angle, base, extension)
        else:
            rotate_left(img, angle, base, extension)
            rotate_right(img, angle, base, extension)
            
        flip_image(img, base, extension)