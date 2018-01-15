import cv2
import numpy as np
from random import randint
import os

width = 512
height = 512
min_side = min_radius = 15
max_side = max_radius = 40
max_num_objects = 5

def draw_circle(img):
    img = cv2.circle(img, center=(randint(max_radius, height-max_radius), randint(max_radius, width-max_radius)),
           radius=randint(min_radius, max_radius),
           color=(randint(0, 255),randint(0, 255),randint(0, 255)), thickness=-1)
    return img

def draw_square(img):
    pt1_y = randint(max_side, height - max_side)
    pt1_x = randint(max_side, width - max_side)
    side = randint(min_side, max_side)
    img = cv2.rectangle(img, pt1=(pt1_x, pt1_y), pt2=(pt1_x+side, pt1_y+side),
           color=(randint(0, 255),randint(0, 255),randint(0, 255)), thickness=-1)
    return img

def draw_up_rect(img):
    pt1_y = randint(max_side, height - max_side)
    pt1_x = randint(max_side, width - max_side)
    img = cv2.rectangle(img, pt1=(pt1_x, pt1_y), pt2=(pt1_x+min_side, pt1_y+max_side),
           color=(randint(0, 255),randint(0, 255),randint(0, 255)), thickness=-1)
    return img

def draw_side_rect(img):
    pt1_y = randint(max_side, height - max_side)
    pt1_x = randint(max_side, width - max_side)
    img = cv2.rectangle(img, pt1=(pt1_x, pt1_y), pt2=(pt1_x+max_side, pt1_y+min_side),
           color=(randint(0, 255),randint(0, 255),randint(0, 255)), thickness=-1)
    return img


def generate_image(max_num_objects, functions):
    img = np.zeros((height, width, 3), np.uint8)
    for object in range(randint(0, max_num_objects)):
        cls = randint(0,3)
        img = functions[cls](img)
    return img


functions = [draw_circle, draw_square, draw_side_rect, draw_up_rect]

path = 'synthetic2'

if not os.path.isdir(path):
    os.mkdir(path)
for i in range(100):
    img = generate_image(max_num_objects, functions)
    cv2.imwrite(path + '/' + str(i) + '.png', img)
