import os

import cv2
import numpy as np

from cell_net_utils import xml_as_tensor, create_augmented_tf_records
from parameters import params

images_path = 'VOCdevkit 2012/VOC2012/JPEGImages/'
xmls_path = 'VOCdevkit 2012/VOC2012/Annotations/'
root_folder = 'VOCdevkit 2012/VOC2012/'
t_images_path = 'VOCdevkit 2012/VOC2012/train_images/'
t_labels_path = 'VOCdevkit 2012/VOC2012/train_labels/'
train_records_path = 'VOCdevkit 2012/VOC2012/tfrecords/'

S = 14
threshold_area = int(params.img_size / S) ** 2 / 2

image_filenames = sorted([images_path + name for name in os.listdir(images_path)])
xmls_filenames = sorted([xmls_path + name for name in os.listdir(xmls_path)])

classes = ['aeroplane', 'bicycle', 'bird', 'boat',
           'bottle', 'bus', 'car', 'cat', 'chair',
           'cow', 'diningtable', 'dog', 'horse',
           'motorbike', 'person', 'pottedplant',
           'sheep', 'sofa', 'train', 'tvmonitor']

name_converter = dict(zip(classes, classes))


def generate_cell_net_data(root_folder, image_filenames, xmls_filenames, img_size, name_converter, classes):
    t_images_dir = os.path.join(root_folder, 'train_images')
    t_labels_dir = os.path.join(root_folder, 'train_labels')

    if not os.path.isdir(t_images_dir):
        os.mkdir(t_images_dir)
    if not os.path.isdir(t_labels_dir):
        os.mkdir(t_labels_dir)

    # train data
    for i, (imagename, xmlname) in enumerate(zip(image_filenames, xmls_filenames)):
        print('\rTraining data: %d of %d' % (i, len(image_filenames)), end='', flush=True)
        img = cv2.imread(imagename)
        img = cv2.resize(img, dsize=(img_size, img_size))
        label = xml_as_tensor(xmlname, img_size, name_converter, classes)

        cv2.imwrite(os.path.join(t_images_dir, str(i) + '.jpg'), img)
        np.save(os.path.join(t_labels_dir, str(i) + '.npy'), label)


# generate_cell_net_data(root_folder, image_filenames, xmls_filenames, params.img_size, name_converter, classes)

# augmentations = 1  # number of dataset augmentations
# train_image_filenames = sorted([t_images_path + name for name in os.listdir(t_images_path)])
# train_labels_filenames = sorted([t_labels_path + name for name in os.listdir(t_labels_path)])
# for i in range(augmentations):
#     create_augmented_tf_records(i, train_image_filenames, train_labels_filenames, train_records_path, 10, S,
#                                 threshold_area)
