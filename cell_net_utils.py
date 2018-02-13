import numpy as np
import xml.etree.ElementTree as ET
import cv2
import os
def xml_as_tensor(xml_path, dst_img_size, name_converter, classes):
    """
    Returns presence tensor [img-size, img_size, C] encoded as one hot, where objects are present
    """
    tree = ET.parse(xml_path)
    size = tree.find('size')
    width = int(size.find('width').text)
    height = int(size.find('height').text)
    if height == 0 or width == 0:
        raise Exception

    h_ratio = dst_img_size / height
    w_ratio = dst_img_size / width

    label = np.zeros(shape=[dst_img_size, dst_img_size, len(classes)], dtype=np.float32)
    objs = tree.findall('object')

    for obj in objs:
        bbox = obj.find('bndbox')
        xmin = int(float(bbox.find('xmin').text) * w_ratio)
        xmax = int(float(bbox.find('xmax').text) * w_ratio)
        ymin = int(float(bbox.find('ymin').text) * h_ratio)
        ymax = int(float(bbox.find('ymax').text) * h_ratio)
        class_index = classes.index(name_converter[obj.find('name').text.lower().strip()])
        label[ymin: ymax, xmin: xmax, class_index] = 1

    return label

def generate_cell_net_data(root_folder, img_size, name_converter, classes):
    images_path = 'data/imagenet/detection_images/'
    xmls_path = 'data/imagenet/detection_annotations/'

    images_filenames = sorted([images_path + name for name in os.listdir(images_path)])
    xmls_filenames = sorted([xmls_path + name for name in os.listdir(xmls_path)])

    images_dir = os.path.join(root_folder, 'images')
    labels_dir = os.path.join(root_folder, 'labels')

    if not os.path.isdir(root_folder):
        os.mkdir(root_folder)
    # subfolders
    if not os.path.isdir(images_dir):
        os.mkdir(images_dir)
    if not os.path.isdir(labels_dir):
        os.mkdir(labels_dir)

    for i, (imagename, xmlname) in enumerate(zip(images_filenames, xmls_filenames)):
        print(i, 'of', len(images_filenames))
        img = cv2.imread(imagename)
        img = cv2.resize(img, dsize=(img_size, img_size))
        label = xml_as_tensor(xmlname, img_size, name_converter, classes)

        cv2.imwrite(os.path.join(images_dir, str(i) + '.jpg'), img)
        np.save(os.path.join(labels_dir, str(i) + '.npy'), label)

def resize_label(label, S, C, src_img_size, threshold_area):
    resized_label = np.zeros([S, S, C], dtype=np.float32)
    for y in range(S):
        for x in range(S):
            x_s = int(x * src_img_size / S)
            x_e = int((x + 1) * src_img_size / S)
            y_s = int(y * src_img_size / S)
            y_e = int((y + 1) * src_img_size / S)
            column = label[y_s: y_e, x_s: x_e]
            sums = np.sum(np.sum(column, axis=0), axis=0)
            sums[sums < threshold_area] = 0
            sums[sums >= threshold_area] = 1
            resized_label[y, x] = sums
    return resized_label

def image_read(imgname):
    image = cv2.imread(imgname)
    image = (image / 255.0) * 2.0 - 1.0
    return image

def embed_output(float_img, logits, threshold, S, src_img_size):
    logits[logits >= threshold] = 1
    logits[logits < threshold] = 0
    # tymczasowo scalam wszystko do jednego, dla celow debugowania
    overlay = np.max(logits, axis = 2)
    output = np.ones_like(float_img)[..., 0]
    for y in range(S):
        for x in range(S):
            x_s = int(x * src_img_size / S)
            x_e = int((x + 1) * src_img_size / S)
            y_s = int(y * src_img_size / S)
            y_e = int((y + 1) * src_img_size / S)
            output[y_s: y_e, x_s: x_e] *= overlay[y, x]
    output = np.stack([np.zeros_like(output), output, np.zeros_like(output)], axis=2)
    return cv2.addWeighted(float_img, 0.6, output, 0.4, 0)