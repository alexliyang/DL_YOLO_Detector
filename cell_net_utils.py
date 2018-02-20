import os
import pickle
import random
import xml.etree.ElementTree as ET

import cv2
import numpy as np
import tensorflow as tf
from sklearn.utils import shuffle

from parameters import params

C = 20

# colours = [[1, 0, 0], [0, 0, 1], [0, 1, 0],
#            [0.81, 0.89, 0.25], [0, 0.647, 1],
#            [0.502, 0, 0.502], [0.196, 0.804, 0.196],
#            [0.439, 0.01, 0.01], [0.663, 0.663, 0.663],
#            [0.118, 0.412, 0.824], [0, 0, 0.5],
#            [0.310, 0.310, 0.184], [0, 0.502, 0.502],
#            [1, 0, 1], [0.769, 0.894, 1]]

colours = [[1, 0, 0], [0, 0, 1], [0, 1, 0],
           [0.81, 0.89, 0.25], [0, 0.647, 1],
           [0.502, 0, 0.502], [0.196, 0.804, 0.196],
           [0.439, 0.01, 0.01], [0.663, 0.663, 0.663],
           [0.118, 0.412, 0.824], [0, 0, 0.5],
           [0.310, 0.310, 0.184], [0, 0.502, 0.502],
           [1, 0, 1], [0.769, 0.894, 1],
           [0.118, 0.412, 0.824], [0, 0, 0.5],
           [0.310, 0.310, 0.184], [0, 0.502, 0.502],
           [1, 0, 1], [0.769, 0.894, 1]
           ]

def get_gt_bdboxes(xml_filename, name_converter, classes, dst_img_size):
    tree = ET.parse(xml_filename)
    size = tree.find('size')
    width = int(size.find('width').text)
    height = int(size.find('height').text)
    if height == 0 or width == 0:
        raise Exception
    h_ratio = dst_img_size / height
    w_ratio = dst_img_size / width
    bdboxes=[]

    objs = tree.findall('object')
    for obj in objs:
        bbox = obj.find('bndbox')
        xmin = int(float(bbox.find('xmin').text) * w_ratio)
        xmax = int(float(bbox.find('xmax').text) * w_ratio)
        ymin = int(float(bbox.find('ymin').text) * h_ratio)
        ymax = int(float(bbox.find('ymax').text) * h_ratio)
        class_id = classes.index(name_converter[obj.find('name').text.lower().strip()])
        bdboxes.append([class_id, xmin, ymin, xmax, ymax, 1])
    return bdboxes


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

    t_images_dir = os.path.join(root_folder, 'train_images')
    t_labels_dir = os.path.join(root_folder, 'train_labels')
    v_images_dir = os.path.join(root_folder, 'val_images')
    v_labels_dir = os.path.join(root_folder, 'val_labels')

    if not os.path.isdir(t_images_dir):
        os.mkdir(t_images_dir)
    if not os.path.isdir(t_labels_dir):
        os.mkdir(t_labels_dir)
    if not os.path.isdir(v_images_dir):
        os.mkdir(v_images_dir)
    if not os.path.isdir(v_labels_dir):
        os.mkdir(v_labels_dir)

    # # one time process, don't use it
    # images_filenames = sorted([images_path + name for name in os.listdir(images_path)])
    # xmls_filenames = sorted([xmls_path + name for name in os.listdir(xmls_path)])
    # images_filenames, xmls_filenames = shuffle(images_filenames, xmls_filenames)
    # t_images_filenames = images_filenames[:int(0.9 * len(images_filenames))]
    # t_xmls_filenames = xmls_filenames[:int(0.9 * len(xmls_filenames))]
    # v_images_filenames = images_filenames[int(0.9 * len(images_filenames)):]
    # v_xmls_filenames = xmls_filenames[int(0.9 * len(xmls_filenames)):]
    # pickle.dump([t_images_filenames, t_xmls_filenames, v_images_filenames, v_xmls_filenames], open(os.path.join(root_folder, 'dataset_info.p'), 'wb'))

    t_images_filenames, t_xmls_filenames, v_images_filenames, v_xmls_filenames = pickle.load(
        open(os.path.join(root_folder, 'dataset_info.p'), 'rb'))

    # train data
    for i, (imagename, xmlname) in enumerate(zip(t_images_filenames, t_xmls_filenames)):
        print('\rTraining data: %d of %d' % (i, len(t_images_filenames)), end='', flush=True)
        img = cv2.imread(imagename)
        img = cv2.resize(img, dsize=(img_size, img_size))
        label = xml_as_tensor(xmlname, img_size, name_converter, classes)

        cv2.imwrite(os.path.join(t_images_dir, str(i) + '.jpg'), img)
        np.save(os.path.join(t_labels_dir, str(i) + '.npy'), label)

    # validation data
    for i, (imagename, xmlname) in enumerate(zip(v_images_filenames, v_xmls_filenames)):
        print('\rValidation data: %d of %d' % (i, len(v_images_filenames)), end='', flush=True)
        img = cv2.imread(imagename)
        img = cv2.resize(img, dsize=(img_size, img_size))
        label = xml_as_tensor(xmlname, img_size, name_converter, classes)

        cv2.imwrite(os.path.join(v_images_dir, str(i) + '.jpg'), img)
        np.save(os.path.join(v_labels_dir, str(i) + '.npy'), label)


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


def remove_outliers(thresh_logits, min_num_neighbours=1):
    """
    Removes outlying values, for eg ones that have no neighbours
    :param thresh_logits: thresholded logit
    :param min_num_neighbours: minimum number of neighboiuring pixels
    :return: cleaned logits
    """
    kernel = np.ones((3, 3), np.float32)
    cleaned_logits = cv2.filter2D(thresh_logits, -1, kernel)
    thresh_logits[cleaned_logits <= min_num_neighbours] = 0
    return thresh_logits


def get_bounding_boxes(thresh_logits, logits, S, src_img_size, min_contour_area=2.0):
    """
    Creates list of bounding boxes
    :param logits: real valued logits (needed to compute confidence
    :param thresh_logits: binary logits (0 or 1)
    :param S: number of cells per side
    :param src_img_size: size of image
    :return: list of bounding boxes, each box: [x1, y1, x2, y2, class]
    """
    step = int(src_img_size / S)

    # find contours and appropriate bounding boxes
    bounding_boxes = []
    for class_id in range(thresh_logits.shape[-1]):
        plane = np.copy(thresh_logits[..., class_id:class_id + 1]).astype(np.uint8)
        _, contours, _ = cv2.findContours(plane, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)
                if cv2.contourArea(contour) <= min_contour_area:
                    continue
                confidence = np.sum(logits[y:y + h, x: x + w, class_id]) / np.count_nonzero(
                    logits[y:y + h, x: x + w, class_id])
                bounding_boxes.append([class_id, x * step, y * step, (x + w) * step, (y + h) * step, confidence])
    return bounding_boxes


def draw_bounding_boxes(float_img, bounding_boxes, colours):
    """
    Draws bounding boxes on image
    :param float_img: real valued image
    :param bounding_boxes: list of bounding boxes, each [class, x1, y1, x2, y2, confidence]
    :param colours: list of colours per class
    :return: image with embedded bounding boxes
    """
    for bdbox in bounding_boxes:
        text_size, _ = cv2.getTextSize('%s %.2f' % (params.classes[bdbox[0]], bdbox[-1]), cv2.FONT_HERSHEY_COMPLEX, 0.5, 1)
        float_img = cv2.rectangle(float_img, (bdbox[1], bdbox[2]), (bdbox[1] + text_size[0] + 5, bdbox[2] + 20),
                                  colours[bdbox[0]], thickness=-1)
        float_img = cv2.rectangle(float_img, (bdbox[1], bdbox[2]), (bdbox[3], bdbox[4]), colours[bdbox[0]], thickness=2)
        float_img = cv2.putText(float_img, '%s %.2f' % (params.classes[bdbox[0]], bdbox[-1]), (bdbox[1] + 5, bdbox[2] + 15),
                                cv2.FONT_HERSHEY_COMPLEX, 0.5,
                                (0, 0, 0) if np.sum(colours[bdbox[0]]) > 1 else (1, 1, 1), thickness=1)
    return float_img

def draw_predicted_cells(float_img, thresh_logits, S, src_img_size):
    step = int(src_img_size / S)
    overlay = np.max(thresh_logits, axis=2)
    output = np.ones_like(float_img) * 0.5
    for y in range(S):
        for x in range(S):
            x_s = int(x * src_img_size / S)
            x_e = int((x + 1) * src_img_size / S)
            y_s = int(y * src_img_size / S)
            y_e = int((y + 1) * src_img_size / S)
            output[y_s: y_e, x_s: x_e] *= overlay[y, x]

    output = cv2.addWeighted(float_img, 1, output, 0.4, 0)

    # {0: 'axe', 1: 'bottle', 2: 'broom', 3: 'button', 4: 'driller', 5: 'hammer', 6: 'light_bulb', 7: 'nail', 8: 'pliers',
    #  9: 'scissors', 10: 'screw', 11: 'screwdriver', 12: 'tape', 13: 'vial', 14: 'wrench'}

    for y in range(S):
        for x in range(S):
            for c in range(thresh_logits.shape[-1]):
                if thresh_logits[y, x, c] == 1:
                    output = cv2.putText(output, str(c), (x * step + 10, y * step + 20), cv2.FONT_HERSHEY_COMPLEX, 0.6,
                                         (0, 0, 0), 1)
    return output


def possibly_create_dirs(embedded_images_path, model_to_save_path):
    if not os.path.isdir(embedded_images_path):
        os.mkdir(embedded_images_path)
    if not os.path.isdir(model_to_save_path):
        os.mkdir(model_to_save_path)


def augment_rotate(image, label):
    """
    Rotates both image and label k times 90 degress
    :param label: original label (not resized
    :return:
    """
    k = random.randint(0, 4)
    return np.rot90(image, k), np.rot90(label, k)


def augment_crop(image, label, k):
    """
    Crops image and resizes it, remains label intact
    :param label: original label (not resized
    :param k: maximum cropping range from each side
    :return:
    """
    h, w = image.shape[:2]
    return cv2.resize(
        image[random.randint(0, k): h - random.randint(0, k), random.randint(0, k): w - random.randint(0, k)],
        (h, w)), label


def augment_translate(image, label, k):
    """
    Rolls both image and label k pixels maximum in each direction (but only if it would not cause to roll bounding box)
    :param label: original label (not resized
    :param k: maximum roll range
    :return:
    """
    flat_label = np.max(label, axis=2)
    non_zero_ys, non_zero_xs = np.nonzero(flat_label)
    y_low_margin = np.min(non_zero_ys)
    y_high_margin = image.shape[0] - np.max(non_zero_ys)
    x_low_margin = np.min(non_zero_xs)
    x_high_margin = image.shape[1] - np.max(non_zero_xs)
    d_y = np.min([k, y_low_margin, y_high_margin])
    d_x = np.min([k, x_low_margin, x_high_margin])
    vertical_shift = random.randint(-d_y, d_y)
    horizontal_shift = random.randint(-d_x, d_x)

    image = np.roll(image, axis=0, shift=vertical_shift)
    image = np.roll(image, axis=1, shift=horizontal_shift)
    label = np.roll(label, axis=0, shift=vertical_shift)
    label = np.roll(label, axis=1, shift=horizontal_shift)
    return image, label


def augment_stack(images, labels):
    """
    Stacks four images into one, then resizes both image and label
    :return:
    """
    images, labels = shuffle(images, labels)

    images_upper_stack = np.hstack([images[0], images[1]])
    images_lower_stack = np.hstack([images[2], images[3]])
    image_stack = np.vstack([images_upper_stack, images_lower_stack])

    labels_upper_stack = np.hstack([labels[0], labels[1]])
    labels_lower_stack = np.hstack([labels[2], labels[3]])
    label_stack = np.vstack([labels_upper_stack, labels_lower_stack])

    # resize label to single image size
    label_stack = resize_label(label_stack, images[0].shape[0], labels[0].shape[-1], image_stack.shape[0], 0.5)
    image_stack = cv2.resize(image_stack, (images[0].shape[0], images[0].shape[0]))
    return image_stack, label_stack


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def create_record_writer(writer_path, record_name):
    """
    Creates tfrecord writer in writer_path and assigns a name to it
    """
    writer = tf.python_io.TFRecordWriter(os.path.join(writer_path, record_name + '.tfrecord'))
    return writer


def create_augmented_tf_records(augmentation_number, image_names, label_names, data_folder, record_size, S,
                                threshold_area):
    """
    Creates tf records of size record_size (with augmentation) and puts them into data folder
    """
    if not os.path.isdir(data_folder):
        os.mkdir(data_folder)

    beginning_index = len(os.listdir(data_folder))
    image_names, label_names = shuffle(image_names, label_names)

    # for stacking
    images_buffer = []
    labels_buffer = []

    tf_records_count = 0
    writer = create_record_writer(data_folder, str(beginning_index))

    for i, (image_name, label_name) in enumerate(zip(image_names[:500], label_names[:500])):
        print("\rAugmentation %d, generating train TFRecords (%.2f)" % (augmentation_number, i / len(image_names)),
              end='', flush=True)
        image = image_read(image_name)
        label = np.load(label_name)

        # augmentation pipeline
        image, label = augment_rotate(image, label)
        image, label = augment_crop(image, label, 10)
        image, label = augment_translate(image, label, 100)
        image = image.astype(np.float32)

        # probability of adding to buffer
        if np.random.uniform(0, 1) < 0.2:
            images_buffer.append(image)
            labels_buffer.append(label)

        label = resize_label(label, S, 20, params.img_size, threshold_area)
        feature = {'train/label': _bytes_feature(tf.compat.as_bytes(label.tostring())),
                   'train/image': _bytes_feature(tf.compat.as_bytes(image.tostring()))}
        example = tf.train.Example(features=tf.train.Features(feature=feature))
        writer.write(example.SerializeToString())

        if len(images_buffer) == 4:
            image, label = augment_stack(images_buffer, labels_buffer)
            label = resize_label(label, S, C, params.img_size, threshold_area)
            feature = {'train/label': _bytes_feature(tf.compat.as_bytes(label.tostring())),
                       'train/image': _bytes_feature(tf.compat.as_bytes(image.tostring()))}
            example = tf.train.Example(features=tf.train.Features(feature=feature))
            writer.write(example.SerializeToString())
            images_buffer = []
            labels_buffer = []

        if i % record_size == 0 and i > 0:
            writer.close()
            tf_records_count += 1
            writer = create_record_writer(data_folder, str(beginning_index + tf_records_count))
    writer.close()
    print()


def decode_train_data(data_path, S, batch_size, capacity, num_threads, min_after_deque):
    """
    Decodes detection tf records on the fly.
    capacity, num_threads, min_after_deque - for shuffle batch
    :param batch_size:
    :param mode: 'train' or 'validation'
    :return:
    """
    feature = {'train/image': tf.FixedLenFeature([], tf.string),
               'train/label': tf.FixedLenFeature([], tf.string)}

    filenames = [os.path.join(data_path, name) for name in os.listdir(data_path)]
    print(filenames)
    filename_queue = tf.train.string_input_producer(filenames)
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(serialized_example, features=feature)
    image = tf.reshape(tf.decode_raw(features['train/image'], tf.float32), [params.img_size, params.img_size, 3])
    label = tf.reshape(tf.decode_raw(features['train/label'], tf.float32), [S, S, C])

    images, labels = tf.train.shuffle_batch([image, label],
                                            batch_size=batch_size,
                                            capacity=capacity,
                                            num_threads=num_threads,
                                            min_after_dequeue=min_after_deque,
                                            allow_smaller_final_batch=True)
    return images, labels
