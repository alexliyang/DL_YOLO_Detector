import cv2
import numpy as np
import pickle
import tensorflow as tf
import os

import time

from mAP_utils import get_info, compute_mAP_recall_precision
from architecture.convolution import conv_model
from cell_net_utils import get_bounding_boxes, remove_outliers, draw_bounding_boxes, colours, draw_predicted_cells, \
    get_gt_bdboxes
from parameters import params



# params
S = 14
threshold_area = int(params.img_size / S) ** 2 / 2
# paths
pretrained_model_path = 'models/cell_network_15_02_eta0_00001_adam_50epochs_batch10'

# image_names, xml_names = pickle.load(open('cell_data/custom_dataset_info.p', 'rb'))
_, _, image_names, xml_names = pickle.load(open('cell_data/dataset_info.p', 'rb'))
dupa = [name for name in image_names if 'wrench' in name]
print(len(dupa))

images_placeholder = tf.placeholder(dtype=tf.float32, shape=[None, params.img_size, params.img_size, 3])
conv = conv_model(images_placeholder)
output = tf.layers.conv2d(conv, params.C, kernel_size=[1, 1],
                          kernel_initializer=tf.truncated_normal_initializer(0.0, 0.1))
sigmoid_output = tf.nn.sigmoid(output)

with tf.Session() as sess:
    init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
    sess.run(init_op)
    saver = tf.train.Saver()

    saver.restore(sess, pretrained_model_path + '/model.ckpt')
    print(pretrained_model_path, ' model loaded')

    times = []
    mAPs = []
    m_precision = []
    m_recall = []

    per_class_AP = dict((k, []) for k in range(params.C))
    per_class_recall = dict((k, []) for k in range(params.C))
    per_class_precision = dict((k, []) for k in range(params.C))

    for img_name, xml_name in zip(image_names, xml_names):
        img = cv2.imread(img_name)
        img = cv2.resize(img, (params.img_size, params.img_size))
        img = (img / 255.0) * 2.0 - 1.0

        # get gt bdboxes
        gt_bounding_boxes = get_gt_bdboxes(xml_name, params.name_converter, params.classes, params.img_size)

        # predict bdboxes
        s = time.time()
        logits = sess.run(sigmoid_output, feed_dict={images_placeholder: [img]})
        logits = logits[0]
        thres_logits = np.copy(logits)
        thres_logits[thres_logits >= 0.5] = 1
        thres_logits[thres_logits < 0.5] = 0
        thres_logits = remove_outliers(thres_logits, 1)
        pred_bounding_boxes = get_bounding_boxes(thres_logits, logits, S, params.img_size, min_contour_area=0)
        e = time.time()
        times.append(e-s)

        AP, recall, precision, mean_AP, mean_recall, mean_precision = compute_mAP_recall_precision(gt_bounding_boxes, pred_bounding_boxes, params.C)

        # for gt_box in gt_bounding_boxes:
        #     img = cv2.rectangle(img, (gt_box[1], gt_box[2]), (gt_box[3], gt_box[4]), color=(0, 0, 1), thickness=2)
        # for pred_box in pred_bounding_boxes:
        #     img = cv2.rectangle(img, (pred_box[1], pred_box[2]), (pred_box[3], pred_box[4]), color=(0, 1, 0),
        #                         thickness=2)
        # cv2.imshow('', (img + 1) / 2)
        # cv2.waitKey(2000)

        for key, value in AP.items():
            if value is not None:
                per_class_AP[key].append(value)
        for key, value in recall.items():
            if value is not None:
                per_class_recall[key].append(value)
        for key, value in precision.items():
            if value is not None:
                per_class_precision[key].append(value)

        mAPs.append(mean_AP)
        m_precision.append(mean_precision)
        m_recall.append(mean_recall)


    for key, value in per_class_AP.items():
        per_class_AP[key] = np.mean(value)
    for key, value in per_class_precision.items():
        per_class_precision[key] = np.mean(value)
    for key, value in per_class_recall.items():
        per_class_recall[key] = np.mean(value)

    print('per class ap', per_class_AP)
    print('per class recall', per_class_recall)
    print('per class precision' ,per_class_precision)

    print('mAP', np.mean(mAPs))
    print('m_precision', np.mean(m_precision))
    print('m_recall', np.mean(m_recall))
    print('time', np.mean(times))

# {0: 'axe', 1: 'bottle', 2: 'broom', 3: 'button', 4: 'driller', 5: 'hammer', 6: 'light_bulb', 7: 'nail', 8: 'pliers',
#  9: 'scissors', 10: 'screw', 11: 'screwdriver', 12: 'tape', 13: 'vial', 14: 'wrench'}
