import cv2
import numpy as np
import pickle
import tensorflow as tf
import os
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
_, _, image_names, xml_names = pickle.load(open('cell_data/dataset_info.p', 'rb'))
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

    for img_name, xml_name in zip(image_names, xml_names):
        img = cv2.imread(img_name)
        img = cv2.resize(img, (params.img_size, params.img_size))
        img = (img / 255.0) * 2.0 - 1.0

        # get gt bdboxes
        gt_bounding_boxes = get_gt_bdboxes(xml_name, params.name_converter, params.classes, params.img_size)

        # predict bdboxes
        logits = sess.run(sigmoid_output, feed_dict={images_placeholder: [img]})
        logits = logits[0]
        thres_logits = np.copy(logits)
        thres_logits[thres_logits >= 0.5] = 1
        thres_logits[thres_logits < 0.5] = 0
        thres_logits = remove_outliers(thres_logits, 1)
        pred_bounding_boxes = get_bounding_boxes(thres_logits, logits, S, params.img_size, min_contour_area=0)

        compute_mAP_recall_precision(gt_bounding_boxes, pred_bounding_boxes, params.C)
        for gt_box in gt_bounding_boxes:
            img = cv2.rectangle(img, (int(gt_box[1]), int(gt_box[2])), (int(gt_box[3]), int(gt_box[4])), color = (0,0,1), thickness=2)
        for pred_box in pred_bounding_boxes:
            img = cv2.rectangle(img, (int(pred_box[1]), int(pred_box[2])), (int(pred_box[3]), int(pred_box[4])), color = (0,1,0), thickness=2)
        cv2.imshow('', img)
        cv2.waitKey(1000)

        # print('gt', gt_bounding_boxes)
        # print(get_info(gt_bounding_boxes))
        # print('predicted', pred_bounding_boxes)
        # print(get_info(pred_bounding_boxes))
        # print('map', compute_mAP(gt_bounding_boxes, pred_bounding_boxes))
        # print()

        # drawable_img = (img + 1) / 2
        # embedded_cells = draw_predicted_cells(drawable_img, thres_logits, S, params.img_size)
        # embedded_bdboxes = draw_bounding_boxes(embedded_cells, pred_bounding_boxes, colours)
        # cv2.imshow('Detections', embedded_bdboxes)
        # cv2.waitKey(2000)

# {0: 'axe', 1: 'bottle', 2: 'broom', 3: 'button', 4: 'driller', 5: 'hammer', 6: 'light_bulb', 7: 'nail', 8: 'pliers',
#  9: 'scissors', 10: 'screw', 11: 'screwdriver', 12: 'tape', 13: 'vial', 14: 'wrench'}
