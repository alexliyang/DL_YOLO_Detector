import cv2
import numpy as np
import tensorflow as tf

from architecture.convolution import conv_model
from cell_net_utils import get_bounding_boxes, remove_outliers, draw_bounding_boxes, colours, draw_predicted_cells
from parameters import params
import os
# params
S = 14
threshold_area = int(params.img_size / S) ** 2 / 2

# paths
pretrained_model_path = 'models/cell_network_15_02_eta0_00001_adam_50epochs_batch10'

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

    with open('somefile.txt', 'a') as the_file:
        for name in os.listdir('sample_images'):
            img = cv2.imread('sample_images/' + name)
            img = cv2.resize(img, (params.img_size, params.img_size))
            img = (img / 255.0) * 2.0 - 1.0

            logits = sess.run(sigmoid_output, feed_dict={images_placeholder: [img]})
            logits = logits[0]
            thres_logits = np.copy(logits)
            thres_logits[thres_logits < 0.5] = 0
            thres_logits[thres_logits >= 0.5] = 1
            thres_logits = remove_outliers(thres_logits, 4)
            bounding_boxes = get_bounding_boxes(thres_logits, logits, S, params.img_size)

            for box in bounding_boxes:
                x, y, w, h, c = box[1], box[2], box[3], box[4], params.classes[box[0]]
                x_c = x + 0.5 * w
                y_c = y + 0.5 * h
                print(name, c, x_c, y_c)
                the_file.write("%s_%s_%s_%s_%s" % (name, c, x_c, y_c, '\n'))

            drawable_img = (img + 1) / 2
            embedded_cells = draw_predicted_cells(drawable_img, thres_logits, S, params.img_size)
            embedded_bdboxes = draw_bounding_boxes(embedded_cells, bounding_boxes, colours)
            cv2.imshow('Detections', embedded_bdboxes)
            cv2.waitKey(2000)

    # cap = cv2.VideoCapture('sample_vids/5.mp4')
    # while cap.isOpened():
    #     ret, frame = cap.read()
    #
    #     img = cv2.resize(frame, (params.img_size, params.img_size))
    #     img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    #     img = (img / 255.0) * 2.0 - 1.0
    #
    #     logits = sess.run(sigmoid_output, feed_dict={images_placeholder: [img]})
    #     logits = logits[0]
    #     thres_logits = np.copy(logits)
    #     thres_logits[thres_logits >= 0.5] = 1
    #     thres_logits[thres_logits < 0.5] = 0
    #     thres_logits = remove_outliers(thres_logits, 1)
    #     bounding_boxes = get_bounding_boxes(thres_logits, logits, S, params.img_size)
    #
    #     drawable_img = (img + 1) / 2
    #     embedded_cells = draw_predicted_cells(drawable_img, thres_logits, S, params.img_size)
    #     embedded_bdboxes = draw_bounding_boxes(embedded_cells, bounding_boxes, colours)
    #
    #     cv2.imshow('Detections', embedded_bdboxes)
    #     if cv2.waitKey(1) & 0xFF == ord('q'):
    #         break

# {0: 'axe', 1: 'bottle', 2: 'broom', 3: 'button', 4: 'driller', 5: 'hammer', 6: 'light_bulb', 7: 'nail', 8: 'pliers',
#  9: 'scissors', 10: 'screw', 11: 'screwdriver', 12: 'tape', 13: 'vial', 14: 'wrench'}
