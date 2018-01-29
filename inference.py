import os

import cv2
import numpy as np
import tensorflow as tf

import params
from architecture import convolution, fully_connected
from utils import draw_boxes, draw_video_boxes

# params
mode = 'camera'  # images or camera

images_path = 'sample_data'
model_name = 'model13'
conv_weights_path = 'pretrained_weights/YOLO_small.ckpt'

# placeholders
images_placeholder = tf.placeholder(tf.float32, shape=[1, params.img_size, params.img_size, 3])

# layers
conv = convolution.slim_conv(images_placeholder)
logits = fully_connected.custom_dense(conv, params.num_outputs)

with tf.Session() as sess:
    init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
    sess.run(init_op)
    saver_conv = tf.train.Saver(var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='yolo'))
    saver_dense = tf.train.Saver(var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='dense'))

    try:
        saver_conv.restore(sess, conv_weights_path)
        saver_dense.restore(sess, os.path.join('models', model_name, 'model.ckpt'))
    except Exception:
        print("No weights found in " + model_name)
        exit(0)

    if mode == 'images':
        if not os.path.isdir(os.path.join(images_path, 'tagged_images')):
            os.mkdir(os.path.join(images_path, 'tagged_images'))

        filenames = os.listdir(images_path)

        for filename in filenames:
            if filename != 'tagged_images':
                image = cv2.imread(os.path.join(images_path, filename))
                image = cv2.resize(image, (params.img_size, params.img_size))
                image = (image / 255.0) * 2.0 - 1.0
                image = np.expand_dims(image, axis=0)
                net_output = sess.run(logits, feed_dict={images_placeholder: image})
                tagged_image = draw_boxes(image[0], net_output)
                cv2.imwrite(os.path.join(images_path, 'tagged_images', filename), tagged_image)

    elif mode == 'camera':
        cv2.namedWindow("preview")
        vc = cv2.VideoCapture(0)

        if vc.isOpened():  # try to get the first frame
            rval, frame = vc.read()
        else:
            rval = False

        while rval:
            rval, frame = vc.read()
            image = cv2.resize(frame, (params.img_size, params.img_size))
            image = (image / 255.0) * 2.0 - 1.0
            image = np.expand_dims(image, axis=0)
            net_output = sess.run(logits, feed_dict={images_placeholder: image})
            tagged_image = draw_video_boxes(image[0], net_output)
            cv2.imshow("preview", tagged_image)

            key = cv2.waitKey(1)
            if key == ord('q'):  # exit on q
                break

        cv2.destroyWindow("preview")
        vc.release()
