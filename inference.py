import os
import numpy as np
import tensorflow as tf
import cv2
from architecture import convolution, fully_connected
from parameters import params

# placeholders
from utils import draw_boxes

images_placeholder = tf.placeholder(tf.float32, shape=[None, params.img_size, params.img_size, 3])
dropout_placeholder = tf.placeholder(tf.bool, shape=())

conv = convolution.slim_conv(images_placeholder)
logits = fully_connected.detection_dense(conv, dropout_placeholder)

# params
video_path = 'sample_vids/3.mp4'
conv_model_name = 'custom_model_1_laptop'
dense_model_name = 'custom_model_1_laptop'

"""
ATTENTION! When attaching dense models of different S, make sure to change S in params.py properly in order
to match params S with dense model S!
"""
with tf.Session() as sess:
    init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
    sess.run(init_op)

    saver_conv = tf.train.Saver(var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='yolo'))
    saver_dense = tf.train.Saver(var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='detection_dense'))

    saver_conv.restore(sess, os.path.join('models', conv_model_name + '_C', 'model_conv.ckpt'))
    saver_dense.restore(sess, os.path.join('models', dense_model_name + '_D', 'model.ckpt'))

    # video
    cap = cv2.VideoCapture(video_path)
    while (cap.isOpened()):
        ret, frame = cap.read()
        frame = cv2.resize(frame, (params.img_size, params.img_size))
        frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        img = frame.astype(np.float32)
        # # img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).astype(np.float32)
        img = (img / 255.0) * 2.0 - 1.0
        out = sess.run(logits, feed_dict={images_placeholder: [img],
                                          dropout_placeholder: False})
        tagged_img = draw_boxes(img, out, None)  # labels[0] or None
        tagged_img = tagged_img.astype(np.uint8)
        cv2.imshow('frame', tagged_img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
