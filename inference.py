import os
import numpy as np
import tensorflow as tf
import cv2
import time

from architecture import convolution, fully_connected
from parameters import params

# placeholders
from utils import draw_boxes, interpret_output

images_placeholder = tf.placeholder(tf.float32, shape=[None, params.img_size, params.img_size, 3])
dropout_placeholder = tf.placeholder(tf.bool, shape=())

conv = convolution.slim_conv(images_placeholder)
detection_logits = fully_connected.detection_dense(conv, dropout_placeholder)
classification_logits = fully_connected.classification_dense(conv, dropout_placeholder)

# params
video_path = 'sample_vids/3.mp4'
conv_model_name = 'custom_model_1_laptop'
dense_model_name = 'custom_model_1_laptop'

"""
ATTENTION! When attaching dense models of different S, make sure to change S in params.py properly in order
to match params S with dense model S. Take care also of C (temporarily switch dataset)!
"""
with tf.Session() as sess:
    init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
    sess.run(init_op)

    saver_conv = tf.train.Saver(var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='yolo'))
    saver_detection_dense = tf.train.Saver(var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='detection_dense'))
    saver_classification_dense = tf.train.Saver(var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='classification_dense'))

    saver_conv.restore(sess, os.path.join('models', conv_model_name + '_C', 'model_conv.ckpt'))
    saver_detection_dense.restore(sess, os.path.join('models', dense_model_name + '_D', 'model.ckpt'))
    saver_classification_dense.restore(sess, os.path.join('models', conv_model_name + '_C', 'model_dense.ckpt'))

    # video
    cap = cv2.VideoCapture(video_path)
    while (cap.isOpened()):
        ret, frame = cap.read()
        frame = cv2.resize(frame, (params.img_size, params.img_size))
        frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)

        img = frame.astype(np.float32)
        # # img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).astype(np.float32)
        img = (img / 255.0) * 2.0 - 1.0

        out = sess.run(detection_logits, feed_dict={images_placeholder: [img],
                                          dropout_placeholder: False})

        result = interpret_output(out[0, ...])
        coords = []
        ROIs = []
        for obj_data in result:
            # temporarily increased field of prediction
            obj_x1 = max(0, int(obj_data[1]) - int(0.75 * obj_data[3]))
            obj_x2 = min(params.img_size-1, int(obj_data[1]) + int(0.75 * obj_data[3]))
            obj_y2 = min(params.img_size-1, int(obj_data[2]) + int(0.75 * obj_data[4]))
            obj_y1 = max(0, int(obj_data[2]) - int(0.75 * obj_data[4]))

            coords.append([obj_y1, obj_x1, obj_y2, obj_x2])
            ROIs.append(cv2.resize(img[obj_y1: obj_y2, obj_x1: obj_x2], (params.img_size, params.img_size)))


        if ROIs:
            ROIs_classes = sess.run(classification_logits, feed_dict={images_placeholder: ROIs,
                                                    dropout_placeholder: False})
            ROIs_classes = np.argmax(ROIs_classes, axis=1)

            for i in range(len(ROIs_classes)):
                frame = cv2.rectangle(frame, (coords[i][1], coords[i][0]), (coords[i][3], coords[i][2]), color=(0,255,0), thickness=2)

                cv2.putText(frame, params.classes[ROIs_classes[i]], (coords[i][1], coords[i][0]+20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.75,
                            (0,255,0), 2)

        cv2.imshow('frame', frame)
        if cv2.waitKey(20) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
