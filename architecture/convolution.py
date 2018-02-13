import tensorflow as tf
import numpy as np

slim = tf.contrib.slim


def slim_conv(input_placeholder):
    """
    Returns tensor of dimensions: [batch_size, S, S, 1024]. To use pretrained weights (YOLO_small.cktp), S must be equal
    to 7.
    This is a modified version of huseinzol05 network (https://github.com/huseinzol05/YOLO-Object-Detection-Tensorflow)
    :param input_placeholder: tensor of size [batch_size, img_size, img_size, 3]
    """

    def leaky_relu(alpha):
        def op(inputs):
            return tf.maximum(alpha * inputs, inputs)

        return op

    with tf.variable_scope('yolo'):
        with slim.arg_scope([slim.conv2d], activation_fn=leaky_relu(0.1),
                            weights_initializer=tf.truncated_normal_initializer(0.0, 0.01),
                            weights_regularizer=slim.l2_regularizer(0.0005)):
            net = tf.pad(input_placeholder, np.array([[0, 0], [3, 3], [3, 3], [0, 0]]), name='pad_1')
            net = slim.conv2d(net, 64, 7, 2, padding='VALID', scope='conv_2')
            net = slim.max_pool2d(net, 2, padding='SAME', scope='pool_3')
            net = slim.conv2d(net, 192, 3, scope='conv_4')
            net = slim.max_pool2d(net, 2, padding='SAME', scope='pool_5')
            net = slim.conv2d(net, 128, 1, scope='conv_6')
            net = slim.conv2d(net, 256, 3, scope='conv_7')
            net = slim.conv2d(net, 256, 1, scope='conv_8')
            net = slim.conv2d(net, 512, 3, scope='conv_9')
            net = slim.max_pool2d(net, 2, padding='SAME', scope='pool_10')
            net = slim.conv2d(net, 256, 1, scope='conv_11')
            net = slim.conv2d(net, 512, 3, scope='conv_12')
            net = slim.conv2d(net, 256, 1, scope='conv_13')
            net = slim.conv2d(net, 512, 3, scope='conv_14')
            net = slim.conv2d(net, 256, 1, scope='conv_15')
            net = slim.conv2d(net, 512, 3, scope='conv_16')
            net = slim.conv2d(net, 256, 1, scope='conv_17')
            net = slim.conv2d(net, 512, 3, scope='conv_18')
            net = slim.conv2d(net, 512, 1, scope='conv_19')
            net = slim.conv2d(net, 1024, 3, scope='conv_20')
            net = slim.max_pool2d(net, 2, padding='SAME', scope='pool_21')
            net = slim.conv2d(net, 512, 1, scope='conv_22')
            net = slim.conv2d(net, 1024, 3, scope='conv_23')
            net = slim.conv2d(net, 512, 1, scope='conv_24')
            net = slim.conv2d(net, 1024, 3, scope='conv_25')
            net = slim.conv2d(net, 1024, 3, scope='conv_26')
            net = tf.pad(net, np.array([[0, 0], [1, 1], [1, 1], [0, 0]]), name='pad_27')
            net = slim.conv2d(net, 1024, 3, 2, padding='VALID', scope='conv_28')
            net = slim.conv2d(net, 1024, 3, scope='conv_29')
            net = slim.conv2d(net, 1024, 3, scope='conv_30')
            return net

def conv_model(input_placeholder):
    """
    Returns tensor of dimensions: [batch_size, S, S, 1024]. To use pretrained weights (YOLO_small.cktp), S must be equal
    to 7.
    This is a modified version of huseinzol05 network (https://github.com/huseinzol05/YOLO-Object-Detection-Tensorflow)
    :param input_placeholder: tensor of size [batch_size, img_size, img_size, 3]
    """

    def leaky_relu(alpha):
        def op(inputs):
            return tf.maximum(alpha * inputs, inputs)

        return op

    with tf.variable_scope('yolo'):
        with slim.arg_scope([slim.conv2d], activation_fn=leaky_relu(0.1),
                            weights_initializer=tf.truncated_normal_initializer(0.0, 0.01),
                            weights_regularizer=slim.l2_regularizer(0.0005)):
            net = tf.pad(input_placeholder, np.array([[0, 0], [3, 3], [3, 3], [0, 0]]), name='pad_1')
            net = slim.conv2d(net, 64, 7, 2, padding='VALID', scope='conv_2')
            net = slim.max_pool2d(net, 2, padding='SAME', scope='pool_3')
            net = slim.conv2d(net, 192, 3, scope='conv_4')
            net = slim.max_pool2d(net, 2, padding='SAME', scope='pool_5')
            net = slim.conv2d(net, 128, 1, scope='conv_6')
            net = slim.conv2d(net, 256, 3, scope='conv_7')
            net = slim.conv2d(net, 256, 1, scope='conv_8')
            net = slim.conv2d(net, 512, 3, scope='conv_9')
            net = slim.max_pool2d(net, 2, padding='SAME', scope='pool_10')
            net = slim.conv2d(net, 256, 1, scope='conv_11')
            net = slim.conv2d(net, 512, 3, scope='conv_12')
            net = slim.conv2d(net, 256, 1, scope='conv_13')
            net = slim.conv2d(net, 512, 3, scope='conv_14')
            net = slim.conv2d(net, 256, 1, scope='conv_15')
            net = slim.conv2d(net, 512, 3, scope='conv_16')
            net = slim.conv2d(net, 256, 1, scope='conv_17')
            net = slim.conv2d(net, 512, 3, scope='conv_18')
            net = slim.conv2d(net, 512, 1, scope='conv_19')
            net = slim.conv2d(net, 1024, 3, scope='conv_20') # (5, 28, 28, 1024)
            net = slim.max_pool2d(net, 2, padding='SAME', scope='pool_21')
            net = slim.conv2d(net, 512, 1, scope='conv_22')
            net = slim.conv2d(net, 1024, 3, scope='conv_23')
            net = slim.conv2d(net, 512, 1, scope='conv_24')
            net = slim.conv2d(net, 1024, 3, scope='conv_25')
            net = slim.conv2d(net, 1024, 3, scope='conv_26') # (5, 14, 14, 1024)
            # net = tf.pad(net, np.array([[0, 0], [1, 1], [1, 1], [0, 0]]), name='pad_27')
            # net = slim.conv2d(net, 1024, 3, 2, padding='VALID', scope='conv_28')
            # net = slim.conv2d(net, 1024, 3, scope='conv_29')
            # net = slim.conv2d(net, 1024, 3, scope='conv_30')
            return net