import tensorflow as tf
from parameters import params

slim = tf.contrib.slim


def slim_dense(conv_output, num_classes, is_training=False):
    """
    Returns tensor of dimensions: [batch_size, num_classes]. To use pretrained weights (YOLO_small.cktp), num_classes must be equal
    to 20. This set of layers is appropriate to testing original network, but not especially for training on own datasets
    with num_classes different than original 20.
    This is a modified version of huseinzol05 network (https://github.com/huseinzol05/YOLO-Object-Detection-Tensorflow)
    :param conv_output: output of convolutional part of network
    """

    def leaky_relu(alpha):
        def op(inputs):
            return tf.maximum(alpha * inputs, inputs)

    with tf.variable_scope('yolo'):
        with slim.arg_scope([slim.fully_connected], activation_fn=leaky_relu(0.1),
                            weights_initializer=tf.truncated_normal_initializer(0.0, 0.01),
                            weights_regularizer=slim.l2_regularizer(0.0005)):
            net = tf.transpose(conv_output, [0, 3, 1, 2], name='trans_31')
            net = slim.flatten(net, scope='flat_32')
            net = slim.fully_connected(net, 512, scope='fc_33')
            net = slim.fully_connected(net, 4096, scope='fc_34')
            net = slim.dropout(net, keep_prob=0.5, is_training=is_training, scope='dropout_35')
            net = slim.fully_connected(net, num_classes, activation_fn=None, scope='fc_36')
            return net


def detection_dense(conv_output, dropout_placeholder):
    """
    Custom fully connected set of layers. Enables training on other sets than original with 20 classes
    :param conv_output: output of convolutional part of network
    :return: tensor of dimensions: [batch_size, num_classes]
    """
    with tf.variable_scope('detection_dense', reuse=tf.AUTO_REUSE):
        tran = tf.transpose(conv_output, [0, 3, 1, 2])
        flat = tf.layers.flatten(tran)
        dense = tf.layers.dense(flat, 512,
                                activation=tf.nn.leaky_relu,
                                kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                use_bias=True,
                                kernel_regularizer=tf.contrib.layers.l2_regularizer(0.1))
        dropout = tf.layers.dropout(dense, training=dropout_placeholder)
        dense = tf.layers.dense(dropout, 2048,
                                activation=tf.nn.leaky_relu,
                                kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                use_bias=True,
                                kernel_regularizer=tf.contrib.layers.l2_regularizer(0.1))
        dropout = tf.layers.dropout(dense, training=dropout_placeholder)
        dense = tf.layers.dense(dropout, 4096,
                                activation=tf.nn.leaky_relu,
                                kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                use_bias=True,
                                kernel_regularizer=tf.contrib.layers.l2_regularizer(0.1))
        dropout = tf.layers.dropout(dense, training=dropout_placeholder)
        logits = tf.layers.dense(dropout, params.num_dense_outputs, activation=None)
        return logits


def classification_dense(conv_output, dropout_placeholder):
    """
    Custom fully connected set of layers, created for classification. Enables training on other sets than original with 20 classes
    :param conv_output: output of convolutional part of network
    :return: tensor of dimensions: [batch_size, num_classes]
    """
    with tf.variable_scope('classification_dense', reuse=tf.AUTO_REUSE):
        tran = tf.transpose(conv_output, [0, 3, 1, 2])
        flat = tf.layers.flatten(tran)
        dense = tf.layers.dense(flat, 512,
                                activation=tf.nn.leaky_relu,
                                kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                use_bias=True,
                                kernel_regularizer=tf.contrib.layers.l2_regularizer(0.1))
        dropout = tf.layers.dropout(dense, training=dropout_placeholder)
        dense = tf.layers.dense(dropout, 2048,
                                activation=tf.nn.leaky_relu,
                                kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                use_bias=True,
                                kernel_regularizer=tf.contrib.layers.l2_regularizer(0.1))
        dropout = tf.layers.dropout(dense, training=dropout_placeholder)
        dense = tf.layers.dense(dropout, 4096,
                                activation=tf.nn.leaky_relu,
                                kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                use_bias=True,
                                kernel_regularizer=tf.contrib.layers.l2_regularizer(0.1))
        logits = tf.layers.dense(dense, params.C, activation=None)
        return logits
