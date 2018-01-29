import os

import cv2
import tensorflow as tf
import time
import numpy as np
import params
from architecture import convolution, fully_connected, loss_layer
from data_preparator import DataPreparator
from utils import prepare_training_dirs, draw_boxes

# params
model_name = 'temp-model'
conv_weights_path = 'pretrained_weights/YOLO_small.ckpt'

# data generation + dirs preparation
preparator = DataPreparator()
train_batches, val_batches = preparator.num_batches
prepare_training_dirs()

# noisers

# data provider
train_images, train_labels = preparator.decode_data(params.batch_size, 'train')
val_images, val_labels = preparator.decode_data(params.batch_size, 'validation')
t_channels = tf.unstack(train_images, axis=-1)
train_images = tf.stack([t_channels[2], t_channels[1], t_channels[0]], axis=-1)
v_channels = tf.unstack(train_images, axis=-1)
val_images = tf.stack([v_channels[2], v_channels[1], v_channels[0]], axis=-1)

# placeholders
images_placeholder = tf.placeholder(tf.float32, shape=[None, params.img_size, params.img_size, 3])
labels_placeholder = tf.placeholder(tf.float32, shape=[None, params.S, params.S, 5+params.C])
labels_noise_placeholder = tf.placeholder(tf.float32, shape=[None, params.S, params.S, 4])

# layers
conv = convolution.slim_conv(images_placeholder)
logits = fully_connected.custom_dense(conv, params.num_outputs, is_training=True)

# train_op
noised_xywh = tf.multiply(labels_placeholder[:, :, :, 1:5], labels_noise_placeholder)
a = tf.expand_dims(tf.ones_like(noised_xywh[:,:,:,0]), 3)
b = tf.tile(a, [1,1,1,params.C])
# a = tf.ones([None, params.S, params.S, 1], dtype=tf.float32)
# b = tf.ones([None, params.S, params.S, params.C], dtype=tf.float32)
dupa = tf.concat([a, noised_xywh, b], axis=3)
class_loss, object_loss, noobject_loss, coord_loss = loss_layer.losses(logits, labels_placeholder)
loss = class_loss + object_loss + noobject_loss + coord_loss

with tf.name_scope('summaries'):
    tf.summary.scalar('class_loss', class_loss)
    tf.summary.scalar('object_loss', object_loss)
    tf.summary.scalar('noobject_loss', noobject_loss)
    tf.summary.scalar('coord_loss', coord_loss)
    tf.summary.scalar('loss', loss)

trainable_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'dense')
train_op = tf.train.AdamOptimizer(params.detection_eta).minimize(loss, var_list=trainable_vars)
merged = tf.summary.merge_all()

with tf.Session() as sess:
    init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
    sess.run(init_op)
    saver_conv = tf.train.Saver(var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='yolo'))
    saver_dense = tf.train.Saver(var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='dense'))

    saver_conv.restore(sess, conv_weights_path)
    if os.path.isdir(os.path.join('models', model_name)):
        saver_dense.restore(sess, os.path.join('models', model_name, 'model.ckpt'))
        print(model_name + ' model loaded')
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)

    # TB writers
    train_writer = tf.summary.FileWriter(os.path.join('summaries', model_name + '_T'), sess.graph, flush_secs=60)
    val_writer = tf.summary.FileWriter(os.path.join('summaries', model_name + '_V'), flush_secs=60)
    images, labels = sess.run([train_images, train_labels])
    for epoch in range(params.epochs):
        for batch_idx in range(train_batches):

            lbl_noise = np.random.uniform(0.99, 1.01, size=[params.batch_size, params.S, params.S, 4])
            plac = sess.run(dupa, feed_dict={labels_placeholder: labels, labels_noise_placeholder: lbl_noise})
            print(np.mean(plac), plac.shape)

        #     _, cost, summary = sess.run([train_op, loss, merged], feed_dict={images_placeholder: images, labels_placeholder: labels})
        #     print('\rEpoch: %d of %d, batch: %d of %d, loss: %f' % (epoch, params.epochs, batch_idx, train_batches, cost))
        #     train_writer.add_summary(summary, global_step=epoch + epoch * batch_idx)
        #     train_writer.flush()
        #
        # for batch_idx in range(val_batches):
        #     images, labels = sess.run([val_images, val_labels])
        #     summary = sess.run(merged, feed_dict={images_placeholder: images, labels_placeholder: labels})
        #     val_writer.add_summary(summary, global_step=epoch + epoch * batch_idx)
        #     val_writer.flush()
        #
        # saver_dense.save(sess, os.path.join('models', model_name, 'model.ckpt'))
        #
        # images = sess.run(val_images)
        # output = sess.run(logits, feed_dict={images_placeholder: images})
        # tagged_img = draw_boxes(images[0], output)
        # cv2.imwrite(os.path.join('saved_images', model_name + str(epoch) + '.jpg'), tagged_img)

    coord.request_stop()
    coord.join(threads)
    sess.close()
