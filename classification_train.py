import os

import cv2
import numpy as np
import tensorflow as tf

from architecture import convolution, fully_connected, loss_layer
from data_preparators.imagenet_preparator import ImagenetPreparator
from data_preparators.custom_data_preparator import CustomDataPreparator
from parameters import params
from utils import prepare_before_training

# data generation + dirs preparation
prepare_before_training()
if not os.path.isdir('saved_images/' + params.classification_model_name + '_C'):
    os.mkdir('saved_images/' + params.classification_model_name + '_C')
if params.dataset == 'imagenet':
    preparator = ImagenetPreparator(data_root_path=params.root_path,
                                    classes=params.classes,
                                    name_converter=params.name_converter)
elif params.dataset == 'custom':
    preparator = CustomDataPreparator(data_root_path=params.root_path,
                                    classes=params.classes,
                                    name_converter=params.name_converter)

num_batches = preparator.num_batches(type='classification', batch_size=params.classification_batch_size)

# classification data
images_feed, labels_feed = preparator.decode_classification_data(params.classification_batch_size, params.c_capacity,
                                                                 params.c_num_threads, params.c_min_after_deque)

# placeholders
images_placeholder = tf.placeholder(tf.float32, shape=[None, params.img_size, params.img_size, 3])
labels_palceholder = tf.placeholder(tf.int32, shape=None)
dropout_placeholder = tf.placeholder(tf.bool, shape=())

# layers
conv = convolution.slim_conv(images_placeholder)
logits = fully_connected.classification_dense(conv, dropout_placeholder)
softmax_out = tf.nn.softmax(logits)

# train op
loss = loss_layer.classification_loss(logits, labels_palceholder)
with tf.name_scope('classification_summaries'):
    tf.summary.scalar('classification_loss', loss)
train_op = tf.train.AdamOptimizer(params.classification_eta).minimize(loss)
merged = tf.summary.merge_all()

with tf.Session() as sess:
    init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
    sess.run(init_op)
    saver_dense = tf.train.Saver(
        var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='classification_dense'))
    saver_conv = tf.train.Saver(var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='yolo'))

    # fine tuned model available
    if os.path.isdir(os.path.join('models', params.classification_model_name + '_C')):
        saver_conv.restore(sess, os.path.join('models', params.classification_model_name + '_C', 'model_conv.ckpt'))
        saver_dense.restore(sess, os.path.join('models', params.classification_model_name + '_C', 'model_dense.ckpt'))
        print(params.classification_model_name + ' model loaded (fine tuned model)')
    # only pretrained slim weights available
    else:
        saver_conv.restore(sess, params.yolo_weights_path)
        print('Pretrained model loaded')

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)

    writer = tf.summary.FileWriter(os.path.join('summaries/classification_summaries', params.classification_model_name + '_C'),
                                   flush_secs=120)

    i = 0
    for epoch in range(params.classification_epochs):
        for batch_idx in range(num_batches):
            images, labels = sess.run([images_feed, labels_feed])
            _, cost, summary = sess.run([train_op, loss, merged],
                                        feed_dict={images_placeholder: images,
                                                   labels_palceholder: labels,
                                                   dropout_placeholder: True})
            if batch_idx % 10 == 0:
                print('Classification epoch: %d of %d, batch: %d of %d, loss: %f' % (
                epoch, params.classification_epochs, batch_idx, num_batches, cost), labels)
            writer.add_summary(summary, global_step=epoch * num_batches + batch_idx)

        images = sess.run(images_feed)
        out = sess.run(softmax_out, feed_dict={images_placeholder: images,
                                               dropout_placeholder: False})
        for (img, lbl) in zip(images, out):
            cv2.imwrite('saved_images/' + params.classification_model_name + '_C/' + params.classes[np.argmax(lbl)] + '_' + str(i) + '.jpg',
                        (img + 1.0) * 0.5 * 255)
            i += 1

        saver_conv.save(sess, os.path.join('models', params.classification_model_name + '_C', 'model_conv.ckpt'))
        saver_dense.save(sess, os.path.join('models', params.classification_model_name + '_C', 'model_dense.ckpt'))
        print()

    coord.request_stop()
    coord.join(threads)
    sess.close()
