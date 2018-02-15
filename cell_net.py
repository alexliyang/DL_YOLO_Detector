import os
from random import randint

import cv2
import numpy as np
import tensorflow as tf

from architecture.convolution import conv_model
from cell_net_utils import resize_label, image_read, embed_output, possibly_create_dirs, decode_train_data, \
    augment_rotate, augment_crop, augment_translate, augment_stack, create_augmented_tf_records, generate_cell_net_data
from parameters import params


# use if data was not generated
#generate_cell_net_data('cell_data', params.img_size, params.name_converter, params.classes)

# params
S = 14  # match with conv output
eta = 0.00001
threshold_area = int(params.img_size / S) ** 2 / 2
batch_size = 15
epochs = 50
img_save_checkpoint = 500
tfrecord_length = 10
capacity = 400
num_threads = 1
min_after_deque = 50

# paths
yolo_weights_path = 'models/yolo_pretrained/YOLO_small.ckpt'
model_to_save_path = 'models/cell_network_15_02_eta0_00001_adam_50epochs_batch10'
pretrained_model_path = None

t_images_path = 'cell_data/train_images/'
t_labels_path = 'cell_data/train_labels/'
v_images_path = 'cell_data/val_images/'
v_labels_path = 'cell_data/val_labels/'
train_records_path = 'cell_data/train_records/'
embedded_images_path = 'cell_data/output_images/'
possibly_create_dirs(embedded_images_path, model_to_save_path)

# delete if generated!!
#augmentations = 10# number of dataset augmentations
#train_image_filenames = sorted([t_images_path + name for name in os.listdir(t_images_path)])
#train_labels_filenames = sorted([t_labels_path + name for name in os.listdir(t_labels_path)])
#for i in range(augmentations):
#    create_augmented_tf_records(i, train_image_filenames, train_labels_filenames, train_records_path, tfrecord_length , S, threshold_area)

val_image_filenames = sorted([v_images_path + name for name in os.listdir(v_images_path)])
val_labels_filenames = sorted([v_labels_path + name for name in os.listdir(v_labels_path)])
val_data_len = len(val_image_filenames)

num_batches_t = len(os.listdir(train_records_path)) * tfrecord_length // batch_size
num_batches_v = val_data_len // batch_size

# model
train_images, train_labels = decode_train_data(train_records_path, S, batch_size, capacity,
                                                               num_threads, min_after_deque)
images_placeholder = tf.placeholder(dtype=tf.float32, shape=[None, params.img_size, params.img_size, 3])
labels_placeholder = tf.placeholder(dtype=tf.float32, shape=[None, S, S, params.C])
conv = conv_model(images_placeholder)
output = tf.layers.conv2d(conv, params.C, kernel_size=[1, 1],
                          kernel_initializer=tf.truncated_normal_initializer(0.0, 0.1))
sigmoid_output = tf.nn.sigmoid(output)

# characteristics
loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=labels_placeholder, logits=output)
loss = tf.reduce_mean(loss)
accuracy = tf.reduce_sum(
    tf.cast(tf.equal(tf.argmax(labels_placeholder, axis=3), tf.argmax(sigmoid_output, axis=3)), tf.float32)) / (
                       S * S * batch_size)

with tf.name_scope('summaries'):
    tf.summary.scalar('loss', loss)
    tf.summary.scalar('accuracy', accuracy)
train_op = tf.train.AdamOptimizer(eta).minimize(loss)
merged = tf.summary.merge_all()

with tf.Session() as sess:
    init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
    sess.run(init_op)
    saver = tf.train.Saver()

    # loading models
    if pretrained_model_path:
        saver.restore(sess, pretrained_model_path + '/model.ckpt')
        print(pretrained_model_path, ' model loaded')
    else:
        saver_yolo = tf.train.Saver(var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='yolo'))
        saver_yolo.restore(sess, yolo_weights_path)
        print('YOLO model loaded')
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)
    train_writer = tf.summary.FileWriter('summaries/cell_network/' + model_to_save_path.replace('models/', '') + '_T',
                                         sess.graph)
    val_writer = tf.summary.FileWriter('summaries/cell_network/' + model_to_save_path.replace('models/', '') + '_V',
                                       sess.graph)

    for epoch in range(epochs):
        for batch_idx in range(num_batches_t):
            # read from tf records
            images, labels = sess.run([train_images, train_labels])
            _, cost, summary = sess.run([train_op, loss, merged], feed_dict={images_placeholder: images,
                                                                             labels_placeholder: labels})
            print('\rTraining, epoch: %d of %d, batch: %d of %d, loss: %f' % (epoch, epochs, batch_idx, num_batches_t, cost))
            train_writer.add_summary(summary, epoch * num_batches_t + batch_idx)
            train_writer.flush()

            if batch_idx % img_save_checkpoint == 0:
                image = image_read(val_image_filenames[randint(0, val_data_len - 1)])
                output = sess.run(sigmoid_output, feed_dict={images_placeholder: [image]})
                embedded_output = embed_output((image + 1) / 2, output[0], 0.5, S, params.img_size)
                cv2.imwrite(embedded_images_path + '_' + str(epoch) + '_' + str(batch_idx) + '.jpg',
                            embedded_output * 255.0)

        # read from disc
        for batch_idx in range(num_batches_v):
            ids = np.random.choice(range(val_data_len), batch_size)
            images = [image_read(val_image_filenames[i]) for i in ids]
            labels = [resize_label(np.load(val_labels_filenames[i]), S, params.C, params.img_size, threshold_area) for i
                      in ids]

            summary = sess.run(merged, feed_dict={images_placeholder: images,
                                                  labels_placeholder: labels})
            print('\rValidation, epoch: %d of %d, batch: %d of %d' % (epoch, epochs, batch_idx, num_batches_v))
            val_writer.add_summary(summary, epoch * num_batches_v + batch_idx)
            val_writer.flush()

        saver.save(sess, os.path.join(model_to_save_path, 'model.ckpt'))
        if epoch % 10 == 0 and epoch > 0:
            saver.save(sess, os.path.join(model_to_save_path, str(epoch) + '_model.ckpt'))

    coord.request_stop()
    coord.join(threads)
    sess.close()