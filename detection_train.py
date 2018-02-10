import os
import cv2
import tensorflow as tf

from architecture import convolution, fully_connected, loss_layer
from data_preparators.imagenet_preparator import ImagenetPreparator
from data_preparators.custom_data_preparator import CustomDataPreparator
from parameters import params
from utils import prepare_before_training, draw_boxes

# data generation + dirs preparation
prepare_before_training()
if not os.path.isdir('saved_images/' + params.detection_model_name + '_D'):
    os.mkdir('saved_images/' + params.detection_model_name + '_D')
if params.dataset == 'imagenet':
    preparator = ImagenetPreparator(data_root_path=params.root_path,
                                    classes=params.classes,
                                    name_converter=params.name_converter)
elif params.dataset == 'custom':
    preparator = CustomDataPreparator(data_root_path=params.root_path,
                                    classes=params.classes,
                                    name_converter=params.name_converter)

train_batches = preparator.num_batches(type='train', batch_size=params.detection_batch_size)
val_batches = preparator.num_batches(type='validation', batch_size=params.detection_batch_size)

# training data
train_images, train_labels = preparator.decode_detection_data(params.detection_batch_size, 'train', params.d_capacity,
                                                               params.d_num_threads, params.d_min_after_deque)

# validation data
val_images, val_labels = preparator.decode_detection_data(params.detection_batch_size, 'validation', params.d_capacity,
                                                           params.d_num_threads, params.d_min_after_deque)

# placeholders
images_placeholder = tf.placeholder(tf.float32, shape=[None, params.img_size, params.img_size, 3])
labels_placeholder = tf.placeholder(tf.float32, shape=[None, params.S, params.S, 5 + params.C])
dropout_placeholder = tf.placeholder(tf.bool, shape=())

# labels augmentation
ones = tf.expand_dims(tf.ones_like(labels_placeholder[:, :, :, 0]), 3)
noise_mask = tf.concat([ones, tf.random_uniform([params.detection_batch_size, params.S, params.S, 4],
                                                params.augmentation_noise_low, params.augmentation_noise_high),
                        tf.tile(ones, [1, 1, 1, params.C])], axis=3)
noisy_labels = tf.multiply(labels_placeholder, noise_mask)

# images augmentation
noisy_images = tf.multiply(images_placeholder,
                           tf.random_uniform([params.detection_batch_size, params.img_size, params.img_size, 3],
                                             params.augmentation_noise_low, params.augmentation_noise_high))

# layers
conv = convolution.slim_conv(noisy_images)
logits = fully_connected.detection_dense(conv, dropout_placeholder)

# train_op
class_loss, object_loss, noobject_loss, coord_loss = loss_layer.losses(logits, noisy_labels)
loss = class_loss + object_loss + noobject_loss + coord_loss

with tf.name_scope('summaries'):
    tf.summary.scalar('class_loss', class_loss)
    tf.summary.scalar('object_loss', object_loss)
    tf.summary.scalar('noobject_loss', noobject_loss)
    tf.summary.scalar('coord_loss', coord_loss)
    tf.summary.scalar('loss', loss)

trainable_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'detection_dense')
train_op = tf.train.AdamOptimizer(params.detection_eta).minimize(loss, var_list=trainable_vars)
merged = tf.summary.merge_all()

with tf.Session() as sess:
    init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
    sess.run(init_op)
    saver_conv = tf.train.Saver(var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='yolo'))
    saver_dense = tf.train.Saver(var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='detection_dense'))

    if os.path.isdir(os.path.join('models', params.classification_model_name + '_C')):
        saver_conv.restore(sess, os.path.join('models', params.classification_model_name + '_C', 'model_conv.ckpt'))
        print(params.classification_model_name + ' model loaded (fine tuned conv)')
    else:
        saver_conv.restore(sess, params.yolo_weights_path)
        print('Pretrained yolo conv model loaded')

    if os.path.isdir(os.path.join('models', params.detection_model_name + '_D')):
        saver_dense.restore(sess, os.path.join('models', params.detection_model_name + '_D', 'model.ckpt'))
        print(params.detection_model_name + ' model loaded (dense)')

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)

    # TB writers
    train_writer = tf.summary.FileWriter(os.path.join('summaries/detection_summaries', params.detection_model_name + '_T'), sess.graph, flush_secs=120)
    val_writer = tf.summary.FileWriter(os.path.join('summaries/detection_summaries', params.detection_model_name + '_V'), flush_secs=120)

    for epoch in range(params.detection_epochs):
        for batch_idx in range(train_batches):
            images, labels = sess.run([train_images, train_labels])
            _, cost, summary = sess.run([train_op, loss, merged],
                                        feed_dict={images_placeholder: images, labels_placeholder: labels,
                                                   dropout_placeholder: True})
            if batch_idx % 10 == 0:
                print('\rEpoch: %d of %d, batch: %d of %d, loss: %f' % (epoch, params.detection_epochs, batch_idx, train_batches, cost))
            train_writer.add_summary(summary, global_step=epoch * train_batches + batch_idx)

        for batch_idx in range(val_batches):
            images, labels = sess.run([val_images, val_labels])
            summary = sess.run(merged, feed_dict={images_placeholder: images, labels_placeholder: labels,
                                                  dropout_placeholder: False})

            val_writer.add_summary(summary, global_step=epoch * val_batches + batch_idx)

        saver_dense.save(sess, os.path.join('models', params.detection_model_name + '_D', 'model.ckpt'))

        images = sess.run(val_images)
        output = sess.run(logits, feed_dict={images_placeholder: images, dropout_placeholder: False})
        tagged_img = draw_boxes(images[0], output)
        cv2.imwrite(os.path.join('saved_images', params.detection_model_name + '_D',  str(epoch) + '.jpg'), tagged_img)
        print()

    coord.request_stop()
    coord.join(threads)
    sess.close()
