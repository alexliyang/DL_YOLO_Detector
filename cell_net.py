import os
from random import randint, uniform
from sklearn.utils import shuffle
import cv2
import numpy as np
import tensorflow as tf

from architecture.convolution import conv_model
from cell_net_utils import resize_label, image_read, embed_output
from parameters import params


# use if data was not generated
# generate_cell_net_data('cell_data', params.img_size, params.name_converter, params.classes)

def softmax(target, axis, name=None):
    with tf.name_scope(name, 'softmax', values=[target]):
        max_axis = tf.reduce_max(target, axis, keep_dims=True)
        target_exp = tf.exp(target - max_axis)
        normalize = tf.reduce_sum(target_exp, axis, keep_dims=True)
        softmax = target_exp / normalize
        return softmax


# params
S = 14  # match with conv output
threshold_area = int(params.img_size / S) ** 2 / 2
eta = 0.00001

images_placeholder = tf.placeholder(dtype=tf.float32, shape=[None, params.img_size, params.img_size, 3])
labels_placeholder = tf.placeholder(dtype=tf.float32, shape=[None, S, S, params.C])

conv = conv_model(images_placeholder)
output_layer = tf.layers.conv2d(conv, params.C, kernel_size=[1, 1],
                                kernel_initializer=tf.truncated_normal_initializer(0.0, 0.1))
sigmoid_output = tf.nn.sigmoid(output_layer)

# softmax_output = softmax(output_layer, axis = 3)
# cross_entropy = tf.reduce_mean(-tf.reduce_sum(labels_placeholder * tf.log(softmax_output), reduction_indices=[3]))

cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(labels=labels_placeholder, logits=output_layer)
cross_entropy = tf.reduce_mean(cross_entropy)

# # loss = tf.losses.softmax_cross_entropy(onehot_labels=labels_placeholder, logits=output_layer)
# loss = tf.nn.softmax_cross_entropy_with_logits(labels = output_layer, logits=labels_placeholder)
# loss = tf.reduce_mean(loss)

# todo sigmoid cross entrophy
#
with tf.name_scope('summaries'):
    tf.summary.scalar('loss', cross_entropy)

train_op = tf.train.AdamOptimizer(eta).minimize(cross_entropy)
merged = tf.summary.merge_all()

images_path = 'cell_data/images/'
labels_path = 'cell_data/labels/'
outputs_images_path = 'cell_data/output_images/'
images_filenames = sorted([images_path + name for name in os.listdir(images_path)])
labels_filenames = sorted([labels_path + name for name in os.listdir(labels_path)])
images_filenames, labels_filenames = shuffle(images_filenames, labels_filenames)

train_image_filenames = images_filenames[:int(0.9 * len(images_filenames))]
train_labels_filenames = labels_filenames[:int(0.9 * len(labels_filenames))]
val_image_filenames = images_filenames[int(0.9 * len(images_filenames)):]
val_labels_filenames = labels_filenames[int(0.9 * len(labels_filenames)):]


yolo_weights_path = 'models/yolo_pretrained/YOLO_small.ckpt'

batch_size = 5
epochs = 50

train_data_len = len(train_image_filenames)
val_data_len = len(val_image_filenames)
num_batches = train_data_len // batch_size

if not os.path.isdir(outputs_images_path):
    os.mkdir(outputs_images_path)


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    saver_conv = tf.train.Saver(var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='yolo'))
    saver_conv.restore(sess, params.yolo_weights_path)

    writer = tf.summary.FileWriter('summaries/cell_network/' + str(np.random.randint(0, 999999)), sess.graph)

    for epoch in range(epochs):
        for batch_idx in range(num_batches):
            ids = np.random.choice(range(train_data_len), batch_size)

            images = [image_read(train_image_filenames[i]) for i in ids]
            labels = [resize_label(np.load(train_labels_filenames[i]), S, params.C, params.img_size, threshold_area) for i in
                      ids]

            _, cost, summary= sess.run([train_op, cross_entropy, merged], feed_dict={images_placeholder: images,
                                                            labels_placeholder: labels})
            print(epoch, batch_idx, cost)
            writer.add_summary(summary, epoch * batch_size + batch_idx)
            writer.flush()

            if batch_idx % 100 ==0:
                image = image_read(val_image_filenames[randint(0, val_data_len-1)])
                output = sess.run(sigmoid_output, feed_dict={images_placeholder: [image]})
                embedded_output = embed_output((image+1)/2, output[0], 0.3, S, params.img_size)
                cv2.imwrite(outputs_images_path + '_' + str(epoch) + '_' + str(batch_idx) + '.jpg', embedded_output * 255.0)