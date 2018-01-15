import tensorflow as tf

from synth_data_generator import img
from utils import DataPreparator

S = 13
B = 5
C = 4

img_w = 484
img_h = 484
channels = 3

d_coord = 5

data_preparator = DataPreparator(S)
data = data_preparator.provide_data(img_w, img_h, 'synthetic')

# placeholders
img_placeholder = tf.placeholder(dtype = tf.float32, shape=(1, img_w, img_h, channels), name='img_placeholder')
presence_placeholder = tf.placeholder(dtype = tf.float32, shape=(S, S), name='presence_placeholder')
annotations_placeholder = tf.placeholder(dtype = tf.float32, shape=(None, 5), name='annotations_placeholder')

# conv
conv1 = tf.layers.conv2d(img_placeholder, 32, kernel_size=[3,3], strides=[1,1], activation=tf.nn.relu, kernel_initializer=tf.contrib.layers.xavier_initializer())
pool1 = tf.layers.max_pooling2d(conv1, pool_size=[2,2], strides=[2,2])

conv2 = tf.layers.conv2d(pool1, 64, kernel_size=[3,3], strides=[1,1], activation=tf.nn.relu, kernel_initializer=tf.contrib.layers.xavier_initializer())
pool2 = tf.layers.max_pooling2d(conv2, pool_size=[2,2], strides=[2,2])

conv3 = tf.layers.conv2d(pool2, 128, kernel_size=[3,3], strides=[1,1], activation=tf.nn.relu, kernel_initializer=tf.contrib.layers.xavier_initializer())
pool3 = tf.layers.max_pooling2d(conv3, pool_size=[2,2], strides=[2,2])

conv4 = tf.layers.conv2d(pool3, 256, kernel_size=[3,3], strides=[1,1], activation=tf.nn.relu, kernel_initializer=tf.contrib.layers.xavier_initializer())
pool4 = tf.layers.max_pooling2d(conv4, pool_size=[2,2], strides=[2,2])

conv5 = tf.layers.conv2d(pool4, 512, kernel_size=[3,3], strides=[1,1], activation=tf.nn.relu, kernel_initializer=tf.contrib.layers.xavier_initializer())
pool5 = tf.layers.max_pooling2d(conv5, pool_size=[2,2], strides=[2,2])

final_layer = tf.layers.conv2d(pool5, B*5+C, kernel_size=[1,1], strides=[1,1], kernel_initializer=tf.contrib.layers.xavier_initializer())

# losses
bdbox_slice, class_slice = tf.split(final_layer, num_or_size_splits=[B*5, C], axis=3)
bdbox_slice = tf.squeeze(bdbox_slice, axis=0)
class_slice = tf.squeeze(class_slice, axis=0)





with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    # data: tuple(resized image, presence matrix, (GT bounding boxes)) -> all dataset
    # GT bounding box is a tuple of relative values: x_center, y_center, w, h, class

    i=3
    batch = data[i]
    img = batch[0]
    presence_matrix = batch[1]
    annotations = batch[2]

    a = sess.run(annotations_placeholder, feed_dict={img_placeholder: [img], presence_placeholder: presence_matrix, annotations_placeholder: annotations})
    print(a)