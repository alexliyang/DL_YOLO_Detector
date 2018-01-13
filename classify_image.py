import pickle
import cv2
import tensorflow as tf

from inception.conv_model import conv_model

label_dict = pickle.load(open('inception/dictionary.p', 'rb'))
image = cv2.imread('sample_images/dog3.jpg')

S = 13
B = 5
C = 4

with tf.Session() as sess:
    input_placeholder, last_conv_layer = conv_model(sess)

    # produce S, S x box
    flatten = tf.layers.flatten(last_conv_layer)
    dense_1 = tf.layers.dense(flatten, units=4096, activation=tf.nn.leaky_relu)
    dense_2 = tf.layers.dense(dense_1, units=S*S*(B*5+C), activation=tf.nn.leaky_relu)
    reshaped = tf.reshape(dense_2, shape=[1, S, S, B*5+C])

    #losses itp

    sess.run(tf.global_variables_initializer())

    for i in range(1):
        conv = sess.run(out_box, feed_dict={input_placeholder: image})
        print(conv.shape)



