import pickle
import numpy as np
import cv2
import tensorflow as tf

graph_path = 'inception/graph.pb'
label_dict = pickle.load(open('inception/dictionary.p', 'rb'))
image = cv2.imread('sample_images/dog3.jpg')

def create_graph(path):
    with tf.gfile.FastGFile(path, 'rb') as graph_file:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(graph_file.read())
        _ = tf.import_graph_def(graph_def, name='')


create_graph(graph_path)


with tf.Session() as sess:
    input_image = sess.graph.get_tensor_by_name('DecodeJpeg:0')
    softmax_out = sess.graph.get_tensor_by_name('softmax:0')
    conv_layer = sess.graph.get_tensor_by_name('mixed_10/conv:0')


    dense = tf.layers.dense(softmax_out, 10)
    dense1 = tf.layers.dense(dense, 3)
    mean = tf.reduce_mean(dense1)
    train_op = tf.train.AdamOptimizer().minimize(mean)
    tf.summary.scalar('mean', mean)
    merged = tf.summary.merge_all()

    summary_writer = tf.summary.FileWriter('summaries/', sess.graph)
    sess.run(tf.global_variables_initializer())

    for i in range(1):
        _, summary, dn, out, mn, conv = sess.run([train_op, merged, dense1, softmax_out, mean, conv_layer], feed_dict={input_image: image})
        summary_writer.add_summary(summary)
        summary_writer.flush()

        print(label_dict[np.argmax(out)], mn, conv.shape)



