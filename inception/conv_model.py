import tensorflow as tf

graph_path = 'inception/graph.pb'

def conv_model(sess):
    with tf.gfile.FastGFile(graph_path, 'rb') as graph_file:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(graph_file.read())
        _ = tf.import_graph_def(graph_def, name='')

    input_image = sess.graph.get_tensor_by_name('DecodeJpeg:0')
    conv_layer = sess.graph.get_tensor_by_name('pool_3:0')

    return input_image, conv_layer