import os

import tensorflow as tf

import params
from architecture import convolution, fully_connected, loss_layer
from data_preparator import DataPreparator
from utils import prepare_training_dirs

model_name = 'classification_model_1'
conv_weights_path = 'pretrained_weights/YOLO_small.ckpt'

preparator = DataPreparator()
num_batches = preparator.num_classification_batches
prepare_training_dirs()

# classification data
images_feed, labels_feed = preparator.decode_classification_data(params.cls_batch_size)
# c_channels = tf.unstack(images_feed, axis=-1)
# images_feed = tf.stack([c_channels[2], c_channels[1], c_channels[0]], axis=-1)

# placeholders
images_placeholder = tf.placeholder(tf.float32, shape=[None, params.img_size, params.img_size, 3])
labels_palceholder = tf.placeholder(tf.int32, shape=None)

# layers
conv = convolution.slim_conv(images_placeholder)
logits = fully_connected.classification_dense(conv)

# train op
loss = loss_layer.classification_loss(logits, labels_palceholder)
with tf.name_scope('classification_summaries'):
    tf.summary.scalar('classification_loss', loss)
train_op = tf.train.AdamOptimizer(params.classification_eta).minimize(loss)
merged = tf.summary.merge_all()

with tf.Session() as sess:
    init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
    sess.run(init_op)
    saver = tf.train.Saver()
    saver_pretrained = tf.train.Saver(var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='yolo'))

    # fine tuned model available
    if os.path.isdir(os.path.join('models', model_name + '_C')):
        saver.restore(sess, os.path.join('models', model_name + '_C', 'model.ckpt'))
        print(model_name + ' model loaded (fine tuned model)')
    # only pretrained slim weights available
    else:
        saver_pretrained.restore(sess, conv_weights_path)
        print(model_name + ' model loaded (pretrained model)')

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)

    writer = tf.summary.FileWriter(os.path.join('classification_summaries', model_name + '_C'), flush_secs=60)

    for epoch in range(params.classification_epochs):
        for batch_idx in range(num_batches):
            images, labels = sess.run([images_feed, labels_feed])
            _, cost, summary, out = sess.run([train_op, loss, merged, logits],
                                        feed_dict={images_placeholder: images,
                                                   labels_palceholder: labels})
            print('\rClassification epoch: %d of %d, batch: %d of %d, loss: %f' % (epoch, params.classification_epochs, batch_idx, num_batches, cost), flush=True, end='')
            writer.add_summary(summary, global_step=epoch * num_batches + batch_idx)
            writer.flush()
        saver.save(sess, os.path.join('models', model_name + 'C', 'model.ckpt'))

    coord.request_stop()
    coord.join(threads)
    sess.close()
