import numpy as np
import tensorflow as tf

from parameters import params


def calc_iou(boxes1, boxes2):
    boxes1 = tf.stack([boxes1[:, :, :, :, 0] - boxes1[:, :, :, :, 2] / 2.0,
                       boxes1[:, :, :, :, 1] - boxes1[:, :, :, :, 3] / 2.0,
                       boxes1[:, :, :, :, 0] + boxes1[:, :, :, :, 2] / 2.0,
                       boxes1[:, :, :, :, 1] + boxes1[:, :, :, :, 3] / 2.0])
    boxes1 = tf.transpose(boxes1, [1, 2, 3, 4, 0])

    boxes2 = tf.stack([boxes2[:, :, :, :, 0] - boxes2[:, :, :, :, 2] / 2.0,
                       boxes2[:, :, :, :, 1] - boxes2[:, :, :, :, 3] / 2.0,
                       boxes2[:, :, :, :, 0] + boxes2[:, :, :, :, 2] / 2.0,
                       boxes2[:, :, :, :, 1] + boxes2[:, :, :, :, 3] / 2.0])
    boxes2 = tf.transpose(boxes2, [1, 2, 3, 4, 0])

    lu = tf.maximum(boxes1[:, :, :, :, :2], boxes2[:, :, :, :, :2])
    rd = tf.minimum(boxes1[:, :, :, :, 2:], boxes2[:, :, :, :, 2:])

    intersection = tf.maximum(0.0, rd - lu)
    inter_square = intersection[:, :, :, :, 0] * intersection[:, :, :, :, 1]

    square1 = (boxes1[:, :, :, :, 2] - boxes1[:, :, :, :, 0]) * (boxes1[:, :, :, :, 3] - boxes1[:, :, :, :, 1])
    square2 = (boxes2[:, :, :, :, 2] - boxes2[:, :, :, :, 0]) * (boxes2[:, :, :, :, 3] - boxes2[:, :, :, :, 1])

    union_square = tf.maximum(square1 + square2 - inter_square, 1e-10)
    return tf.clip_by_value(inter_square / union_square, 0.0, 1.0)


def losses(logits, labels):
    offset = np.transpose(np.reshape(np.array([np.arange(params.S)] * params.S * params.B),
                                     (params.B, params.S, params.S)), (1, 2, 0))

    with tf.variable_scope('loss'):
        predict_classes = tf.reshape(logits[:, :params.boundary1],
                                     [params.detection_batch_size, params.S, params.S, params.C])
        predict_scales = tf.reshape(logits[:, params.boundary1:params.boundary2],
                                    [params.detection_batch_size, params.S, params.S, params.B])
        predict_boxes = tf.reshape(logits[:, params.boundary2:],
                                   [params.detection_batch_size, params.S, params.S, params.B, 4])
        response = tf.reshape(labels[:, :, :, 0], [params.detection_batch_size, params.S, params.S, 1])
        boxes = tf.reshape(labels[:, :, :, 1:5], [params.detection_batch_size, params.S, params.S, 1, 4])
        boxes = tf.tile(boxes, [1, 1, 1, params.B, 1]) / params.img_size
        classes = labels[:, :, :, 5:]
        offset = tf.constant(offset, dtype=tf.float32)
        offset = tf.reshape(offset, [1, params.S, params.S, params.B])
        offset = tf.tile(offset, [params.detection_batch_size, 1, 1, 1])

        predict_boxes_tran = tf.stack([(predict_boxes[:, :, :, :, 0] + offset) / params.S,
                                       (predict_boxes[:, :, :, :, 1] + tf.transpose(offset, (0, 2, 1, 3))) / params.S,
                                       tf.square(predict_boxes[:, :, :, :, 2]),
                                       tf.square(predict_boxes[:, :, :, :, 3])])
        predict_boxes_tran = tf.transpose(predict_boxes_tran, [1, 2, 3, 4, 0])

        iou_predict_truth = calc_iou(predict_boxes_tran, boxes)
        object_mask = tf.reduce_max(iou_predict_truth, 3, keep_dims=True)
        object_mask = tf.cast((iou_predict_truth >= object_mask), tf.float32) * response
        noobject_mask = tf.ones_like(object_mask, dtype=tf.float32) - object_mask
        boxes_tran = tf.stack([boxes[:, :, :, :, 0] * params.S - offset,
                               boxes[:, :, :, :, 1] * params.S - tf.transpose(offset, (0, 2, 1, 3)),
                               tf.sqrt(boxes[:, :, :, :, 2]),
                               tf.sqrt(boxes[:, :, :, :, 3])])
        boxes_tran = tf.transpose(boxes_tran, [1, 2, 3, 4, 0])

        class_delta = response * (predict_classes - classes)
        class_loss = tf.reduce_mean(tf.reduce_sum(tf.square(class_delta), axis=[1, 2, 3]),
                                    name='class_loss') * params.class_coefficient

        object_delta = object_mask * (predict_scales - iou_predict_truth)
        object_loss = tf.reduce_mean(tf.reduce_sum(tf.square(object_delta), axis=[1, 2, 3]),
                                     name='object_loss') * params.object_coefficient

        noobject_delta = noobject_mask * predict_scales
        noobject_loss = tf.reduce_mean(tf.reduce_sum(tf.square(noobject_delta), axis=[1, 2, 3]),
                                       name='noobject_loss') * params.no_object_coefficient

        coord_mask = tf.expand_dims(object_mask, 4)
        boxes_delta = coord_mask * (predict_boxes - boxes_tran)
        coord_loss = tf.reduce_mean(tf.reduce_sum(tf.square(boxes_delta), axis=[1, 2, 3, 4]),
                                    name='coord_loss') * params.coord_coefficient

        return class_loss, object_loss, noobject_loss, coord_loss


def classification_loss(logits, labels):
    return tf.losses.sparse_softmax_cross_entropy(labels, logits)
