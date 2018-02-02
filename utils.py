import os
import cv2
import numpy as np
import params


def prepare_training_dirs():
    if not os.path.isdir('models'):
        os.mkdir('models')
    if not os.path.isdir('summaries'):
        os.mkdir('summaries')
    if not os.path.isdir('classification_summaries'):
        os.mkdir('classification_summaries')


def draw_result(img, result):
    for i in range(len(result)):
        x = int(result[i][1])
        y = int(result[i][2])
        w = int(result[i][3] / 2)
        h = int(result[i][4] / 2)
        cv2.rectangle(img, (x - w, y - h), (x + w, y + h), (0, 255, 0), 2)
        cv2.rectangle(img, (x - w, y - h - 20), (x + w, y - h), (125, 125, 125), -1)
        cv2.putText(img, result[i][0] + ' : %.2f' % result[i][5], (x - w + 5, y - h - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (-0.5,-0.5,-0.5,), 1)
    return img


def iou(box1, box2):
    tb = min(box1[0] + 0.5 * box1[2], box2[0] + 0.5 * box2[2]) - max(box1[0] - 0.5 * box1[2], box2[0] - 0.5 * box2[2])
    lr = min(box1[1] + 0.5 * box1[3], box2[1] + 0.5 * box2[3]) - max(box1[1] - 0.5 * box1[3], box2[1] - 0.5 * box2[3])
    if tb < 0 or lr < 0:
        intersection = 0
    else:
        intersection = tb * lr
    return intersection / (box1[2] * box1[3] + box2[2] * box2[3] - intersection)


def interpret_output(output):
    probs = np.zeros((params.S, params.S, params.B, params.C))
    class_probs = np.reshape(output[0: params.boundary1], (params.S, params.S, params.C))
    scales = np.reshape(output[params.boundary1: params.boundary2], (params.S, params.S, params.B))
    boxes = np.reshape(output[params.boundary2:], (params.S, params.S, params.B, 4))
    offset = np.transpose(
        np.reshape(np.array([np.arange(params.S)] * params.S * params.B),
                   [params.B, params.S, params.S]), (1, 2, 0))

    boxes[:, :, :, 0] += offset
    boxes[:, :, :, 1] += np.transpose(offset, (1, 0, 2))
    boxes[:, :, :, :2] = 1.0 * boxes[:, :, :, 0:2] / params.S
    boxes[:, :, :, 2:] = np.square(boxes[:, :, :, 2:])

    boxes *= params.img_size

    for i in range(params.B):
        for j in range(params.C):
            probs[:, :, i, j] = np.multiply(class_probs[:, :, j], scales[:, :, i])

    filter_mat_probs = np.array(probs >= params.threshold, dtype='bool')
    filter_mat_boxes = np.nonzero(filter_mat_probs)
    boxes_filtered = boxes[filter_mat_boxes[0], filter_mat_boxes[1], filter_mat_boxes[2]]
    probs_filtered = probs[filter_mat_probs]
    classes_num_filtered = np.argmax(filter_mat_probs, axis=3)[
        filter_mat_boxes[0], filter_mat_boxes[1], filter_mat_boxes[2]]

    argsort = np.array(np.argsort(probs_filtered))[::-1]
    boxes_filtered = boxes_filtered[argsort]
    probs_filtered = probs_filtered[argsort]
    classes_num_filtered = classes_num_filtered[argsort]

    for i in range(len(boxes_filtered)):
        if probs_filtered[i] == 0:
            continue
        for j in range(i + 1, len(boxes_filtered)):
            if iou(boxes_filtered[i], boxes_filtered[j]) > params.IOU_threshold:
                probs_filtered[j] = 0.0

    filter_iou = np.array(probs_filtered > 0.0, dtype='bool')
    boxes_filtered = boxes_filtered[filter_iou]
    probs_filtered = probs_filtered[filter_iou]
    classes_num_filtered = classes_num_filtered[filter_iou]

    result = []
    for i in range(len(boxes_filtered)):
        result.append([params.classes[classes_num_filtered[i]], boxes_filtered[i][0], boxes_filtered[i][1],
                       boxes_filtered[i][2], boxes_filtered[i][3], probs_filtered[i]])

    return result


def net_readable_img(img):
    img = cv2.resize(img, (params.img_size, params.img_size))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)
    img = (img / 255.0) * 2.0 - 1.0
    img = np.reshape(img, (1, params.img_size, params.img_size, 3))
    return img


def draw_boxes(img, logits):
    img_h = img.shape[0]
    img_w = img.shape[1]
    result = interpret_output(logits[0, ...])
    print(result)
    for i in range(len(result)):
        result[i][1] *= (1.0 * img_w / params.img_size)
        result[i][2] *= (1.0 * img_h / params.img_size)
        result[i][3] *= (1.0 * img_w / params.img_size)
        result[i][4] *= (1.0 * img_h / params.img_size)
    tagged_img = draw_result(img, result)
    return (tagged_img + 1.0) / 2 * 255

def draw_video_boxes(img, logits):
    img_h = img.shape[0]
    img_w = img.shape[1]
    result = interpret_output(logits[0, ...])
    for i in range(len(result)):
        result[i][1] *= (1.0 * img_w / params.img_size)
        result[i][2] *= (1.0 * img_h / params.img_size)
        result[i][3] *= (1.0 * img_w / params.img_size)
        result[i][4] *= (1.0 * img_h / params.img_size)
    tagged_img = draw_result(img, result)
    return (tagged_img + 1.0) / 2
