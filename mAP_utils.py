import numpy as np

def get_info(bdboxes_list):
    info_dict = dict()
    bbox_num = 0
    for line in bdboxes_list:
        id, xmin, ymin, xmax, ymax, score = line
        if not id in info_dict:
            info_dict[id] = list()
        info_dict[id].append([float(xmin), float(ymin), float(xmax), float(ymax), float(score)])
        bbox_num += 1
    return bbox_num, info_dict

def iou(predict_bbox, ground_truth_bbox):
    predict_area = (predict_bbox[2] - predict_bbox[0]) * (predict_bbox[3] - predict_bbox[1])
    ground_truth_area = (ground_truth_bbox[2] - ground_truth_bbox[0]) * (ground_truth_bbox[3] - ground_truth_bbox[1])
    inter_x = min(predict_bbox[2], ground_truth_bbox[2]) - max(predict_bbox[0], ground_truth_bbox[0])
    inter_y = min(predict_bbox[3], ground_truth_bbox[3]) - max(predict_bbox[1], ground_truth_bbox[1])
    if inter_x <= 0 or inter_y <= 0:
        return 0
    inter_area = inter_x * inter_y
    return inter_area / (predict_area + ground_truth_area - inter_area)

def compare(predict_list, ground_truth_list):
    score_list = []
    match_list = []
    ground_truth_unused = [True for i in range(len(ground_truth_list))]
    for predict_bbox in predict_list:
        match = False
        for i in range(len(ground_truth_list)):
            if ground_truth_unused[i]:
                if iou(predict_bbox, ground_truth_list[i]) > 0.5:
                    match = True
                    ground_truth_unused[i] = False
                    break

        score_list.append(predict_bbox[-1])
        match_list.append(int(match))
    score_match_list = list(zip(score_list, match_list))
    score_match_list.sort(key=lambda x: x[0], reverse=True)

    return [element[1] for element in score_match_list]


def compute_mAP_recall_precision(gt_bdbox_list, pred_bdbox_list, num_classes):
    """
    Computes mAP, recall and precision for single image, for every single class and average
    """

    # values per classes, if class is not present both in predicted and ground truth values, it should not be counted!
    AP = dict(zip(range(num_classes), [None] * num_classes))
    recall = dict(zip(range(num_classes), [None] * num_classes))
    precision = dict(zip(range(num_classes), [None] * num_classes))

    _, predict_dict = get_info(pred_bdbox_list)
    _, ground_truth_dict = get_info(gt_bdbox_list)

    for key in predict_dict.keys():
        # bdboxes for no gt boxes - precision drops to zero
        if key not in ground_truth_dict.keys():
            AP[key] = 0
            precision[key] = 0
            recall[key] = 0

        else:
            score_match_list = compare(predict_dict[key], ground_truth_dict[key])
            p = list()
            r = list()
            predict_num = 0
            truth_num = 0
            ground_truth_bbox_num = len(ground_truth_dict[key])
            for item in score_match_list:
                predict_num += 1
                truth_num += item
                p.append(float(truth_num) / predict_num)
                r.append(float(truth_num) / ground_truth_bbox_num)
            p = [0] + p
            r = [0] + r
            mAP = 0
            for i in range(1, len(p)):
                mAP += p[i] * (r[i] - r[i - 1])
            AP[key] = mAP
            precision[key] = truth_num / predict_num
            recall[key] = truth_num / ground_truth_bbox_num


    # if there is a gt box without predicted box
    for gt_key in ground_truth_dict.keys():
        if AP[gt_key] == None:
            AP[gt_key] = 0
            precision[gt_key] = 0
            recall[gt_key] = 0

    mean_AP = np.mean([item for item in AP.values() if item is not None])
    mean_precision = np.mean([item for item in precision.values() if item is not None])
    mean_recall = np.mean([item for item in recall.values() if item is not None])

    return AP, recall, precision, mean_AP, mean_recall, mean_precision