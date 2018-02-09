from parameters.specific_params import CustomDatasetParams, ImagenetParams


# names
classification_model_name = 'custom_model_1'
detection_model_name = 'custom_model_1'
dataset = 'custom' # 'imagenet', 'custom'
root_path = 'data/custom'


yolo_weights_path = 'models/yolo_pretrained/YOLO_small.ckpt'

# general params
classification_batch_size = 5
classification_epochs = 10
classification_eta = 0.00001

detection_batch_size = 10
detection_epochs = 50
detection_eta = 0.00001

# detection
d_capacity = 400
d_num_threads = 4
d_min_after_deque = 50

# classification
c_capacity = 400
c_num_threads = 4
c_min_after_deque = 50

threshold = 0.01
IOU_threshold = 0.1

B = 2
S = 7
img_size = 448
detection_tf_record_size_limit = 100
classification_tf_record_size_limit = 100

object_coefficient = 2.0
no_object_coefficient = 1.0
class_coefficient = 2.0
coord_coefficient = 5.0

augmentation_noise_low = 0.999
augmentation_noise_high = 1.001

# dataset specific params
if dataset=='imagenet':
    specific_params = ImagenetParams()
elif dataset == 'custom':
    specific_params = CustomDatasetParams()

name_converter = specific_params.name_converter
classes = specific_params.classes
C = len(classes)
num_dense_outputs = S * S * (C + 5 * B)
boundary1 = S * S * C
boundary2 = boundary1 + S * S * B