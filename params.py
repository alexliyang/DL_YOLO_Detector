# training params
batch_size = 10
epochs = 50
classification_eta = 0.001
detection_eta = 0.00001

# loss coefficients
object_coefficient = 2.0
no_object_coefficient = 1.0
class_coefficient = 7.0 # 2.0
coord_coefficient = 5.0


classes = ['guzik', 'klucz_plaski', 'kombinerki', 'nozyczki', 'probowka', 'srubokret', 'tasma']
C = len(classes)
B = 2
S = 7
num_outputs = S * S * (C + 5 * B)

img_size = 448
boundary1 = S * S * C
boundary2 = boundary1 + S * S * B

# thresholds
threshold = 0.01
IOU_threshold = 0.1
# threshold = 0.2
# IOU_threshold = 0.5