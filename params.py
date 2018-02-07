imagenet_dictionary = {
    # buttons
    'n02928608': 'button',
    'n04197781': 'button',
    'n03057541': 'button',

    # wrenches
    'n04606574': 'wrench',
    'n02886434': 'wrench',
    'n04155177': 'wrench',
    'n03218446': 'wrench',
    'n02680754': 'wrench',
    'n03848168': 'wrench',
    'n03130866': 'wrench',

    # pliers
    'n03966976': 'pliers',
    'n03816530': 'pliers',
    'n03675907': 'pliers',

    # scissors
    'n04148054': 'scissors',
    'n03044934': 'scissors',
    'n04424692': 'scissors',

    # vials
    'n03923379': 'vial',

    # screwdrivers
    'n04154565': 'screwdriver',
    'n03923692': 'screwdriver',
    'n03361683': 'screwdriver',
    'n04279987': 'screwdriver',

    # tapes
    'n02992795': 'tape',

    # hammers
    'n03482001': 'hammer',
    'n03481172': 'hammer',
    'n03481521': 'hammer',
    'n02966545': 'hammer',

    # bottles
    'n02876657': 'bottle',
    'n04557648': 'bottle',
    'n03983396': 'bottle',

    # light bulbs
    'n03665924': 'light_bulb',
    'n13864035': 'light_bulb',

    # nails
    'n03804744': 'nail',
    'n03052917': 'nail',

    # screws
    'n04153751': 'screw',
    'n03633886': 'screw',
    'n04154340': 'screw',

    # drillers
    'n03239726': 'driller',
    'n03995372': 'driller',
    'n03240140': 'driller',

    # brooms
    'n02906734': 'broom',

    # axes
    'n02764044': 'axe',
    'n03346289': 'axe',
}
imagenet_classes = list(set(imagenet_dictionary.values()))


# training params
cls_batch_size = 2
batch_size = 10
epochs = 50
classification_epochs = 50
classification_eta = 0.0001
detection_eta = 0.00001

# loss coefficients
object_coefficient = 2.0
no_object_coefficient = 1.0
class_coefficient = 2.0  # 2.0
coord_coefficient = 5.0

classes_pl = ['guzik', 'klucz_plaski', 'kombinerki', 'nozyczki', 'probowka', 'srubokret', 'tasma']
classes = ['button', 'wrench', 'pliers', 'scissors', 'vial', 'screwdriver', 'tape']
transl = dict(zip(classes_pl, classes))
C = len(imagenet_classes)
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
