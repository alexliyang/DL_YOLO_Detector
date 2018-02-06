import os
import xml.etree.ElementTree as ET
import params

def get_distributions(filenames):
    def name_from_xml(filename):
        tree = ET.parse(filename)
        size = tree.find('size')
        width = int(size.find('width').text)
        height = int(size.find('height').text)
        if height == 0 or width == 0:
            raise Exception

        objs = tree.findall('object')

        classes = set()
        for obj in objs:
            obj_name = params.transl[obj.find('name').text]
            classes.add(obj_name)

        return classes

    names_by_classes = {}
    for name in filenames:
        classes = name_from_xml(os.path.join('data', 'annotations', name + '.xml'))
        if list(classes)[0] in names_by_classes:
            names_by_classes[list(classes)[0]].append(name)
        else:
            names_by_classes[list(classes)[0]] = [name]

    distr_by_files = dict(zip(names_by_classes.keys(), [len(value) for value in names_by_classes.values()]))
    return distr_by_files, names_by_classes

def get_imagenet_distributions(filenames):
    def name_from_xml(filename):
        tree = ET.parse(filename)
        size = tree.find('size')
        width = int(size.find('width').text)
        height = int(size.find('height').text)
        if height == 0 or width == 0:
            raise Exception

        objs = tree.findall('object')

        classes = set()
        for obj in objs:
            obj_name = params.imagenet_dictionary[obj.find('name').text]
            classes.add(obj_name)

        return classes

    names_by_classes = {}

    for name in filenames:
        classes = name_from_xml(name)
        if list(classes)[0] in names_by_classes:
            names_by_classes[list(classes)[0]].append(name)
        else:
            names_by_classes[list(classes)[0]] = [name]

    distr_by_files = dict(zip(names_by_classes.keys(), [len(value) for value in names_by_classes.values()]))
    return distr_by_files, names_by_classes

# filenames = [name.replace('.jpg', '') for name in os.listdir('data/images')]
# a, b = get_distributions(filenames)
# print(a)
# print(b)
#
# # for (key, val) in zip(b.keys(), b.values()):
# #     print(key, len(val))
