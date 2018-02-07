import os
import xml.etree.ElementTree as ET
from random import shuffle

import cv2
import numpy as np
import tensorflow as tf
from sklearn.utils import shuffle

import params
from statistics import get_imagenet_distributions
import math

class ImagenetDataPreparator:
    def __init__(self):
        self.images_path = 'imagenet_data/images'
        self.annotations_path = 'imagenet_data/annotations'
        self.tensor_anno_path = 'imagenet_data/tensor_annotations'
        self.writers_path = 'imagenet_data/tf_records'
        self.classification_path = 'imagenet_data/class_tf_records'

        self.train_ratio = 0.9

        self.image_names, self.label_names = self.prepare()

        self.distribution, self.names_by_classes = get_imagenet_distributions(
            [os.path.join(self.annotations_path, name) for name in os.listdir(self.annotations_path)])

        self.create_TFRecords(self.image_names, self.label_names)
        self.create_classification_data()
        print("Dataset ready!")

    def create_classification_data(self):
        if os.path.isdir(self.classification_path):
            print("No need to generate classification TFRecords")
            return

        os.mkdir(self.classification_path)
        imnames = [os.path.join(self.images_path, image) for image in os.listdir(self.images_path)]
        labels = [name.split('/')[-1].replace('.jpg', '') for name in imnames]
        labels = [''.join([i for i in s if not i.isdigit()])[:-1] for s in labels]
        labels = [params.imagenet_classes.index(label) for label in labels]

        writer = self.create_writers(self.classification_path, 'train_0')

        train_i = 0
        for i, (imgname, label) in enumerate(zip(imnames, labels)):
            print("\rGenerating classification TFRecords (%.2f)" % (i / len(imnames)), end='', flush=True)
            img = self.image_read(imgname)

            train_i += 1
            if train_i % 100 == 0:
                writer.close()
                writer = self.create_writers(self.classification_path, 'train_' + str(int(train_i / 100)))

            feature = {'train/image': self._bytes_feature(tf.compat.as_bytes(img.tostring())),
                       'train/label': self._int64_feature(label)}
            example = tf.train.Example(features=tf.train.Features(feature=feature))
            writer.write(example.SerializeToString())
        writer.close()
        print()

    @property
    def num_batches(self):
        train_names, val_names = self.get_tf_record_names()
        # train_len = sum(sum(1 for record in tf.python_io.tf_record_iterator(name)) for name in train_names)
        # val_len = sum(sum(1 for record in tf.python_io.tf_record_iterator(name)) for name in val_names)
        train_len = 6043
        val_len = 556
        return math.ceil(train_len / params.batch_size), math.ceil(val_len / params.batch_size)

    @property
    def num_classification_batches(self):
        filenames = [os.path.join(self.classification_path, name) for name in os.listdir(self.classification_path)]
        # train_len = sum(sum(1 for record in tf.python_io.tf_record_iterator(name)) for name in filenames)
        train_len = 5734
        return train_len // params.cls_batch_size

    def prepare(self):
        if not os.path.isdir(self.tensor_anno_path):
            os.mkdir(self.tensor_anno_path)
        print("Preparing dataset")
        image_names = sorted([name.replace('.jpg', '') for name in os.listdir(self.images_path)])
        annotation_names = sorted([name.replace('.xml', '') for name in os.listdir(self.annotations_path)])
        tensor_names = sorted([name.replace('.npy', '') for name in os.listdir(self.tensor_anno_path)])
        if not image_names == tensor_names:
            print("Generating tensor annotations")
            common_names = set(image_names).intersection(annotation_names)
            empty_annotations = list(set(image_names) - common_names)

            for name in common_names:
                try:
                    label = self.parse_xml(os.path.join(self.annotations_path, name + '.xml'))
                    if np.max(label) == 0.0:
                        empty_annotations.append(name)
                except Exception:
                    empty_annotations.append(name)

                else:
                    np.save(os.path.join(self.tensor_anno_path, name), label)
            for name in empty_annotations:
                if os.path.isfile(os.path.join(self.images_path, name + '.jpg')):
                    os.remove(os.path.join(self.images_path, name + '.jpg'))
                # in case or other extensions, like png
                if os.path.isfile(os.path.join(self.images_path, name)):
                    os.remove(os.path.join(self.images_path, name))
        return sorted([os.path.join(self.images_path, name) for name in os.listdir(self.images_path)]), sorted(
            [os.path.join(self.tensor_anno_path, name) for name in os.listdir(self.tensor_anno_path)])

    def parse_xml(self, filename):
        tree = ET.parse(filename)
        size = tree.find('size')
        width = int(size.find('width').text)
        height = int(size.find('height').text)
        if height == 0 or width == 0:
            raise Exception
        h_ratio = 1.0 * params.img_size / height
        w_ratio = 1.0 * params.img_size / width

        label = np.zeros((params.S, params.S, 5 + params.C))
        objs = tree.findall('object')

        for obj in objs:
            bbox = obj.find('bndbox')
            x1 = max(min((float(bbox.find('xmin').text) - 1) * w_ratio, params.img_size - 1), 0)
            y1 = max(min((float(bbox.find('ymin').text) - 1) * h_ratio, params.img_size - 1), 0)
            x2 = max(min((float(bbox.find('xmax').text) - 1) * w_ratio, params.img_size - 1), 0)
            y2 = max(min((float(bbox.find('ymax').text) - 1) * h_ratio, params.img_size - 1), 0)

            name = obj.find('name').text.lower().strip()
            name_en = params.imagenet_dictionary[name]
            cls_ind = params.imagenet_classes.index(name_en)

            boxes = [(x2 + x1) / 2.0, (y2 + y1) / 2.0, x2 - x1, y2 - y1]

            x_ind = int(boxes[0] * params.S / params.img_size)
            y_ind = int(boxes[1] * params.S / params.img_size)

            if label[y_ind, x_ind, 0] == 1:
                continue
            label[y_ind, x_ind, 0] = 1
            label[y_ind, x_ind, 1:5] = boxes
            label[y_ind, x_ind, 5 + cls_ind] = 1

        return label


    def image_read(self, imname):
        image = cv2.imread(imname)
        image = cv2.resize(image, (params.img_size, params.img_size))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
        image = (image / 255.0) * 2.0 - 1.0
        return image

    def create_writers(self, writers_folder, name):
        writer = tf.python_io.TFRecordWriter(os.path.join(writers_folder, name + '.tfrecord'))
        return writer

    def create_TFRecords(self, image_names, label_names):
        if os.path.isdir(self.writers_path):
            print("No need to generate TFRecords")
            return

        os.mkdir(self.writers_path)

        # shuffle it thoroughly! :)
        image_names, label_names = shuffle(image_names, label_names)

        train_i = 0
        val_i = 0
        train_writer = self.create_writers(self.writers_path, 'train_0')
        val_writer = self.create_writers(self.writers_path, 'val_0')

        for i, (imgname, label) in enumerate(zip(image_names, label_names)):
            print("\rGenerating TFRecords (%.2f)" % (i / len(image_names)), end='', flush=True)
            img = self.image_read(imgname)
            lbl = np.load(label).astype(np.float32)
            e = np.random.uniform(0,1)
            if e < self.train_ratio:
                train_i +=1
                if train_i % 100 == 0:
                    train_writer.close()
                    train_writer = self.create_writers(self.writers_path, 'train_' + str(int(train_i/100)))

                feature = {'train/label': self._bytes_feature(tf.compat.as_bytes(lbl.tostring())),
                           'train/image': self._bytes_feature(tf.compat.as_bytes(img.tostring()))}
                example = tf.train.Example(features=tf.train.Features(feature=feature))
                train_writer.write(example.SerializeToString())
            else:
                val_i += 1
                if val_i % 100 == 0:
                    val_writer.close()
                    val_writer = self.create_writers(self.writers_path, 'val_' + str(int(val_i/100)))

                feature = {'validation/label': self._bytes_feature(tf.compat.as_bytes(lbl.tostring())),
                           'validation/image': self._bytes_feature(tf.compat.as_bytes(img.tostring()))}
                example = tf.train.Example(features=tf.train.Features(feature=feature))
                val_writer.write(example.SerializeToString())
        train_writer.close()
        val_writer.close()
        print()

        # equalize data distribution
        max_distr = max(self.distribution.values())
        all_to_create = sum([max_distr - val for val in self.distribution.values()])
        i=0
        for (key, dist), names in zip(self.distribution.items(), self.names_by_classes.values()):
            to_create = max_distr - dist
            names = np.random.choice(names, to_create)
            image_names = [os.path.join(self.images_path, name + '.jpg') for name in names]
            label_names = [os.path.join(self.tensor_anno_path, name + '.npy') for name in names]

            train_i = 0
            val_i = 0
            train_writer = self.create_writers(self.writers_path, '_train_0')
            val_writer = self.create_writers(self.writers_path, '_val_0')

            for k, (imgname, label) in enumerate(zip(image_names, label_names)):
                print("\rUpdating TFRecords (%.2f)" % (i / all_to_create), end='', flush=True)
                img = self.image_read(imgname)
                img += np.random.uniform(0.0, 0.02) # randomize data
                lbl = np.load(label).astype(np.float32)
                lbl[:, :, 1:5] *= np.random.uniform(0.99, 1.01)
                e = np.random.uniform(0, 1)
                if e < self.train_ratio:
                    train_i += 1
                    if train_i % 100 == 0:
                        train_writer.close()
                        train_writer = self.create_writers(self.writers_path, '_train_' + str(int(train_i / 100)))

                    feature = {'train/label': self._bytes_feature(tf.compat.as_bytes(lbl.tostring())),
                               'train/image': self._bytes_feature(tf.compat.as_bytes(img.tostring()))}
                    example = tf.train.Example(features=tf.train.Features(feature=feature))
                    train_writer.write(example.SerializeToString())
                else:
                    val_i += 1
                    if val_i % 100 == 0:
                        val_writer.close()
                        val_writer = self.create_writers(self.writers_path, 'val_' + str(int(val_i/ 100)))

                    feature = {'validation/label': self._bytes_feature(tf.compat.as_bytes(lbl.tostring())),
                               'validation/image': self._bytes_feature(tf.compat.as_bytes(img.tostring()))}
                    example = tf.train.Example(features=tf.train.Features(feature=feature))
                    val_writer.write(example.SerializeToString())
                i += 1
            train_writer.close()
            val_writer.close()
        print()

    def _bytes_feature(self, value):
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

    def _int64_feature(self, value):
        return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

    def decode_data(self, batch_size, mode):
        feature = {mode + '/image': tf.FixedLenFeature([], tf.string),
                   mode + '/label': tf.FixedLenFeature([], tf.string)}

        train_names, val_names = self.get_tf_record_names()
        filenames = train_names if mode =='train' else val_names
        filename_queue = tf.train.string_input_producer(filenames)
        reader = tf.TFRecordReader()
        _, serialized_example = reader.read(filename_queue)
        features = tf.parse_single_example(serialized_example, features=feature)
        image = tf.reshape(tf.decode_raw(features[mode + '/image'], tf.float32), [params.img_size, params.img_size, 3])
        label = tf.reshape(tf.decode_raw(features[mode + '/label'], tf.float32), [params.S, params.S, 5 + params.C])

        images, labels = tf.train.shuffle_batch([image, label],
                                                batch_size=batch_size,
                                                capacity=5000,
                                                num_threads=4,
                                                min_after_dequeue=400,
                                                allow_smaller_final_batch=True)
        return images, labels

    def get_tf_record_names(self):
        filenames = os.listdir(self.writers_path)
        train = [os.path.join(self.writers_path, file) for file in filenames if 'train' in file]
        val = [os.path.join(self.writers_path, file) for file in filenames if 'val' in file]
        return train, val

    def decode_classification_data(self, batch_size):
        feature = {'train/image': tf.FixedLenFeature([], tf.string),
                   'train/label': tf.FixedLenFeature([], tf.int64)}
        filenames = [os.path.join(self.classification_path, name) for name in os.listdir(self.classification_path)]
        filename_queue = tf.train.string_input_producer(filenames)
        reader = tf.TFRecordReader()
        _, serialized_example = reader.read(filename_queue)
        features = tf.parse_single_example(serialized_example, features=feature)
        image = tf.reshape(tf.decode_raw(features['train/image'], tf.float32), [params.img_size, params.img_size, 3])
        label = tf.cast(features['train/label'], tf.int32)

        images, labels = tf.train.shuffle_batch([image, label],
                                                batch_size=batch_size,
                                                capacity=5000, # 4000 w maire ok
                                                num_threads=4,
                                                min_after_dequeue=400,
                                                allow_smaller_final_batch=True)
        return images, labels