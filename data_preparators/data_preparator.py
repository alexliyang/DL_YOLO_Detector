import os
import pickle

import xml.etree.ElementTree as ET

import cv2
import numpy as np
import tensorflow as tf

from parameters import params


class DataPreparator:
    def __init__(self, data_root_path, classes, name_converter):
        """
        :param data_root_path: parent folder for all data
        :param classes: list with classes to use
        :param name_converter: dictionary with converions rules (pl>en or wnid>en)
        """
        # things to be specified in implementation
        self.data_root_path = data_root_path
        self.classes = classes
        self.name_converter = name_converter

        # thing not needing specification
        self.train_ratio = 0.9
        self.batch_stats = None
        self.detection_images_path = None
        self.detection_annotations_path = None
        self.classification_images_path = None
        self.detection_labels_path = None
        self.detection_tfrecords_path = None
        self.classification_tfrecords_path = None
        self.batch_stats_path = None

        self.make_dirs()
        self.download_data()

        self.prepare_valid_data(self.name_converter, self.classes)
        self.train_image_names, self.train_label_names, self.validation_image_names, self.validation_label_names = self.get_saved_names()
        self.generate_classification_tfrecords(params.classification_tf_record_size_limit)
        self.generate_detection_tfrecords(params.detection_tf_record_size_limit)

    def tf_record_filenames(self, folder_path, suffix=None):
        """
        Returns a list of .tfrecord filenames.
        :param folder_path: path to folder containing tf records.
        :param suffix: In case of many different types of tf records in one folder (eg. train and validation ones)
        suffix is required to distinguish between them (despite param name, it doesn't have to be actual suffix).
        :return: List of filenames with full paths.
        """
        filenames = os.listdir(folder_path)
        if suffix:
            names = [os.path.join(folder_path, file) for file in filenames if suffix in file]
            if not names:
                raise Exception("Cannot match suffix '%s' to any .tfrecord!" % suffix)
            return names
        else:
            return [os.path.join(folder_path, file) for file in filenames]

    def num_batches(self, type, batch_size):
        """
        Returns number of available batches for data. All the computations are done only once - later they are being read from disc.
        If type is not available (for eg when only one type of tfrecords were recreated) it replaces only that one in pickle
        :param tf_records_path: path to tfrecords
        :param batch_stats_path: path to pickle with statistics
        :param batch_size: batch size is different for classification and detection, hence this param
        :param type: "train", "validation", "classification"
        :return:
        """
        possible_keys = ["train", "validation", "classification"]
        suffixes = ["train", "validation", None]
        tf_records_paths = [self.detection_tfrecords_path, self.detection_tfrecords_path,
                            self.classification_tfrecords_path]
        # if loaded from disc and cached in RAM
        if self.batch_stats and type in self.batch_stats.keys():
            return self.batch_stats[type] // batch_size

        # if not cached in RAM, try to load stats from disc
        else:
            if os.path.isfile(self.batch_stats_path):
                self.batch_stats = pickle.load(open(self.batch_stats_path, 'rb'))
            else:
                self.batch_stats = {}

            # if type available in loaded file
            if type in self.batch_stats.keys():
                return self.batch_stats[type] // batch_size

            # if type not available in loaded file, load it and update batch_stats
            else:
                idx = possible_keys.index(type)
                tf_records_path = tf_records_paths[possible_keys.index(type)]

                print("Need to calculate '%s' length - it might take some time" % type)
                filenames = self.tf_record_filenames(tf_records_path, suffixes[idx])
                count = sum(sum(1 for record in tf.python_io.tf_record_iterator(name)) for name in filenames)
                self.batch_stats[type] = count
                pickle.dump(self.batch_stats, open(self.batch_stats_path, 'wb'))
                return count // batch_size

    def make_dirs(self):
        """
        Creates all necesarry directories
        :return:
        """
        self.possibly_create_root_path()

        needed_folders = ['detection_images', 'detection_annotations', 'classification_images', 'detection_labels',
                          'detection_tfrecords', 'classification_tfrecords']

        for folder in needed_folders:
            if not os.path.isdir(os.path.join(os.path.join(self.data_root_path, folder))):
                os.mkdir(os.path.join(self.data_root_path, folder))

        self.detection_images_path = os.path.join(self.data_root_path, 'detection_images')
        self.detection_annotations_path = os.path.join(self.data_root_path, 'detection_annotations')
        self.classification_images_path = os.path.join(self.data_root_path, 'classification_images')
        self.detection_labels_path = os.path.join(self.data_root_path, 'detection_labels')
        self.detection_tfrecords_path = os.path.join(self.data_root_path, 'detection_tfrecords')
        self.classification_tfrecords_path = os.path.join(self.data_root_path, 'classification_tfrecords')
        self.batch_stats_path = os.path.join(self.data_root_path, 'batch_stats.p')

    def possibly_create_root_path(self):
        splitted_path = self.data_root_path.split('/')
        paths = ['/'.join(splitted_path[:(i + 1)]) for i in range(len(splitted_path))]
        for path in paths:
            if not os.path.isdir(path):
                os.mkdir(path)

    def data_distribution(self, filenames, name_converter):
        """
        Computes distribution of data. It helps in later data upsampling (some classes are strongly undersampled).
        It extracts classes from xml instead of names of images - in case of problems with names that are not made from
        classes
        :param name_converter: dictionary that helps to convert one set of names into uniform set (pl -> en, wnid -> eng)
        :param filenames: list of names (with full paths) to xml files
        :return: distribution - dict of distributions (computed per file, because each file contains only one class, filenames_by_class - filenames with
        their classes - it helps in later upsamling. Those names are deprived of file extensions in order to match both
        images and labels (numpy tensors)
        """

        def name_from_xml(filename):
            tree = ET.parse(filename)
            size = tree.find('size')
            width = int(size.find('width').text)
            height = int(size.find('height').text)
            if height == 0 or width == 0:
                raise Exception
            objs = tree.findall('object')

            # get only first name, because they are all the same per image
            return name_converter[objs[0].find('name').text]

        filenames_by_class = {}
        for name in filenames:
            try:
                cls = name_from_xml(name)
            except Exception:
                continue
            if cls in filenames_by_class:
                filenames_by_class[cls].append(name)
            else:
                filenames_by_class[cls] = [name]

        distribution = dict(zip(filenames_by_class.keys(), [len(value) for value in filenames_by_class.values()]))
        return distribution, filenames_by_class

    def image_read(self, imgname, img_size):
        """
        Reads image from disc and preprocess it (resizing)
        :param imgname:
        :param img_size:
        :return:
        """
        image = cv2.imread(imgname)
        image = cv2.resize(image, (img_size, img_size))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
        image = (image / 255.0) * 2.0 - 1.0
        return image

    def create_record_writer(self, writer_path, record_name):
        """
        Creates tfrecord writer in writer_path and assigns a name to it
        """
        writer = tf.python_io.TFRecordWriter(os.path.join(writer_path, record_name + '.tfrecord'))
        return writer

    @staticmethod
    def _bytes_feature(value):
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

    @staticmethod
    def _int64_feature(value):
        return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

    def decode_detection_data(self, batch_size, mode, capacity, num_threads, min_after_deque):
        """
        Decodes detection tf records on the fly.
        capacity, num_threads, min_after_deque - for shuffle batch
        :param batch_size:
        :param mode: 'train' or 'validation'
        :return:
        """
        if not (mode == 'train' or mode == 'validation'):
            raise Exception("Mode %s is not available! Try 'train' or 'validation'" % mode)

        feature = {mode + '/image': tf.FixedLenFeature([], tf.string),
                   mode + '/label': tf.FixedLenFeature([], tf.string)}

        train_names = self.tf_record_filenames(self.detection_tfrecords_path, 'train')
        validation_names = self.tf_record_filenames(self.detection_tfrecords_path, 'validation')
        filenames = train_names if mode == 'train' else validation_names

        filename_queue = tf.train.string_input_producer(filenames)
        reader = tf.TFRecordReader()
        _, serialized_example = reader.read(filename_queue)
        features = tf.parse_single_example(serialized_example, features=feature)
        image = tf.reshape(tf.decode_raw(features[mode + '/image'], tf.float32), [params.img_size, params.img_size, 3])
        label = tf.reshape(tf.decode_raw(features[mode + '/label'], tf.float32), [params.S, params.S, 5 + params.C])

        images, labels = tf.train.shuffle_batch([image, label],
                                                batch_size=batch_size,
                                                capacity=capacity,
                                                num_threads=num_threads,
                                                min_after_dequeue=min_after_deque,
                                                allow_smaller_final_batch=True)
        return images, labels

    def decode_classification_data(self, batch_size, capacity, num_threads, min_after_deque):
        """
        Decodes classification tf records on the fly.
        capacity, num_threads, min_after_deque - for shuffle batch
        :param batch_size:
        :return:
        """
        feature = {'train/image': tf.FixedLenFeature([], tf.string),
                   'train/label': tf.FixedLenFeature([], tf.int64)}

        filenames = self.tf_record_filenames(self.classification_tfrecords_path)
        filename_queue = tf.train.string_input_producer(filenames)
        reader = tf.TFRecordReader()
        _, serialized_example = reader.read(filename_queue)
        features = tf.parse_single_example(serialized_example, features=feature)
        image = tf.reshape(tf.decode_raw(features['train/image'], tf.float32), [params.img_size, params.img_size, 3])
        label = tf.cast(features['train/label'], tf.int32)

        images, labels = tf.train.shuffle_batch([image, label],
                                                batch_size=batch_size,
                                                capacity=capacity,
                                                num_threads=num_threads,
                                                min_after_dequeue=min_after_deque,
                                                allow_smaller_final_batch=True)
        return images, labels

    def tensor_label(self, filename, name_converter, classes):
        """
        Parses single xml and creates numpy tensor as label
        :param name_converter: dict to convert from one namespace to another
        (pl -> en, wnid -> en)
        :param classes: list with classes to use. Might be a mix of many datasets -implementations should take care of it
        :return: numpy tensor as label
        """
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

            name = name_converter[obj.find('name').text.lower().strip()]
            index = classes.index(name)

            boxes = [(x2 + x1) / 2.0, (y2 + y1) / 2.0, x2 - x1, y2 - y1]

            x_ind = int(boxes[0] * params.S / params.img_size)
            y_ind = int(boxes[1] * params.S / params.img_size)

            if label[y_ind, x_ind, 0] == 1:
                continue
            label[y_ind, x_ind, 0] = 1
            label[y_ind, x_ind, 1:5] = boxes
            label[y_ind, x_ind, 5 + index] = 1
        return label

    def prepare_valid_data(self, name_converter, classes):
        """
        Prepares data - extracts only images with annotations etc - after calling this method, data MUST be ready
        to .tfrecord conversion. Implementation strongly depends on dataset.
        :param name_converter: dictionary that helps to convert one set of names into uniform set (pl -> en, wnid -> eng)
        :param classes: list with classes to use. Might be a mix of many datasets -implementations should take care of it
        :return train_image_names, train_label_names (with paths)
        """
        raise NotImplementedError

    def generate_classification_tfrecords(self, size_limit=None):
        """
        Converts classification data to tfrecords
        :param size_limit: limit of examples in single tf record. Unlimited if none
        :return:
        """
        raise NotImplementedError

    def generate_detection_tfrecords(self, size_limit=None):
        """
        Converts detection data to tfrecords
        :param size_limit: limit of examples in single tf record. Unlimited if none
        :return:
        """
        raise NotImplementedError

    def download_data(self):
        """
        Possibly downloads and extracts data.
        """
        raise NotImplementedError

    def get_saved_names(self):
        """
        Returns precomputed filenames of train/validation data
        :return:
        """
        raise NotImplementedError
