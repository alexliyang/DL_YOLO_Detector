import os
import pickle
import xml.etree.ElementTree as ET
import tensorflow as tf


class DataPreparatorBase:
    def __init__(self):
        self.train_ratio = 0.9

        self.batch_stats = None
        self.data_root_path = None

        # self.make_dirs() # todo uncomment
        # prepare_valid_data # todo uncomment
        # distribution, filenames_by_class = self.data_distribution() # todo move it to tensor update

    def tf_record_filenames(self, path, suffix=None):
        """
        Returns a list of .tfrecord filenames.
        :param path: ath to folder containing tf records.
        :param suffix: In case of many different types of tf records in one folder (eg. train and validation ones)
        suffix is required to distinguish between them (despite param name, it doesn't have to be actual suffix).
        :return: List of filenames with full paths.
        """
        filenames = os.listdir(path)
        if suffix:
            names = [os.path.join(path, file) for file in filenames if suffix in file]
            if not names:
                raise Exception("Cannot match suffix '%s' to any .tfrecord!" % suffix)
            return names
        else:
            return [os.path.join(path, file) for file in filenames]

    def num_batches(self, type, batch_size, batch_stats_path=None, tf_records_path=None):
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
        # if loaded from disc and cached in RAM
        if self.batch_stats and type in self.batch_stats.keys():
            return self.batch_stats[type] // batch_size

        # if not cached in RAM, try to load stats from disc
        else:
            if not batch_stats_path:
                raise Exception("Path to pickle with batch stats must be specified!")

            if os.path.isfile(batch_stats_path):
                self.batch_stats = pickle.load(open(batch_stats_path, 'rb'))
            else:
                self.batch_stats = {}

            # if type available in loaded file
            if type in self.batch_stats.keys():
                return self.batch_stats[type] // batch_size

            # if type not available in loaded file, load it and update batch_stats
            else:
                if not tf_records_path:
                    raise Exception("Path to tfrecords must be specified!")

                print("Need to calculate '%s' length - it might take some time" % type)
                filenames = self.tf_record_filenames(tf_records_path, suffixes[possible_keys.index(type)])
                count = sum(sum(1 for record in tf.python_io.tf_record_iterator(name)) for name in filenames)
                self.batch_stats[type] = count
                pickle.dump(self.batch_stats, open(batch_stats_path, 'wb'))
                return count // batch_size

    def make_dirs(self, root_folder):
        """
        Creates all necesarry directories.
        :type root_folder: path to folder that is parent to all needed subfolders
        :return:
        """
        needed_folders = ['detection_images', 'detection_annotations', 'classification_images', 'detection_labels',
                          'detection_tfrecords', 'classification_tfrecords']

        for folder in needed_folders:
            if not os.path.isdir(os.path.join(os.path.join(root_folder, folder))):
                os.mkdir(os.path.join(root_folder, folder))

        self.detection_images_path = os.path.join(root_folder, 'detection_images')
        self.detection_annotations_path = os.path.join(root_folder, 'detection_annotations')
        self.classification_images_path = os.path.join(root_folder, 'classification_images')
        self.detection_labels_path = os.path.join(root_folder, 'detection_labels')
        self.detection_tfrecords_path = os.path.join(root_folder, 'detection_tfrecords')
        self.classification_tfrecords_path = os.path.join(root_folder, 'classification_tfrecords')

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
            cls = name_from_xml(filenames)
            if cls in filenames_by_class:
                filenames_by_class[cls].append(name)
            else:
                filenames_by_class[cls] = [name]

        distribution = dict(zip(filenames_by_class.keys(), [len(value) for value in filenames_by_class.values()]))
        return distribution, filenames_by_class

    def prepare_valid_data(self, classes):
        """
        Prepares data - extracts only images with annotations etc - after calling this method, data MUST be ready
        to .tfrecord conversion. Implementation strongly depends on dataset.
        :type classes: list with classes to use. Might be a mix of many datasets -implementations should take care of it
        """
        raise NotImplementedError


p = DataPreparatorBase()
