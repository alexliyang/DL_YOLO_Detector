import os
import pickle

import tensorflow as tf


class DataPreparatorBase:
    def __init__(self):
        self.batch_stats = None

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


p = DataPreparatorBase()
