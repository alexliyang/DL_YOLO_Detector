from data_preparators.data_preparator import DataPreparator
from parameters import params
from utils import download_file_from_google_drive
import os
from sklearn.utils import shuffle
import tarfile
import xml.etree.ElementTree as ET
import cv2
import numpy as np
import tensorflow as tf

class CustomDataPreparator(DataPreparator):
    def download_data(self):
        """
        Possibly downloads and extracts data.
        """
        path = os.path.join(self.data_root_path, 'custom.tar.gz')
        if len(os.listdir(self.detection_images_path)) == 0 or len(os.listdir(self.detection_annotations_path)) == 0:
            print('Custom data needs to be downloaded. Please be patient (file is about 3GB)')
            download_file_from_google_drive('1UACRpbi9vvbMiF7p6du8c-532woNExGf', path)
            print('ImageNet data downloaded, extracting..')
            with tarfile.open(path) as tar:
                tar.extractall(self.data_root_path)
            print('Cleaning unnecesary files')

            # rename folders instead of copying (allows to preserve inheritance of python)
            os.rmdir(self.detection_images_path)
            os.rmdir(self.detection_annotations_path)
            os.rename(os.path.join(self.data_root_path, 'images'), self.detection_images_path)
            os.rename(os.path.join(self.data_root_path, 'annotations'), self.detection_annotations_path)
            os.remove(path)

    def prepare_valid_data(self, name_converter, classes):
        print("Preparing dataset")

        image_names = sorted([name.rstrip('.jpg') for name in os.listdir(self.detection_images_path)])
        annotation_names = sorted([name.rstrip('.xml') for name in os.listdir(self.detection_annotations_path)])
        label_names = sorted([name.rstrip('.npy') for name in os.listdir(self.detection_labels_path)])

        if not image_names == label_names:
            print("Generating tensor labels")
            common_names = set(image_names).intersection(annotation_names)
            empty_annotations = list(set(image_names) - common_names)
            for name in common_names:
                try:
                    label = self.tensor_label(os.path.join(self.detection_annotations_path, name + '.xml'),
                                              name_converter, classes)
                    if np.max(label) == 0.0:
                        empty_annotations.append(name)
                except Exception:
                    empty_annotations.append(name)
                else:
                    np.save(os.path.join(self.detection_labels_path, name), label)

                for name in empty_annotations:
                    if os.path.isfile(os.path.join(self.detection_images_path, name + '.jpg')):
                        os.remove(os.path.join(self.detection_images_path, name + '.jpg'))
                    # in case or other extensions, like png
                    if os.path.isfile(os.path.join(self.detection_images_path, name)):
                        os.remove(os.path.join(self.detection_images_path, name))
        return sorted([os.path.join(self.detection_images_path, name) for name in os.listdir(self.detection_images_path)]), sorted(
            [os.path.join(self.detection_labels_path, name) for name in os.listdir(self.detection_labels_path)])

    def generate_detection_tfrecords(self, size_limit=None):
        if len(os.listdir(self.detection_tfrecords_path)) > 0:
            print("No need to generate detection TFRecords")
            return

        self.image_names, self.label_names = shuffle(self.image_names, self.label_names)
        self.create_base_tfrecords(size_limit)
        self.upsample_base_tfrecords(size_limit)

    def upsample_base_tfrecords(self, size_limit=None):
        """
        Equalizes data distribution (some classes are highly undersampled)
        :param size_limit: size limit of single tf record
        """

        xml_filenames = [os.path.join(self.detection_annotations_path, name).replace('jpg', 'xml') for name in
                         os.listdir(self.detection_images_path)]


        distribution, names_by_classes = self.data_distribution(xml_filenames, params.name_converter)
        max_count = max(distribution.values())
        all_to_create = sum([max_count - val for val in distribution.values()])

        i = 0
        for (key, dist), names in zip(distribution.items(), names_by_classes.values()):
            to_create = max_count - dist
            names = np.random.choice(names, to_create)
            names = [name.split('/')[-1].rstrip('.xml') for name in names]
            image_names = [os.path.join(self.detection_images_path, name + '.jpg') for name in names]
            label_names = [os.path.join(self.detection_labels_path, name + '.npy') for name in names]
            train_i = 0
            val_i = 0
            train_writer = self.create_record_writer(self.detection_tfrecords_path, '_train_0')
            val_writer = self.create_record_writer(self.detection_tfrecords_path, '_validation_0')


            for imgname, label in zip(image_names, label_names):
                print("\rUpdating TFRecords (%.2f)" % (i / all_to_create), end='', flush=True)

                img = self.image_read(imgname, params.img_size)
                img *= np.random.uniform(params.augmentation_noise_low, params.augmentation_noise_high)
                lbl = np.load(label).astype(np.float32)
                lbl[:, :, 1:5] *= np.random.uniform(params.augmentation_noise_low, params.augmentation_noise_high)
                e = np.random.uniform(0, 1)
                if e < self.train_ratio:
                    train_i += 1

                    if size_limit and train_i % size_limit == 0 and train_i > 0:
                        train_writer.close()
                        train_writer = self.create_record_writer(self.detection_tfrecords_path,
                                                                 '_train_' + str(int(train_i / size_limit)))

                    feature = {'train/label': self._bytes_feature(tf.compat.as_bytes(lbl.tostring())),
                               'train/image': self._bytes_feature(tf.compat.as_bytes(img.tostring()))}
                    example = tf.train.Example(features=tf.train.Features(feature=feature))
                    train_writer.write(example.SerializeToString())
                else:
                    val_i += 1
                    if size_limit and val_i % size_limit == 0 and val_i > 0:
                        val_writer.close()
                        val_writer = self.create_record_writer(self.detection_tfrecords_path,
                                                               'validation_' + str(int(val_i / size_limit)))

                    feature = {'validation/label': self._bytes_feature(tf.compat.as_bytes(lbl.tostring())),
                               'validation/image': self._bytes_feature(tf.compat.as_bytes(img.tostring()))}
                    example = tf.train.Example(features=tf.train.Features(feature=feature))
                    val_writer.write(example.SerializeToString())
                i += 1
            train_writer.close()
            val_writer.close()
        print()

    def create_base_tfrecords(self, size_limit=None):
        """
        Creates detection tf records without resampling
        :param size_limit: size limit of single tf record
        """
        train_i = 0
        val_i = 0
        train_writer = self.create_record_writer(self.detection_tfrecords_path, 'train_0')
        val_writer = self.create_record_writer(self.detection_tfrecords_path, 'validation_0')

        for i, (image_name, label_name) in enumerate(zip(self.image_names, self.label_names)):
            print("\rGenerating TFRecords (%.2f)" % (i / len(self.image_names)), end='', flush=True)
            image = self.image_read(image_name, params.img_size)
            label = np.load(label_name).astype(np.float32)
            e = np.random.uniform(0, 1)
            if e < self.train_ratio:
                train_i += 1
                if size_limit and train_i % size_limit == 0 and train_i > 0:
                    train_writer.close()
                    train_writer = self.create_record_writer(self.detection_tfrecords_path,
                                                             'train_' + str(int(train_i / size_limit)))

                feature = {'train/label': self._bytes_feature(tf.compat.as_bytes(label.tostring())),
                           'train/image': self._bytes_feature(tf.compat.as_bytes(image.tostring()))}
                example = tf.train.Example(features=tf.train.Features(feature=feature))
                train_writer.write(example.SerializeToString())
            else:
                val_i += 1
                if size_limit and val_i % size_limit == 0 and val_i > 0:
                    val_writer.close()
                    val_writer = self.create_record_writer(self.detection_tfrecords_path,
                                                           'validation_' + str(int(val_i / size_limit)))

                feature = {'validation/label': self._bytes_feature(tf.compat.as_bytes(label.tostring())),
                           'validation/image': self._bytes_feature(tf.compat.as_bytes(image.tostring()))}
                example = tf.train.Example(features=tf.train.Features(feature=feature))
                val_writer.write(example.SerializeToString())
        train_writer.close()
        val_writer.close()
        print()

    def generate_classification_tfrecords(self, size_limit=None):
        if len(os.listdir(self.classification_tfrecords_path)) > 0:
            print("No need to generate classification TFRecords")
            return

        xmls = [name.replace(self.detection_images_path, self.detection_annotations_path).replace('.jpg', '.xml') for name in self.image_names]
        writer = self.create_record_writer(self.classification_tfrecords_path, 'train_0')

        for i, (img_name, annotation_name) in enumerate(zip(self.image_names, xmls)):
            print("\rGenerating classification data (%.2f)" % (i / len(self.image_names)), end='', flush=True)
            img = cv2.imread(img_name)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)
            img = (img / 255.0) * 2.0 - 1.0

            if size_limit and i % size_limit == 0 and i > 0:
                writer.close()
                writer = self.create_record_writer(self.classification_tfrecords_path, 'train_' + str(int(i / size_limit)))

            tree = ET.parse(annotation_name)
            size = tree.find('size')
            width = int(size.find('width').text)
            height = int(size.find('height').text)
            if height == 0 or width == 0:
                raise Exception

            objs = tree.findall('object')
            for obj in objs:
                bbox = obj.find('bndbox')
                x1 = int(bbox.find('xmin').text) - 1
                y1 = int(bbox.find('ymin').text) - 1
                x2 = int(bbox.find('xmax').text)
                y2 = int(bbox.find('ymax').text)
                name = obj.find('name').text.lower().strip()
                idx = params.classes.index(params.name_converter[name])
                ROI = cv2.resize(img[y1:y2, x1:x2], dsize=(params.img_size, params.img_size))
                ROI = ROI.astype(np.float32)
                feature = {'train/image': self._bytes_feature(tf.compat.as_bytes(ROI.tostring())),
                           'train/label': self._int64_feature(idx)}

                example = tf.train.Example(features=tf.train.Features(feature=feature))
                writer.write(example.SerializeToString())
        writer.close()
        print()