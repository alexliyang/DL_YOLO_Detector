import os
import tarfile

import numpy as np
import tensorflow as tf
from sklearn.utils import shuffle
from utils import download_file_from_google_drive
from data_preparators.data_preparator import DataPreparator
from parameters import params


class ImagenetPreparator(DataPreparator):
    def download_data(self):
        """
        Possibly downloads and extracts data.
        """
        extracted_tars_path = os.path.join(self.data_root_path, 'tars')
        tars_path = os.path.join(self.data_root_path, 'tars.tar.gz')
        if not os.path.isdir(tars_path):
            print('ImageNet data needs to be downloaded. Please be patient (file is about 2.4GB)')
            download_file_from_google_drive('1rkqYfK378ixwvEmHUjB_tXU8kua3DwKm', tars_path)
            print('ImageNet data downloaded, extracting..')
            with tarfile.open(tars_path) as tar:
                tar.extractall(self.data_root_path)
            print('Cleaning unnecesary files')
            os.remove(tars_path)

        if len(os.listdir(self.detection_images_path)) == 0 or len(os.listdir(self.detection_annotations_path)) == 0:
            print('Extracting detection data from .tar files')
            self.extract_localization_data(tars_path=extracted_tars_path,
                                           annotations_path=self.detection_annotations_path,
                                           images_path=self.detection_images_path)
            self.rename_localization_data(annotations_path=self.detection_annotations_path,
                                          images_path=self.detection_images_path)

        if len(os.listdir(self.classification_images_path)) == 0:
            print('Extracting classification data from .tar files')
            self.extract_classification_data(tars_path=extracted_tars_path, images_path=self.classification_images_path)
            self.rename_classification_data(images_path=self.classification_images_path)

    def prepare_valid_data(self, name_converter, classes):
        """
        Cleans images and annotations, then creates numpy tensors as labels
        :return: filenames of images and labels (with paths)
        """
        print("Preparing dataset")

        # check if these list are the same (have the same elements)
        image_names = sorted([name.replace('.jpg', '') for name in os.listdir(self.detection_images_path)])
        label_names = sorted([name.replace('.npy', '') for name in os.listdir(self.detection_labels_path)])

        # if image_names == label_names, there is no need to create label names (numpy tensor labels)
        if not image_names == label_names:
            print("Generating tensor labels")
            empty_annotations = []

            for name in image_names:
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

        return sorted(
            [os.path.join(self.detection_images_path, name) for name in os.listdir(self.detection_images_path)]), \
               sorted(
                   [os.path.join(self.detection_labels_path, name) for name in os.listdir(self.detection_labels_path)])

    def extract_localization_data(self, tars_path, annotations_path, images_path):
        """
        Extracts files from .tar files, checks, wheter each file has annotation and copies it to annotatnions/images folders
        :param tars_path: path to .tar files
        """
        # .tar filenames - only .tars containing annotations
        annotation_tars = sorted([tar for tar in os.listdir(tars_path) if '.gz' in tar])
        image_tars = sorted([tar for tar in os.listdir(tars_path) if tar + '.gz' in annotation_tars])

        class_names = []
        class_count = []
        for a_file, i_file in zip(annotation_tars, image_tars):
            a_tar = tarfile.open(os.path.join(tars_path, a_file), 'r')
            i_tar = tarfile.open(os.path.join(tars_path, i_file), 'r')

            # removes parent folders from names and everything not matching pattern
            valid_xmls = sorted([name for name in a_tar.getnames() if name.endswith('.xml')])
            valid_jpegs = sorted([name for name in i_tar.getnames() if name.endswith('.JPEG')])

            # find only images with annotations
            common_files = sorted(
                list(set([name.split('/')[-1].replace('.xml', '') for name in valid_xmls]).intersection(
                    [name.split('/')[-1].replace('.JPEG', '') for name in valid_jpegs])))

            class_names.append(common_files[0].split('_')[0])
            class_count.append(len(common_files))

            # common files with valid names and paths
            a_common_files = [file + '.xml' for file in common_files]
            i_common_files = [file + '.JPEG' for file in common_files]

            # removes parent folders from nested annotations
            for member in a_tar.getmembers():
                if member.isreg():  # skip if the TarInfo is not files
                    member.name = os.path.basename(member.name)  # remove the path by reset it
            a_tar.extractall(path=annotations_path, members=[x for x in a_tar.getmembers() if x.name in a_common_files])
            i_tar.extractall(path=images_path, members=[x for x in i_tar.getmembers() if x.name in i_common_files])

            a_tar.close()
            i_tar.close()

    def extract_classification_data(self, tars_path, images_path):
        """
        Extracts files from .tar files, checks, wheter each file has annotation and copies it to annotatnions/images folders
        :param tars_path: path to .tar files
        """
        # .tar filenames - only .tars containing annotations
        image_tars = sorted([tar for tar in os.listdir(tars_path) if not '.gz' in tar])

        class_names = []
        class_count = []

        for i_file in image_tars:
            i_tar = tarfile.open(os.path.join(tars_path, i_file), 'r')

            valid_jpegs = sorted([name for name in i_tar.getnames() if name.endswith('.JPEG')])

            class_names.append(valid_jpegs[0].split('_')[0])
            class_count.append(len(valid_jpegs))

            # removes parent folders from nested annotations
            for member in i_tar.getmembers():
                if member.isreg():  # skip if the TarInfo is not files
                    member.name = os.path.basename(member.name)  # remove the path by reset it
            i_tar.extractall(path=images_path, members=[x for x in i_tar.getmembers() if x.name in valid_jpegs])
            i_tar.close()

    def rename_localization_data(self, annotations_path, images_path):
        a_files = sorted(os.listdir(annotations_path))
        i_files = sorted(os.listdir(images_path))

        c = 0
        for a_file, i_file in zip(a_files, i_files):
            os.rename(os.path.join(annotations_path, a_file), os.path.join(annotations_path, params.name_converter[
                a_file.split('_')[0]] + '_' + str(c) + '.xml'))
            os.rename(os.path.join(images_path, i_file), os.path.join(images_path, params.name_converter[
                i_file.split('_')[0]] + '_' + str(c) + '.jpg'))
            c += 1

    def rename_classification_data(self, images_path):
        i_files = sorted(os.listdir(images_path))
        c = 0
        for i_file in i_files:
            os.rename(os.path.join(images_path, i_file), os.path.join(images_path, params.name_converter[
                i_file.split('_')[0]] + '_' + str(c) + '.jpg'))
            c += 1

    def generate_classification_tfrecords(self, size_limit=None):
        if len(os.listdir(self.classification_tfrecords_path)) > 0:
            print("No need to generate classification TFRecords")
            return

        imnames = [os.path.join(self.classification_images_path, image) for image in
                   os.listdir(self.classification_images_path)]
        labels = [name.split('/')[-1].replace('.jpg', '') for name in imnames]
        labels = [''.join([i for i in s if not i.isdigit()])[:-1] for s in labels]
        labels = [params.classes.index(label) for label in labels]

        writer = self.create_record_writer(self.classification_tfrecords_path, 'train_0')

        for i, (imgname, label) in enumerate(zip(imnames, labels)):
            print("\rGenerating classification TFRecords (%.2f)" % (i / len(imnames)), end='', flush=True)
            img = self.image_read(imgname, params.img_size)

            if size_limit and i % size_limit == 0 and i > 0:
                writer.close()
                writer = self.create_record_writer(self.classification_tfrecords_path, 'train_' + str(int(i / size_limit)))

            feature = {'train/image': self._bytes_feature(tf.compat.as_bytes(img.tostring())),
                       'train/label': self._int64_feature(label)}
            example = tf.train.Example(features=tf.train.Features(feature=feature))
            writer.write(example.SerializeToString())
        writer.close()
        print()

    def generate_detection_tfrecords(self, size_limit=None):
        if len(os.listdir(self.detection_tfrecords_path)) > 0:
            print("No need to generate detection TFRecords")
            return
        self.image_names, self.label_names = shuffle(self.image_names, self.label_names)
        self.create_base_tfrecords(size_limit)
        self.upsample_base_tfrecords(size_limit)

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

    def upsample_base_tfrecords(self, size_limit=None):
        """
        Equalizes data distribution (some classes are highly undersampled)
        :param size_limit: size limit of single tf record
        """
        xml_filenames = [os.path.join(self.detection_annotations_path, name) for name in
                         os.listdir(self.detection_annotations_path)]
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