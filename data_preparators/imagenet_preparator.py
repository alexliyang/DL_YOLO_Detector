import os

import numpy as np
import tarfile
import params
from data_preparators.data_preparator import DataPreparator


class ImagenetPreparator(DataPreparator):
    def download_data(self):
        """
        Possibly downloads and extracts data.
        """
        if not os.path.isdir(os.path.join(self.data_root_path, 'tars')):
            print('ImageNet data needs to be downloaded. Please be patient (file is about 2.4GB)')
            self.download_file_from_google_drive('1rkqYfK378ixwvEmHUjB_tXU8kua3DwKm', self.data_root_path + '/tars.tar.gz')
            print('ImageNet data downloaded, extracting..')
            with tarfile.open(self.data_root_path + '/tars.tar.gz') as tar:
                tar.extractall(self.data_root_path)
            print('Cleaning unnecesary files')
            os.remove(self.data_root_path + '/tars.tar.gz')

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

        return sorted([os.path.join(self.detection_images_path, name) for name in os.listdir(self.detection_images_path)]), \
               sorted([os.path.join(self.detection_labels_path, name) for name in os.listdir(self.detection_labels_path)])


p = ImagenetPreparator(data_root_path=params.imagenet_root_path,
                       data_url=params.imagenet_data_url,
                       classes=params.imagenet_classes)
