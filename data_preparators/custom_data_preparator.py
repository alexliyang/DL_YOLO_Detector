from data_preparators.data_preparator import DataPreparator
from parameters import params
from utils import download_file_from_google_drive
import os
import tarfile

class CustomDataPreparator(DataPreparator):
    def download_data(self):
        """
        Possibly downloads and extracts data.
        """
        path = os.path.join(self.data_root_path, 'custom.tar.gz')
        if not os.path.isfile(path):
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

p = CustomDataPreparator(data_root_path=params.root_path,
                         classes=params.classes,
                         name_converter=params.name_converter)
