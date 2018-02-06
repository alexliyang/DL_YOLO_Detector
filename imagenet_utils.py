import os
import tarfile

import params


def make_dirs(imagenet_folder='imagenet_data'):
    imagenet_dir = imagenet_folder
    needed_dirs = ['annotations', 'images', 'tensor_annotations', 'tfrecords', 'classification_images']
    curr_dirs = os.listdir(imagenet_dir)
    for dir in needed_dirs:
        if not dir in curr_dirs:
            os.mkdir(os.path.join(imagenet_dir, dir))


def extract_localization_data(tars_path, annotations_path, images_path):
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
        common_files = sorted(list(set([name.split('/')[-1].replace('.xml', '') for name in valid_xmls]).intersection(
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

    # class_distribution = dict(zip(class_names, class_count))
    # print(class_distribution)


def extract_classification_data(tars_path, images_path):
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

    # class_distribution = dict(zip(class_names, class_count))
    # print(class_distribution)


def rename_localization_data(annotations_path, images_path):
    a_files = sorted(os.listdir(annotations_path))
    i_files = sorted(os.listdir(images_path))

    c = 0
    for a_file, i_file in zip(a_files, i_files):
        os.rename(os.path.join(annotations_path, a_file), os.path.join(annotations_path, params.imagenet_dictionary[
            a_file.split('_')[0]] + '_' + str(c) + '.xml'))
        os.rename(os.path.join(images_path, i_file), os.path.join(images_path, params.imagenet_dictionary[
            i_file.split('_')[0]] + '_' + str(c) + '.jpg'))
        c += 1


def rename_classification_data(images_path):
    i_files = sorted(os.listdir(images_path))
    c = 0
    for i_file in i_files:
        os.rename(os.path.join(images_path, i_file), os.path.join(images_path, params.imagenet_dictionary[
            i_file.split('_')[0]] + '_' + str(c) + '.jpg'))
        c += 1

# print('Extracting data from .tar files')
# make_dirs()
# extract_localization_data(tars_path='imagenet_data/tars',
#                           annotations_path='imagenet_data/annotations',
#                           images_path='imagenet_data/images')
#
# rename_localization_data(annotations_path='imagenet_data/annotations', images_path='imagenet_data/images')

# extract_classification_data(tars_path='imagenet_data/tars',
#                             images_path='imagenet_data/classification_images')
# rename_classification_data(images_path='imagenet_data/classification_images')
