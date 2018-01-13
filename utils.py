import xml.etree.ElementTree as ET
import numpy as np
import cv2


class DataPreparator:
    """
    Prepares data: converts xmls with PASCAL VOC annotations into numpy matrices, then to TFRecords
    """

    def __init__(self, S: int):
        """
        :param S: image will be split into SxS subwindows
        """
        self.classes = ('circle', 'square', 'side_rect', 'up_rect')
        self.S = S

    def parse_xml(self, path):
        """
        Converts annotated bounding boxes to general, relative format
        :param path:
        :return:
        """
        annotations = {}
        tree = ET.parse(path)
        root = tree.getroot()
        size = root.find('size')
        width = int(size.find('width').text)
        height = int(size.find('height').text)
        annotations['dimensions'] = {'width': width,
                                     'height': height}

        bounding_boxes = []
        for obj in root.findall('object'):
            bndbox = obj.find('bndbox')
            bounding_box = {'class': self.classes.index(obj.find('name').text),
                            'xmin': int(bndbox.find('xmin').text) / width,
                            'xmax': int(bndbox.find('xmax').text) / width,
                            'ymin': int(bndbox.find('ymin').text) / height,
                            'ymax': int(bndbox.find('ymax').text) / height}
            bounding_boxes.append(bounding_box)

        annotations['bounding_boxes'] = bounding_boxes
        return annotations

    def presence_matrix(self, annotations):
        """
        Checks, wheter Sij subwindows contains any object
        :return: matrix SxS with ones, where subwindow contains any object
        """
        presence_matrix = np.zeros(shape=(self.S, self.S))
        for bounding_box in annotations['bounding_boxes']:
            xmin = bounding_box['xmin']
            xmax = bounding_box['xmax']
            ymin = bounding_box['ymin']
            ymax = bounding_box['ymax']

            # divides image sizes (relative, from 0 to 1) to S steps and checks, which bins are occupied by xmin etc.
            x_min_bins = [1 if xmin < x_step else 0 for x_step in [(i + 1) * 1 / self.S for i in range(self.S)]]
            x_max_bins = [1 if x_step < xmax else 0 for x_step in [i * 1 / self.S for i in range(self.S)]]
            y_min_bins = [1 if ymin < y_step else 0 for y_step in [(i + 1) * 1 / self.S for i in range(self.S)]]
            y_max_bins = [1 if y_step < ymax else 0 for y_step in [i * 1 / self.S for i in range(self.S)]]

            x_bins = np.logical_and(x_min_bins, x_max_bins).astype(np.uint8)
            y_bins = np.logical_and(y_min_bins, y_max_bins).astype(np.uint8)

            # finds first occurences of ones in occupation lists
            x_min_bin = np.where(x_bins == x_bins.max())[0][0]
            x_max_bin = np.where(x_bins == x_bins.max())[0][-1]
            y_min_bin = np.where(y_bins == y_bins.max())[0][0]
            y_max_bin = np.where(y_bins == y_bins.max())[0][-1]

            presence_matrix[y_min_bin:y_max_bin+1, x_min_bin:x_max_bin+1] = 1

        return presence_matrix

    def visualize_objects_presence(self, img, presence_matrix):
        """
        embedds presence matrix into original image - draws green rectangles, where object is present
        and red, where object is not present
        :return: original image with rectangles drawn onto it
        """
        x_step = img.shape[1] / self.S
        y_step = img.shape[0] / self.S

        alpha_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)

        for y in range(self.S):
            for x in range(self.S):
                if presence_matrix[x,y] == 1:
                    cv2.rectangle(alpha_img, (int(y * y_step), int(x * x_step)), (int((y + 1) * y_step-1), int((x + 1) * x_step-1)),
                                  color=(0,255,0), thickness=4)
                else:
                    cv2.rectangle(alpha_img, (int(y * y_step), int(x * x_step)),
                                  (int((y + 1) * y_step - 1), int((x + 1) * x_step - 1)),
                                  color=(0, 0, 255), thickness=1)

        return cv2.addWeighted(alpha_img, 0.3, img, 1, 0)


