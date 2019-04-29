import os
import cv2

import process_file as file
import process_img as pi

label_root = 'D:\\datasets\\PASCAL\\VOCdevkit\\VOC2012\\Annotations'
img_root = 'D:\\datasets\\PASCAL\\VOCdevkit\\VOC2012\\JPEGImages'

for f in os.listdir(label_root):
    label_path = os.path.join(label_root, f)
    img_path = os.path.join(img_root, f[:-3] + 'jpg')

    xml_img_info, xml_obj = file.read_xml(label_path)
    img = cv2.imread(img_path)

    pi.label(img, xml_obj)
