#-*-coding:utf-8-*-
import cv2
import os
from tools.VOCxml import *
def to_xml(file_path,boxs):
    filename=file_path.rsplit('/',1)[-1]
    img_raw=cv2.imread(file_path)
    anno = GEN_VOC_Annotations(filename)
    anno.set_size(img_raw.shape[0], img_raw.shape[1], img_raw.shape[2])

    for face in boxs:

        xmin = int(face[0])
        ymin = int(face[1])
        xmax = int(face[2])
        ymax = int(face[3])

        anno.add_pic_attr('1', xmin, ymin, xmax, ymax)



    xml_path=file_path.rsplit('.',1)[0]+'.xml'
    anno.savefile(xml_path)