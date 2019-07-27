#-*- coding:utf-8 -*-

from __future__ import division
from __future__ import absolute_import
from __future__ import print_function


import os
import pandas as pd
from tqdm import tqdm
import numpy as np

def process_wider_face():
    WIDER_ROOT = './WIDER'


    train_list_file = os.path.join(WIDER_ROOT, 'wider_face_split',
                                   'wider_face_train_bbx_gt.txt')
    val_list_file = os.path.join(WIDER_ROOT, 'wider_face_split',
                                 'wider_face_val_bbx_gt.txt')

    WIDER_TRAIN = os.path.join(WIDER_ROOT, 'WIDER_train', 'images')
    WIDER_VAL = os.path.join(WIDER_ROOT, 'WIDER_val', 'images')


    def parse_wider_file(root, file):
        with open(file, 'r') as fr:
            lines = fr.readlines()
        face_count = []
        img_paths = []
        face_loc = []
        img_faces = []
        count = 0
        flag = False
        for k, line in enumerate(lines):
            line = line.strip().strip('\n')
            if count > 0:
                line = line.split(' ')
                count -= 1
                loc = [int(line[0]), int(line[1]), int(line[2]), int(line[3])]
                face_loc += [loc]
            if flag:
                face_count += [int(line)]
                flag = False
                count = int(line)
            if 'jpg' in line:
                img_paths += [os.path.join(root, line)]
                flag = True

        total_face = 0
        for k in face_count:
            face_ = []
            for x in range(total_face, total_face + k):
                face_.append(face_loc[x])
            img_faces += [face_]
            total_face += k
        return img_paths, img_faces


    def wider_data_file():
        img_paths, bbox = parse_wider_file(WIDER_TRAIN, train_list_file)
        fw = open('train.txt', 'w')
        for index in range(len(img_paths)):
            tmp_str = ''
            tmp_str =tmp_str+ img_paths[index]+'|'
            boxes = bbox[index]

            for box in boxes:
                data = ' %d,%d,%d,%d,1'%(box[0], box[1], box[0]+box[2],  box[1]+box[3])
                tmp_str=tmp_str+data
            if len(boxes) == 0:
                print(tmp_str)
                continue
            ####err box?
            if box[2] <= 0 or box[3] <= 0:
                pass
            else:
                fw.write(tmp_str + '\n')
        fw.close()

        img_paths, bbox = parse_wider_file(WIDER_VAL, val_list_file)
        fw = open('train.txt', 'a')
        for index in range(len(img_paths)):

            tmp_str=''
            tmp_str =tmp_str+ img_paths[index]+'|'
            boxes = bbox[index]

            for box in boxes:
                data = ' %d,%d,%d,%d,1'%(box[0], box[1], box[0]+box[2],  box[1]+box[3])
                tmp_str=tmp_str+data



            if len(boxes) == 0:
                print(tmp_str)
                continue
            ####err box?
            if box[2] <= 0 or box[3] <= 0:
                pass
            else:
                fw.write(tmp_str + '\n')
        fw.close()

    wider_data_file()


def process_fddb_face():
    FDDB_ROOT='./FDDB'

    IMAGES_DIR = os.path.join(FDDB_ROOT,'img')
    BOXES_DIR = os.path.join(FDDB_ROOT,'FDDB-folds')
    # collect paths to all images

    all_paths = []
    for path, subdirs, files in tqdm(os.walk(IMAGES_DIR)):
        for name in files:

            all_paths.append(os.path.join(path, name))


    annotation_files = os.listdir(BOXES_DIR)
    annotation_files = [f for f in annotation_files if f.endswith('ellipseList.txt')]
    annotation_files = [os.path.join(BOXES_DIR, f) for f in annotation_files]


    def ellipse_to_box(major_axis_radius, minor_axis_radius, angle, center_x, center_y):
        half_h = major_axis_radius * np.sin(-angle)
        half_w = minor_axis_radius * np.sin(-angle)
        xmin, xmax = center_x - half_w, center_x + half_w
        ymin, ymax = center_y - half_h, center_y + half_h
        return xmin, ymin, xmax, ymax

    def get_boxes(path):
        with open(path, 'r') as f:
            content = f.readlines()
            content = [s.strip() for s in content]

        boxes = {}
        num_lines = len(content)
        i = 0
        name = None

        while i < num_lines:
            s = content[i]
            if 'big/img' in s:
                if name is not None:
                    assert len(boxes[name]) == num_boxes
                name = s + '.jpg'
                boxes[name] = []
                i += 1
                num_boxes = int(content[i])
                i += 1
            else:
                numbers = [float(f) for f in s.split(' ')[:5]]
                major_axis_radius, minor_axis_radius, angle, center_x, center_y = numbers

                xmin, ymin, xmax, ymax = ellipse_to_box(
                    major_axis_radius, minor_axis_radius,
                    angle, center_x, center_y
                )
                if xmin == xmax or ymin == ymax:
                    num_boxes -= 1
                else:
                    boxes[name].append((
                        min(xmin, xmax), min(ymin, ymax),
                        max(xmin, xmax), max(ymin, ymax)
                    ))
                i += 1
        return boxes

    boxes = {}
    for p in annotation_files:
        boxes.update(get_boxes(p))

    # check number of images with annotations
    # and number of boxes
    # (these values are taken from the official website)
    assert len(boxes) == 2845
    assert sum(len(b) for b in boxes.values()) == 5171 - 1  # one box is empty

    fw = open('val.txt', 'w')
    for k,v in boxes.items():

        tmp_str = IMAGES_DIR+'/'+ k+'|'

        boxes = v

        for box in v:
            data = ' %d,%d,%d,%d,1'%(box[0], box[1], box[2],  box[3])
            tmp_str=tmp_str+data
        if len(boxes) == 0:
            print(tmp_str)
            continue
        ####err box?

        fw.write(tmp_str + '\n')


    fw.close()


if __name__ == '__main__':
    process_wider_face()
    process_fddb_face()