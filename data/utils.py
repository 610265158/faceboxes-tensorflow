#-*-coding:utf-8-*-

import pickle
import tensorflow as tf
import numpy as np
import cv2
import random
from functools import partial
import copy

from helper.logger import logger
from data.datainfo import data_info
from data.augmentor.augmentation import Pixel_jitter,Fill_img,Random_contrast,Random_brightness,Random_scale_withbbox,Random_flip,Blur_aug
from net.facebox.training_target_creation import get_training_targets
from train_config import config as cfg

from tensorpack.dataflow import BatchData, MultiThreadMapData, PrefetchDataZMQ,DataFromList
def balance(anns):
    res_anns=copy.deepcopy(anns)


    for ann in anns:
        label=ann[-1]
        label = np.array([label.split(' ')], dtype=np.float).reshape((-1, 2))
        bbox = np.array([np.min(label[:, 0]), np.min(label[:, 1]), np.max(label[:, 0]), np.max(label[:, 1])])
        bbox_width = bbox[2] - bbox[0]
        bbox_height = bbox[3] - bbox[1]

        if bbox_width<40 or bbox_height<40:
            res_anns.remove(ann)

        if np.sqrt(np.square(label[37,0]-label[41,0])+np.square(label[37,1]-label[41,1]))/bbox_height<0.02 \
            or np.sqrt(np.square(label[38, 0] - label[40, 0]) + np.square(label[38, 1] - label[40, 1])) / bbox_height < 0.02 \
            or np.sqrt(np.square(label[43,0]-label[47,0])+np.square(label[43,1]-label[47,1]))/bbox_height<0.02 \
            or np.sqrt(np.square(label[44, 0] - label[46, 0]) + np.square(label[44, 1] - label[46, 1])) / bbox_height < 0.02 :
            for i in range(10):
                res_anns.append(ann)
    random.shuffle(res_anns)
    logger.info('befor balance the dataset contains %d images' % (len(anns)))
    logger.info('after balanced the datasets contains %d samples' % (len(res_anns)))
    return res_anns
def get_train_data_list(im_root_path, ann_txt):
    """
    train_im_path : image folder name
    train_ann_path : coco json file name
    """
    logger.info("[x] Get data from {}".format(im_root_path))
    # data = PoseInfo(im_path, ann_path, False)
    data = data_info(im_root_path, ann_txt)
    all_samples=data.get_all_sample()

    return all_samples
def get_data_set(root_path,ana_path):
    data_list=get_train_data_list(root_path,ana_path)
    dataset= DataFromList(data_list, shuffle=True)
    return dataset


def produce_target(bboxes):
    reg_targets, matches=get_training_targets(bboxes,threshold=cfg.MODEL.MATCHING_THRESHOLD)
    return reg_targets, matches

def _data_aug_fn(image, ground_truth,is_training=True):
    """Data augmentation function."""
    ####customed here

    labels = ground_truth.split(' ')
    boxes = []
    for label in labels:
        bbox = np.array(label.split(','), dtype=np.float)
        ##the anchor need ymin,xmin,ymax,xmax
        boxes.append([bbox[1], bbox[0], bbox[3], bbox[2]])

    boxes = np.array(boxes, dtype=np.float)

    ###clip the bbox for the reason that some bboxs are beyond the image
    h_raw_limit, w_raw_limit, _ = image.shape
    boxes[:, 3] = np.clip(boxes[:, 3], 0, w_raw_limit)
    boxes[:, 2] = np.clip(boxes[:, 2], 0, h_raw_limit)
    boxes[boxes < 0] = 0
    #########random scale
    ############## becareful with this func because there is a Infinite loop in its body
    image, boxes=Random_scale_withbbox(image,boxes,target_shape=[cfg.MODEL.hin,cfg.MODEL.win],jitter=0.3)



    if is_training:
        if random.uniform(0, 1) > 0.5:
            image, boxes =Random_flip(image, boxes)
        if random.uniform(0, 1) > 0.5:
            image=Pixel_jitter(image,max_=15)
        if random.uniform(0,1)>0.5:
            image=Random_contrast(image)
        if random.uniform(0,1)>0.5:
            image=Random_brightness(image)
        # if random.uniform(0,1)>0.5:
        #     a=[3,5,7]
        #     k=random.sample(a, 1)[0]
        #     image=Blur_aug(image,ksize=(k,k))

    boxes=np.clip(boxes,0,cfg.MODEL.hin)
    ###cove the small faces
    boxes_clean=[]
    for i in range(boxes.shape[0]):
        box = boxes[i]

        if (box[3]-box[1])*(box[2]-box[0])<cfg.DATA.cover_small_face:
            image[int(box[0]):int(box[2]),int(box[1]):int(box[3]),:]=0
        else:
            boxes_clean.append(box)
    boxes=np.array(boxes_clean)
    boxes=boxes/cfg.MODEL.hin

    # for i in range(boxes.shape[0]):
    #     box=boxes[i]
    #     cv2.rectangle(image, (int(box[1]*cfg.MODEL.hin), int(box[0]*cfg.MODEL.hin)),
    #                                 (int(box[3]*cfg.MODEL.hin), int(box[2]*cfg.MODEL.hin)), (255, 0, 0), 7)

    reg_targets, matches = produce_target(boxes)

    image = image.astype(np.float32)


    return image, reg_targets, matches

def _map_fn(dp,is_training=True):
    fname, annos = dp
    image = cv2.imread(fname, cv2.IMREAD_COLOR)
    if image is None:
        print(fname)
        return
    image=cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image,label,num_bboxs=_data_aug_fn(image,annos,is_training)
    return image, label,num_bboxs




