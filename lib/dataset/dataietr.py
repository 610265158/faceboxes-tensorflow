


import os
import random
import cv2
import numpy as np
from functools import partial

from lib.helper.logger import logger
from tensorpack.dataflow import DataFromGenerator
from tensorpack.dataflow import BatchData, MultiProcessPrefetchData


from lib.dataset.augmentor.augmentation import ColorDistort,\
    Random_scale_withbbox,\
    Random_flip,\
    Fill_img,\
    Gray_aug,\
    baidu_aug,\
    dsfd_aug,\
    Pixel_jitter
from lib.core.model.facebox.training_target_creation import get_training_targets
from train_config import config as cfg


class data_info(object):
    def __init__(self,img_root,txt):
        self.txt_file=txt
        self.root_path = img_root
        self.metas=[]


        self.read_txt()

    def read_txt(self):
        with open(self.txt_file) as _f:
            txt_lines=_f.readlines()
        txt_lines.sort()
        for line in txt_lines:
            line=line.rstrip()

            _img_path=line.rsplit('| ',1)[0]
            _label=line.rsplit('| ',1)[-1]

            current_img_path=os.path.join(self.root_path,_img_path)
            current_img_label=_label
            self.metas.append([current_img_path,current_img_label])

            ###some change can be made here
        logger.info('the dataset contains %d images'%(len(txt_lines)))
        logger.info('the datasets contains %d samples'%(len(self.metas)))


    def get_all_sample(self):
        random.shuffle(self.metas)
        return self.metas

class FaceBoxesDataIter():
    def __init__(self,img_root_path='',ann_file=None,training_flag=True, shuffle=True):

        self.color_augmentor = ColorDistort()

        self.training_flag=training_flag

        self.lst=self.parse_file(img_root_path,ann_file)

        self.shuffle=shuffle

    def __iter__(self):
        idxs = np.arange(len(self.lst))

        while True:
            if self.shuffle:
                np.random.shuffle(idxs)
            for k in idxs:
                yield self._map_func(self.lst[k], self.training_flag)



    def parse_file(self,im_root_path,ann_file):
        '''
        :return:
        '''
        logger.info("[x] Get dataset from {}".format(im_root_path))

        ann_info = data_info(im_root_path, ann_file)
        all_samples = ann_info.get_all_sample()

        return all_samples



    def _map_func(self,dp,is_training):
        fname, annos = dp
        image = cv2.imread(fname, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        labels = annos.split(' ')
        boxes = []
        for label in labels:
            bbox = np.array(label.split(','), dtype=np.float)
            boxes.append([bbox[0], bbox[1], bbox[2], bbox[3], 1])

        boxes = np.array(boxes, dtype=np.float)


        #########random scale
        ############## becareful with this func because there is a Infinite loop in its body
        if is_training:

            random_index=random.uniform(0, 1)
            if random_index>0.7:
                image, boxes = Random_scale_withbbox(image, boxes, target_shape=[cfg.MODEL.hin, cfg.MODEL.win],
                                                     jitter=0.3)
            elif random_index<0.3 and random_index<=0.7:
                boxes_ = boxes[:, 0:4]
                klass_ = boxes[:, 4:]
                image, boxes_, klass_ = baidu_aug(image, boxes_, klass_)

                image = image.astype(np.uint8)
                boxes = np.concatenate([boxes_, klass_], axis=1)
            else:
                boxes_ = boxes[:, 0:4]
                klass_ = boxes[:, 4:]
                image, boxes_, klass_ = dsfd_aug(image, boxes_, klass_)

                image = image.astype(np.uint8)
                boxes = np.concatenate([boxes_, klass_], axis=1)

            boxes_ = boxes[:, 0:4]
            klass_ = boxes[:, 4:]
            if random.uniform(0, 1)>0.5:
                image, shift_x, shift_y = Fill_img(image, target_width=cfg.MODEL.win, target_height=cfg.MODEL.hin)
                boxes_[:, 0:4] = boxes_[:, 0:4] + np.array([shift_x, shift_y, shift_x, shift_y], dtype='float32')
            h, w, _ = image.shape
            boxes_[:, 0] /= w
            boxes_[:, 1] /= h
            boxes_[:, 2] /= w
            boxes_[:, 3] /= h

            interp_methods = [cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_AREA, cv2.INTER_NEAREST,
                              cv2.INTER_LANCZOS4]
            interp_method = random.choice(interp_methods)
            image = cv2.resize(image, (cfg.MODEL.win, cfg.MODEL.hin), interpolation=interp_method)

            boxes_[:, 0] *= cfg.MODEL.win
            boxes_[:, 1] *= cfg.MODEL.hin
            boxes_[:, 2] *= cfg.MODEL.win
            boxes_[:, 3] *= cfg.MODEL.hin
            image = image.astype(np.uint8)
            boxes = np.concatenate([boxes_, klass_], axis=1)

            if random.uniform(0, 1) > 0.5:
                image, boxes = Random_flip(image, boxes)
            if random.uniform(0, 1) > 0.5:
                image=self.color_augmentor(image)


            if random.uniform(0, 1) > 0.5:
                image = Pixel_jitter(image, 15)
            if random.uniform(0, 1) > 0.8:
                image = Gray_aug(image)


        else:
            boxes_ = boxes[:, 0:4]
            klass_ = boxes[:, 4:]
            image, shift_x, shift_y = Fill_img(image, target_width=cfg.MODEL.win, target_height=cfg.MODEL.hin)
            boxes_[:, 0:4] = boxes_[:, 0:4] + np.array([shift_x, shift_y, shift_x, shift_y], dtype='float32')
            h, w, _ = image.shape
            boxes_[:, 0] /= w
            boxes_[:, 1] /= h
            boxes_[:, 2] /= w
            boxes_[:, 3] /= h

            image = cv2.resize(image, (cfg.MODEL.win, cfg.MODEL.hin))

            boxes_[:, 0] *= cfg.MODEL.win
            boxes_[:, 1] *= cfg.MODEL.hin
            boxes_[:, 2] *= cfg.MODEL.win
            boxes_[:, 3] *= cfg.MODEL.hin
            image = image.astype(np.uint8)
            boxes = np.concatenate([boxes_, klass_], axis=1)

        ###cove the small faces
        boxes_clean = []
        for i in range(boxes.shape[0]):
            box = boxes[i]

            if (box[3] - box[1]) * (box[2] - box[0]) < cfg.DATA.cover_small_face:
                image[int(box[1]):int(box[3]), int(box[0]):int(box[2]), :] = cfg.DATA.PIXEL_MEAN
            else:
                boxes_clean.append([box[1], box[0], box[3], box[2]])
        boxes = np.array(boxes_clean)
        boxes = boxes / cfg.MODEL.hin



        if cfg.TRAIN.vis:
            for i in range(boxes.shape[0]):
                box=boxes[i]
                cv2.rectangle(image, (int(box[1]*cfg.MODEL.hin), int(box[0]*cfg.MODEL.hin)),
                                            (int(box[3]*cfg.MODEL.hin), int(box[2]*cfg.MODEL.hin)), (255, 0, 0), 7)

        reg_targets, matches = self.produce_target(boxes)

        image = image.astype(np.float32)


        return image, reg_targets,matches

    def produce_target(self,bboxes):
        reg_targets, matches = get_training_targets(bboxes, threshold=cfg.MODEL.MATCHING_THRESHOLD)
        return reg_targets, matches



class DataIter():
    def __init__(self,img_root_path='',ann_file=None,training_flag=True):


        self.training_flag=training_flag

        self.num_gpu = cfg.TRAIN.num_gpu
        self.batch_size = cfg.TRAIN.batch_size
        self.process_num = cfg.TRAIN.process_num
        self.prefetch_size = cfg.TRAIN.prefetch_size

        self.generator=FaceBoxesDataIter(img_root_path,ann_file,self.training_flag,)

        self.ds=self.build_iter()




    def build_iter(self):


        ds = DataFromGenerator(self.generator)

        ds = BatchData(ds, self.num_gpu *  self.batch_size)
        ds = MultiProcessPrefetchData(ds, self.prefetch_size, self.process_num)
        ds.reset_state()
        ds = ds.get_data()
        return ds


    def __next__(self):
        return next(self.ds)


    def _map_func(self,dp,is_training):

        raise NotImplementedError("you need implemented the map func for your data")

    def set_params(self):
        raise NotImplementedError("you need implemented  func for your data")