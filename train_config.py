

import os
import numpy as np
from easydict import EasyDict as edict

config = edict()
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

config.TRAIN = edict()
config.TRAIN.num_gpu = 1
config.TRAIN.batch_size = 32
config.TRAIN.log_interval = 10
config.TRAIN.epoch = 200
config.TRAIN.train_set_size=16000  ###########u need be sure
config.TRAIN.val_set_size = 2800
config.TRAIN.iter_num_per_epoch = config.TRAIN.train_set_size // config.TRAIN.num_gpu // config.TRAIN.batch_size
config.TRAIN.val_iter=config.TRAIN.val_set_size// config.TRAIN.num_gpu // config.TRAIN.batch_size
config.TRAIN.lr_value_every_step = [0.001,0.0001,0.001,0.0001,0.00001,0.000001]
config.TRAIN.lr_decay_every_step = [1000,2000,20000,40000,60000]
config.TRAIN.weight_decay_factor = 5.e-4
config.TRAIN.vis=False

config.MODEL = edict()
config.MODEL.model_path = './model/'  # save directory
config.MODEL.continue_train=False ### revover from a trained model
config.MODEL.net_structure=None ######'resnet_v1_50,resnet_v1_101,mobilenet
config.MODEL.pretrained_model=None#'resnet_v2_101.ckpt'
#####
config.MODEL.hin = 1024//2  # input size during training , 512
config.MODEL.win = 1024//2
config.MODEL.feature_maps_size=[[32,32],[16,16],[8,8]]
config.MODEL.num_anchors=21824  ##it should be
config.MODEL.score_threshold= 0.05
config.MODEL.iou_threshold= 0.3
config.MODEL.max_boxes= 200
config.MODEL.MATCHING_THRESHOLD = 0.35

config.MODEL.pretrained_model=None
try:
    from net.facebox.anchor_generator import AnchorGenerator
except:
    from anchor_generator import AnchorGenerator
anchorgenerator = AnchorGenerator()
config.MODEL.anchors=anchorgenerator(config.MODEL.feature_maps_size, (config.MODEL.hin*2, config.MODEL.win*2))


config.MODEL.max_negatives_per_positive= 3.0




config.DATA = edict()
config.DATA.root_path=''
config.DATA.train_txt_path='train.txt'
config.DATA.val_txt_path='val.txt'
config.DATA.NUM_CLASS=2

config.DATA.cover_small_face=400.
############NOW the model is trained with RGB mode
config.DATA.PIXEL_MEAN = [123.675, 116.28, 103.53]   ###rgb
config.DATA.PIXEL_STD = [58.395, 57.12, 57.375]











