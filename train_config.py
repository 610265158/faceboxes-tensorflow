

import os
from easydict import EasyDict as edict

config = edict()
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

config.TRAIN = edict()

#### below are params for dataiter
config.TRAIN.num_gpu = 1
config.TRAIN.process_num = 2                    ####processors_num for data provider
config.TRAIN.prefetch_size = 50               ####Q size for data provider
############

config.TRAIN.batch_size = 32
config.TRAIN.log_interval = 10
config.TRAIN.epoch = 200
config.TRAIN.train_set_size=16098               ###########widerface size
config.TRAIN.val_set_size = 2845                #### fddb size
config.TRAIN.iter_num_per_epoch = config.TRAIN.train_set_size // config.TRAIN.num_gpu // config.TRAIN.batch_size
config.TRAIN.val_iter=config.TRAIN.val_set_size// config.TRAIN.num_gpu // config.TRAIN.batch_size
config.TRAIN.lr_value_every_step = [0.00001,0.0001,0.001,0.0001,0.00001,0.000001]    ###########lr policy
config.TRAIN.lr_decay_every_step = [500,1000,60000,80000,100000]
config.TRAIN.weight_decay_factor = 5.e-4                                ###########l2
config.TRAIN.vis=False

config.MODEL = edict()
config.MODEL.model_path = './model/'                                    # save directory
config.MODEL.continue_train=False                                       ### revover from a trained model
config.MODEL.net_structure=None                                         ######
config.MODEL.pretrained_model=None                                    ######
#####
config.MODEL.hin = 512                                                  # input size during training , 512  different with the paper
config.MODEL.win = 512
config.MODEL.feature_maps_size=[[32,32],[16,16],[8,8]]
config.MODEL.num_anchors=21824  ##it should be

config.MODEL.MATCHING_THRESHOLD = 0.35
config.MODEL.max_negatives_per_positive= 3.0

try:
    from lib.core.model.facebox.anchor_generator import AnchorGenerator
except:
    from anchor_generator import AnchorGenerator
anchorgenerator = AnchorGenerator()
config.MODEL.anchors=anchorgenerator(config.MODEL.feature_maps_size, (config.MODEL.hin*2, config.MODEL.win*2))

config.TEST = edict()
config.TEST.score_threshold=0.05
config.TEST.iou_threshold=0.3
config.TEST.max_boxes=100
config.TEST.parallel_iterations=8

config.DATA = edict()
config.DATA.root_path=''
config.DATA.train_txt_path='train.txt'
config.DATA.val_txt_path='val.txt'
config.DATA.NUM_CLASS=2

config.DATA.cover_small_face=400.                      ##small faces are covered
############NOW the model is trained with RGB mode
config.DATA.PIXEL_MEAN = [123., 116., 103.]   ###rgb
config.DATA.PIXEL_STD = [58., 57., 57.]











