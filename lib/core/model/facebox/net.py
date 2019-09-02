import tensorflow.contrib.slim as slim
import tensorflow as tf
import numpy as np


from lib.core.model.facebox.losses_and_ohem import localization_loss, ohem_loss
from lib.core.model.facebox.utils.box_utils import batch_decode
from lib.core.model.facebox.utils.nms import batch_non_max_suppression

from train_config import config as cfg

def facebox_arg_scope(weight_decay=0.00001,
                     batch_norm_decay=0.997,
                     batch_norm_epsilon=1e-5,
                     batch_norm_scale=True,
                     use_batch_norm=True,
                     batch_norm_updates_collections=tf.GraphKeys.UPDATE_OPS):
  """Defines the default ResNet arg scope.
  TODO(gpapan): The batch-normalization related default values above are
    appropriate for use in conjunction with the reference ResNet models
    released at https://github.com/KaimingHe/deep-residual-networks. When
    training ResNets from scratch, they might need to be tuned.
  Args:
    weight_decay: The weight decay to use for regularizing the model.
    batch_norm_decay: The moving average decay when estimating layer activation
      statistics in batch normalization.
    batch_norm_epsilon: Small constant to prevent division by zero when
      normalizing activations by their variance in batch normalization.
    batch_norm_scale: If True, uses an explicit `gamma` multiplier to scale the
      activations in the batch normalization layer.
    activation_fn: The activation function which is used in ResNet.
    use_batch_norm: Whether or not to use batch normalization.
    batch_norm_updates_collections: Collection for the update ops for
      batch norm.
  Returns:
    An `arg_scope` to use for the resnet models.
  """
  batch_norm_params = {
      'decay': batch_norm_decay,
      'epsilon': batch_norm_epsilon,
      'scale': batch_norm_scale,
      'updates_collections': batch_norm_updates_collections,
      'fused': True,  # Use fused batch norm if possible.
  }

  with slim.arg_scope(
      [slim.conv2d,slim.separable_conv2d],
      weights_regularizer=slim.l2_regularizer(weight_decay),
      weights_initializer=tf.glorot_normal_initializer(),
      normalizer_fn=slim.batch_norm if use_batch_norm else None,
      normalizer_params=batch_norm_params):
    with slim.arg_scope([slim.batch_norm], **batch_norm_params):
      # The following implies padding='SAME' for pool1, which makes feature
      # alignment easier for dense prediction tasks. This is also used in
      # https://github.com/facebook/fb.resnet.torch. However the accompanying
      # code of 'Deep Residual Learning for Image Recognition' uses
      # padding='VALID' for pool1. You can switch to that choice by setting
      # slim.arg_scope([slim.max_pool2d], padding='VALID').
      with slim.arg_scope([slim.max_pool2d], padding='SAME') as arg_sc:
        return arg_sc

def inception_block(x,scope):
    # path 1
    x1 = slim.conv2d(x, 32, (1, 1), scope=scope + '/conv_1x1_path1')
    # path 2
    y = slim.avg_pool2d(x, (3, 3), stride=1, padding='SAME', scope=scope + '/pool_3x3_path2')
    x2 = slim.conv2d(y, 32, (1, 1), scope=scope + '/conv_1x1_path2')
    # path 3
    y = slim.conv2d(x, 24, (1, 1), scope=scope + '/conv_1x1_path3')
    x3 = slim.conv2d(y, 32, (3, 3), scope=scope + '/conv_3x3_path3')
    # path 4
    y = slim.conv2d(x, 24, (1, 1), scope=scope + '/conv_1x1_path4')
    y = slim.conv2d(y, 32, (3, 3), scope=scope + '/conv_3x3_path4')
    x4 = slim.conv2d(y, 32, (3, 3), scope=scope + '/conv_3x3_second_path4')
    return tf.concat([x1, x2, x3, x4], axis=3, name=scope + '/concat')

    return net_out



### RDCL in the papre
# def RDCL(net_in):
#     with tf.name_scope('RDCL'):
#         net = slim.conv2d(net_in, 24, [7, 7], stride=2,activation_fn=tf.nn.crelu, scope='init_conv')
#         net = tf.nn.max_pool(net, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding="SAME", name='init_pool')
#         net = slim.conv2d(net, 64, [5, 5], stride=2,activation_fn=tf.nn.crelu,scope='conv1x1_before')
#         net = tf.nn.max_pool(net, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding="SAME", name='init_pool2')
#         return net


### the new one
def RDCL(net_in):
    with tf.name_scope('RDCL'):
        net = slim.conv2d(net_in, 12, [7, 7], stride=2,activation_fn=tf.nn.relu, scope='init_conv1')
        net = slim.conv2d(net, 24, [3, 3], stride=2, activation_fn=tf.nn.crelu, scope='init_conv2')
        #net = tf.nn.max_pool(net, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding="SAME", name='init_pool')

        net = slim.conv2d(net, 32, [3, 3], stride=2,activation_fn=tf.nn.relu,scope='conv1x1_before1')
        net = slim.conv2d(net, 64, [3, 3], stride=2, activation_fn=tf.nn.crelu, scope='conv1x1_before2')
        #net = tf.nn.max_pool(net, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding="SAME", name='init_pool2')
        return net

def MSCL(net):

    with tf.name_scope('MSCL'):
        feature_maps = []
        net = inception_block(net, 'inception1')
        net = inception_block(net, 'inception2')
        net = inception_block(net, 'inception3')

        feature_maps.append(net)
        net = slim.conv2d(net, 128, (1, 1), scope='conv3_1')
        net = slim.conv2d(net, 256, (3, 3), stride=2, scope='conv3_2')

        feature_maps.append(net)
        net = slim.conv2d(net, 128, (1, 1), scope='conv4_1')
        net = slim.conv2d(net, 256, (3, 3), stride=2, scope='conv4_2')
        feature_maps.append(net)

        print("feature_maps shapes:", feature_maps)

        return feature_maps

def output(feature_maps):
    feature_1,feature_2,feature_3=feature_maps

    with tf.name_scope('out'):
        ###level 1

        reg_1 = slim.conv2d(feature_1, 4*21, [3, 3], stride=1, activation_fn=None,
                          normalizer_fn=None, scope='level1_reg_out')

        reg_1=tf.reshape(reg_1, ([-1, 32, 32, 21, 4]))
        reg_1 = tf.reshape(reg_1, ([-1,32* 32*21, 4]))
        ###level 2

        reg_2 = slim.conv2d(feature_2, 4*1, [3, 3], stride=1, activation_fn=None,
                            normalizer_fn=None, scope='level2_reg_out')
        reg_2 = tf.reshape(reg_2, ([-1, 16, 16, 1, 4]))
        reg_2 = tf.reshape(reg_2, ([-1,16*16, 4]))
        ###level 3

        reg_3 = slim.conv2d(feature_3, 4*1, [3, 3], stride=1, activation_fn=None,
                            normalizer_fn=None, scope='level3_reg_out')
        reg_3=tf.reshape(reg_3, ([-1, 8, 8, 1, 4]))
        reg_3 = tf.reshape(reg_3, ([-1,8*8, 4]))


        reg=tf.concat([reg_1,reg_2,reg_3],axis=1)


        ##cla
        cla_1 = slim.conv2d(feature_1, 2 * 21, [3, 3], stride=1, activation_fn=None,
                            normalizer_fn=None, scope='level1_cla_out')

        cla_1 = tf.reshape(cla_1, ([-1, 32, 32, 21, 2]))
        cla_1 = tf.reshape(cla_1, ([-1, 32 * 32 * 21, 2]))

        cla_2 = slim.conv2d(feature_2, 2 * 1, [3, 3], stride=1, activation_fn=None,
                            normalizer_fn=None, scope='level2_cla_out')
        cla_2 = tf.reshape(cla_2, ([-1, 16, 16, 1, 2]))
        cla_2 = tf.reshape(cla_2, ([-1, 16 * 16, 2]))

        cla_3 = slim.conv2d(feature_3, 2 * 1, [3, 3], stride=1, activation_fn=None,
                            normalizer_fn=None, scope='level3_cla_out')
        cla_3=tf.reshape(cla_3, ([-1, 8, 8, 1, 2]))
        cla_3 = tf.reshape(cla_3, ([-1,8*8, 2]))

        cla = tf.concat([cla_1, cla_2, cla_3], axis=1)

        return reg,cla




def preprocess(image):
    with tf.name_scope('image_preprocess'):
        if image.dtype.base_dtype != tf.float32:
            image = tf.cast(image, tf.float32)

        mean = cfg.DATA.PIXEL_MEAN
        #std = np.asarray(cfg.DATA.PIXEL_STD)

        image_mean = tf.constant(mean, dtype=tf.float32)
        #image_invstd = tf.constant(1.0 / std, dtype=tf.float32)
        image = (image - image_mean)#*image_invstd

    return image

def facebox_backbone(inputs,L2_reg,training=True):
    inputs=preprocess(inputs)
    arg_scope = facebox_arg_scope(weight_decay=L2_reg)
    with slim.arg_scope(arg_scope):
        with slim.arg_scope([slim.batch_norm], is_training=training):
            with tf.name_scope('Facebox'):
                net=RDCL(inputs)
                fms=MSCL(net)

                reg,cla =output(fms)

    return reg,cla

def facebox(inputs, reg_targets, matches, L2_reg, training):


    loc_predict, cla_predict = facebox_backbone(inputs, L2_reg, training)

    with tf.name_scope('losses'):
        # whether anchor is matched
        is_matched = tf.greater_equal(matches, 0)
        weights = tf.to_float(is_matched)
        # shape [batch_size, num_anchors]

        # we have binary classification for each anchor
        cls_targets = tf.to_int32(is_matched)

        with tf.name_scope('classification_loss'):
            cls_losses = ohem_loss(
                cla_predict,
                cls_targets,
                is_matched
            )
        with tf.name_scope('localization_loss'):
            location_losses = localization_loss(
                loc_predict,
                reg_targets, weights
            )
        # they have shape [batch_size, num_anchors]

        with tf.name_scope('normalization'):
            matches_per_image = tf.reduce_sum(weights, axis=1)  # shape [batch_size]
            num_matches = tf.reduce_sum(matches_per_image)  # shape []
            normalizer = tf.maximum(num_matches, 1.0)




    reg_loss = tf.reduce_sum(location_losses) / normalizer
    cla_loss = tf.reduce_sum(cls_losses) / normalizer

    ######add nms in the graph
    get_predictions(loc_predict, cla_predict, anchors=cfg.MODEL.anchors)
    return {'localization_loss': reg_loss, 'classification_loss':cla_loss}




def get_predictions(box_encodings,cla,anchors, score_threshold=cfg.TEST.score_threshold, iou_threshold=cfg.TEST.iou_threshold, max_boxes=cfg.TEST.max_boxes):
    """Postprocess outputs of the network.

    Returns:
        boxes: a float tensor with shape [batch_size, N, 4].
        scores: a float tensor with shape [batch_size, N].
        num_boxes: an int tensor with shape [batch_size], it
            represents the number of detections on an image.

        where N = max_boxes.
    """
    with tf.name_scope('postprocessing'):
        boxes = batch_decode(box_encodings, anchors)
        # if the images were padded we need to rescale predicted boxes:

        boxes = tf.clip_by_value(boxes, 0.0, 1.0)
        # it has shape [batch_size, num_anchors, 4]

        scores = tf.nn.softmax(cla, axis=2)[:, :, 1]
        # it has shape [batch_size, num_anchors],  background are ignored

    with tf.device('/cpu:0'), tf.name_scope('nms'):
        boxes, scores, num_detections = batch_non_max_suppression(
            boxes, scores, score_threshold, iou_threshold, max_boxes
        )

    boxes=tf.identity(boxes,name='boxes')
    scores = tf.identity(scores, name='scores')
    num_detections = tf.identity(num_detections, name='num_detections')

    return {'boxes': boxes, 'scores': scores, 'num_boxes': num_detections}




