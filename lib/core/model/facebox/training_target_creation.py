import numpy as np
import sys
sys.path.append('.')
try:
    from lib.core.model.facebox.utils.box_utils import encode, iou
except:
    from utils.box_utils import encode, iou

from train_config import config as cfg

def get_training_targets(groundtruth_boxes, threshold=0.5,anchors=cfg.MODEL.anchors):


    """
    Arguments:
        anchors: a float tensor with shape [num_anchors, 4].
        groundtruth_boxes: a float tensor with shape [N, 4].
        threshold: a float number.
    Returns:
        reg_targets: a float tensor with shape [num_anchors, 4].
        matches: an int tensor with shape [num_anchors], possible values
            that it can contain are [-1, 0, 1, 2, ..., (N - 1)].
    """

    N = np.shape(groundtruth_boxes)[0]
    num_anchors = np.shape(anchors)[0]
    no_match_tensor = np.ones(shape=[num_anchors])*-1

    if N>0:
        matches=_match(anchors, groundtruth_boxes, threshold)
    else:
        matches=no_match_tensor


    matches = np.array(matches,dtype=np.int)


    reg_targets = _create_targets(
       anchors, groundtruth_boxes, matches
    )

    return reg_targets, matches


def _match(anchors, groundtruth_boxes, threshold=0.5):
    """Matching algorithm:
    1) for each groundtruth box choose the anchor with largest iou,
    2) remove this set of anchors from the set of all anchors,
    3) for each remaining anchor choose the groundtruth box with largest iou,
       but only if this iou is larger than `threshold`.

    Note: after step 1, it could happen that for some two groundtruth boxes
    chosen anchors are the same. Let's hope this never happens.
    Also see the comments below.

    Arguments:
        anchors: a float tensor with shape [num_anchors, 4].
        groundtruth_boxes: a float tensor with shape [N, 4].
        threshold: a float number.
    Returns:
        an int tensor with shape [num_anchors].
    """
    num_anchors = np.shape(anchors)[0]

    # for each anchor box choose the groundtruth box with largest iou
    similarity_matrix = iou(groundtruth_boxes, anchors)  # shape [N, num_anchors]
    matches = np.argmax(similarity_matrix, axis=0).astype(np.int32)  # shape [num_anchors]

    matched_vals = np.max(similarity_matrix, axis=0)  # shape [num_anchors]

    below_threshold = np.greater(threshold, matched_vals).astype(np.int32)


    matches = np.add(np.multiply(matches, 1 - below_threshold), -1 * below_threshold)

    # after this, it could happen that some groundtruth
    # boxes are not matched with any anchor box

    # now we must ensure that each row (groundtruth box) is matched to
    # at least one column (which is not guaranteed
    # otherwise if `threshold` is high)

    # for each groundtruth box choose the anchor box with largest iou
    # (force match for each groundtruth box)
    forced_matches_ids = np.argmax(similarity_matrix, axis=1)  # shape [N]

    # if all indices in forced_matches_ids are different then all rows will be matched
    #forced_matches_indicators = tf.one_hot(forced_matches_ids, depth=num_anchors, dtype=tf.int32)  # shape [N, num_anchors]
    forced_matches_indicators = np_one_hot(forced_matches_ids, depth=num_anchors)  # shape [N, num_anchors]
    forced_match_row_ids = np.argmax(forced_matches_indicators, axis=0).astype(np.int)  # shape [num_anchors]

    forced_match_mask = np.greater(np.max(forced_matches_indicators, axis=0), 0)  # shape [num_anchors]

    matches = np.where(forced_match_mask, forced_match_row_ids, matches)
    # even after this it could happen that some rows aren't matched,
    # but i believe that this event has low probability

    #print(np.sum(matches[matches>=0]))

    return matches
def np_one_hot(data,depth):
    return (np.arange(depth) == data[:, None]).astype(np.int)


def _create_targets(anchors, groundtruth_boxes, matches):
    """Returns regression targets for each anchor.

    Arguments:
        anchors: a float tensor with shape [num_anchors, 4].
        groundtruth_boxes: a float tensor with shape [N, 4].
        matches: a int tensor with shape [num_anchors].
    Returns:
        reg_targets: a float tensor with shape [num_anchors, 4].
    """


    matched_anchor_indices = np.array(np.where(np.greater_equal(matches, 0)))  # shape [num_matches, 1]
    matched_anchor_indices = np.squeeze(matched_anchor_indices, axis=0)


    if len(matched_anchor_indices)==0:
        return np.zeros(shape=[cfg.MODEL.num_anchors,4])


    matched_gt_indices =matches[matched_anchor_indices] # shape [num_matches]

    matched_anchors = anchors[matched_anchor_indices]  # shape [num_matches, 4]

    matched_gt_boxes = groundtruth_boxes[matched_gt_indices]  # shape [num_matches, 4]

    matched_reg_targets = encode(matched_gt_boxes, matched_anchors)  # shape [num_matches, 4]

    reg_targets=np.zeros(shape=[cfg.MODEL.num_anchors,4])
    for i,index in enumerate(matched_anchor_indices):
        reg_targets[index,:]=matched_reg_targets[i,:]

    return reg_targets


if __name__=='__main__':





    from train_config import config as cfg

    a, b = get_training_targets(
        groundtruth_boxes=np.array([[0.32759732, 0.30244141, 0.3535777,  0.32003697],
                                     [0.32459959, 0.32942127 ,0.34058751, 0.34818987],
                                     [0.31460713, 0.35288202 ,0.33758978, 0.37868885],
                                     [0.30561392, 0.383381  , 0.33858902, 0.41622605],
                                     [0.31260864 ,0.45024414, 0.33758978, 0.47839704],
                                     [0.31860411, 0.4209182  ,0.34058751, 0.4396868 ],
                                     [0.1 ,0.1   ,0.2, 0.3],
                                     [0.34358525, 0.56050967 ,0.36556865, 0.58866257],
                                     [0.32260109, 0.53822196, 0.33958827, 0.55581752],
                                     [0.32160185, 0.55347145 ,0.34058751, 0.5769322 ],
                                     [0.31260864, 0.6050851 , 0.33459204, 0.62854585],
                                     [0.3096109 , 0.72825405, 0.33259355, 0.74819568],
                                     [0.32360034, 0.77400251 ,0.34358525 ,0.79277111],
                                     [0.32360034, 0.77400251 ,0.34358525 ,0.79277111]]
                                            , dtype=np.float32),
        threshold=0.35,
        anchors=cfg.MODEL.anchors)
    print(np.where(b >= 0)[0].shape)
    print(a)
    print(b)


