import sys
sys.path.append('.')
import numpy as np

try:
    from lib.core.model.facebox.utils.box_utils import to_minmax_coordinates
except:
    from utils.box_utils import to_minmax_coordinates
ANCHOR_SPECIFICATIONS = [
    [(32, 1.0, 4), (64, 1.0, 2), (128, 1.0, 1)],  # scale 0
    [(256, 1.0, 1)],  # scale 1
    [(512, 1.0, 1)],  # scale 2
]


class AnchorGenerator:
    def __init__(self):
        self.box_specs_list = ANCHOR_SPECIFICATIONS

    def __call__(self, feature_map_shape_list, image_size=(1024,1024)):
        """
        Arguments:
            image_features: a list of float tensors where the ith tensor
                has shape [batch, height_i, width_i, channels_i].
            image_size: a tuple of integers (int tensors with shape []) (width, height).
        Returns:
            a float tensor with shape [num_anchor, 4],
            boxes with normalized coordinates (and clipped to the unit square).
        """
        image_width, image_height = image_size

        # number of anchors per cell in a grid
        self.num_anchors_per_location = [
            sum(n*n for _, _, n in layer_box_specs)
            for layer_box_specs in self.box_specs_list
        ]

        anchor_grid_list, num_anchors_per_feature_map = [], []
        for grid_size, box_spec in zip(feature_map_shape_list, self.box_specs_list):

            h, w = grid_size
            stride = (1.0/float(h), 1.0/float(w))
            offset = (0.5/float(h), 0.5/float(w))

            local_anchors = []
            for scale, aspect_ratio, n in box_spec:

                local_anchors.append(tile_anchors(
                    image_size=(image_width, image_height),
                    grid_height=h, grid_width=w, scale=scale,
                    aspect_ratio=aspect_ratio, anchor_stride=stride,
                    anchor_offset=offset, n=n
                ))


            # reshaping in the right order is important
            local_anchors = np.concatenate(local_anchors, axis=2)
            local_anchors = np.reshape(local_anchors, [-1, 4])
            anchor_grid_list.append(local_anchors)

            num_anchors_per_feature_map.append(h * w * sum(n*n for _, _, n in box_spec))

        # constant tensors, anchors for each feature map
        self.anchor_grid_list = anchor_grid_list
        self.num_anchors_per_feature_map = num_anchors_per_feature_map
        anchors = np.concatenate(anchor_grid_list, axis=0)
        ymin, xmin, ymax, xmax = to_minmax_coordinates([anchors[:,0],anchors[:,1],anchors[:,2],anchors[:,3]])
        anchors = np.stack([ymin, xmin, ymax, xmax],axis=1)
        anchors = np.clip(anchors, 0.0, 1.0)
        anchors = np.array(anchors,dtype=np.float32)

        return anchors



def tile_anchors(
        image_size, grid_height, grid_width,
        scale, aspect_ratio, anchor_stride, anchor_offset, n):
    """
    Arguments:
        image_size: a tuple of integers (width, height).
        grid_height: an integer, size of the grid in the y direction.
        grid_width: an integer, size of the grid in the x direction.
        scale: a float number.
        aspect_ratio: a float number.
        anchor_stride: a tuple of float numbers, difference in centers between
            anchors for adjacent grid positions.
        anchor_offset: a tuple of float numbers,
            center of the anchor on upper left element of the grid ((0, 0)-th anchor).
        n: an integer, densification parameter.
    Returns:
        a float tensor with shape [grid_height, grid_width, n*n, 4].
    """
    ratio_sqrt = np.sqrt(aspect_ratio)

    unnormalized_height = scale / ratio_sqrt
    unnormalized_width = scale * ratio_sqrt

    # to [0, 1] range
    image_width, image_height = image_size
    height = unnormalized_height/float(image_height)
    width = unnormalized_width/float(image_width)
    # (sometimes it could be outside the range, but we clip it)


    boxes = generate_anchors_at_upper_left_corner(height, width, anchor_offset, n)
    # shape [n*n, 4]

    y_translation = np.arange(0,grid_height,1,dtype=np.float) * anchor_stride[0]
    x_translation = np.arange(0,grid_width,1,dtype=np.float) * anchor_stride[1]
    x_translation, y_translation = np.meshgrid(x_translation, y_translation)
    # they have shape [grid_height, grid_width]

    center_translations = np.stack([y_translation, x_translation], axis=2)
    translations = np.pad(center_translations, [[0, 0], [0, 0], [0, 2]],'constant', constant_values=(0))
    translations = np.expand_dims(translations, 2)
    translations = np.tile(translations, [1, 1, n*n, 1])
    # shape [grid_height, grid_width, n*n, 4]

    boxes = np.reshape(boxes, [1, 1, n*n, 4])
    boxes = boxes + translations  # shape [grid_height, grid_width, n*n, 4]
    return boxes


def generate_anchors_at_upper_left_corner(height, width, anchor_offset, n):
    """Generate densified anchor boxes at (0, 0) grid position."""

    # a usual center, if n = 1 it will be returned
    cy, cx = anchor_offset[0], anchor_offset[1]

    # a usual left upper corner
    ymin, xmin = cy - 0.5*height, cx - 0.5*width

    # now i shift the usual center a little (densification)
    sy, sx = height/n, width/n

    center_ids = (np.arange(0,n,1,dtype=np.float))
    # shape [n]

    # shifted centers
    new_centers_y = ymin + 0.5*sy + sy*center_ids
    new_centers_x = xmin + 0.5*sx + sx*center_ids
    # they have shape [n]

    # now i must get all pairs of y, x coordinates
    new_centers_y = np.expand_dims(new_centers_y, 0)  # shape [1, n]
    new_centers_x = np.expand_dims(new_centers_x, 1)  # shape [n, 1]

    new_centers_y = np.tile(new_centers_y, [n, 1])

    new_centers_x = np.tile(new_centers_x, [1, n])
    # they have shape [n, n]

    centers = np.stack([new_centers_y, new_centers_x], axis=2)
    # shape [n, n, 2]

    sizes = np.stack([height, width], axis=0)  # shape [2]
    sizes = np.expand_dims(sizes, 0)
    sizes = np.expand_dims(sizes, 0)  # shape [1, 1, 2]
    sizes = np.tile(sizes, [n, n, 1])

    boxes = np.stack([centers, sizes], axis=2)
    boxes = np.reshape(boxes, [-1, 4])

    return boxes


if __name__=='__main__':

    import cv2
    feature_map_shape_list=[[32,32],[16,16],[8,8]]

    anchorgenerator=AnchorGenerator()

    anchors=anchorgenerator(feature_map_shape_list,(1024,1024))
    print(anchors)

    img = np.ones(shape=[1024, 1024, 3])
    for i in range(20000, anchors.shape[0]):
        bbox = anchors[i, :] * 1024
        print(bbox)

        cv2.rectangle(img, (int(bbox[1]), int(bbox[0])),
                      (int(bbox[3]), int(bbox[2])), (255, 0, 0), 2)

        cv2.imshow('tmp', img)
        cv2.waitKey(0)