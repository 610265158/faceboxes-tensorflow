import tensorflow as tf
import numpy as np
import cv2
import time

from train_config import config as cfg


class FaceDetector:
    def __init__(self, model_path):
        """
        Arguments:
            model_path: a string, path to a pb file.
        """
        self._graph = tf.Graph()
        self._sess = tf.Session(graph=self._graph)

        with self._graph.as_default():
            self._graph, self._sess = self.init_model(*model_path)

            self.input_image = tf.get_default_graph().get_tensor_by_name('tower_0/images:0')
            self.training = tf.get_default_graph().get_tensor_by_name('training_flag:0')
            self.output_ops = [
                tf.get_default_graph().get_tensor_by_name('tower_0/boxes:0'),
                tf.get_default_graph().get_tensor_by_name('tower_0/scores:0'),
                tf.get_default_graph().get_tensor_by_name('tower_0/num_detections:0'),
            ]


    def __call__(self, image, score_threshold=0.5):
        """Detect faces.

        Arguments:
            image: a numpy uint8 array with shape [height, width, 3],
                that represents a RGB image.
            score_threshold: a float number.
        Returns:
            boxes: a float numpy array of shape [num_faces, 4].
            scores: a float numpy array of shape [num_faces].

        Note that box coordinates are in the order: ymin, xmin, ymax, xmax!
        """

        image_fornet,scale_x,scale_y=self.preprocess(image,target_width=cfg.MODEL.win,target_height=cfg.MODEL.hin)


        image_fornet = np.expand_dims(image_fornet, 0)

        start = time.time()
        boxes, scores, num_boxes = self._sess.run(
            self.output_ops, feed_dict={self.input_image: image_fornet,self.training:False}
        )
        #print('facebox detect cost', time.time() - start)
        num_boxes = num_boxes[0]
        boxes = boxes[0][:num_boxes]

        scores = scores[0][:num_boxes]

        to_keep = scores > score_threshold
        boxes = boxes[to_keep]
        scores = scores[to_keep]

        ###recorver to raw image
        scaler = np.array([cfg.MODEL.hin*scale_y,
                           cfg.MODEL.win*scale_x,
                           cfg.MODEL.hin*scale_y,
                           cfg.MODEL.win*scale_x], dtype='float32')
        boxes = boxes * scaler

        scores=np.expand_dims(scores, 0).reshape([-1,1])

        #####the tf.nms produce ymin,xmin,ymax,xmax,  swap it in to xmin,ymin,xmax,ymax
        for i in range(boxes.shape[0]):
            boxes[i] = np.array([boxes[i][1], boxes[i][0], boxes[i][3],boxes[i][2]])
        return np.concatenate([boxes, scores],axis=1)

    def preprocess(self,image,target_height,target_width,label=None):

        ###sometimes use in objs detects
        h,w,c=image.shape


        bimage=np.zeros(shape=[target_height,target_width,c],dtype=image.dtype)+np.array(cfg.DATA.PIXEL_MEAN,dtype=image.dtype)

        if h <=target_height and w <=target_width:
            bimage[:h,:w,:]=image
            scale_x=1.
            scale_y=1.
        else:
            long_side=max(h,w)

            scale_x=scale_y=target_height/long_side



            image=cv2.resize(image, None,fx=scale_x,fy=scale_y)


            h_,w_,_=image.shape
            bimage[:h_, :w_, :] = image


        return bimage,scale_x,scale_y







    def init_model(self,*args):

        if len(args) == 1:
            use_pb = True
            pb_path = args[0]
        else:
            use_pb = False
            meta_path = args[0]
            restore_model_path = args[1]

        def ini_ckpt():
            graph = tf.Graph()
            graph.as_default()
            configProto = tf.ConfigProto()
            configProto.gpu_options.allow_growth = True
            sess = tf.Session(config=configProto)
            # load_model(model_path, sess)
            saver = tf.train.import_meta_graph(meta_path)
            saver.restore(sess, restore_model_path)

            print("Model restred!")
            return (graph, sess)

        def init_pb(model_path):
            config = tf.ConfigProto()
            config.gpu_options.per_process_gpu_memory_fraction = 0.2
            compute_graph = tf.Graph()
            compute_graph.as_default()
            sess = tf.Session(config=config)
            with tf.gfile.GFile(model_path, 'rb') as fid:
                graph_def = tf.GraphDef()
                graph_def.ParseFromString(fid.read())
                tf.import_graph_def(graph_def, name='')

            # saver = tf.train.Saver(tf.global_variables())
            # saver.save(sess, save_path='./tmp.ckpt')
            return (compute_graph, sess)

        if use_pb:
            model = init_pb(pb_path)
        else:
            model = ini_ckpt()

        graph = model[0]
        sess = model[1]

        return graph, sess