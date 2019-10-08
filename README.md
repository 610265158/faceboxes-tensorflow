# [faceboxes](https://arxiv.org/abs/1708.05234)

## introduction

A tensorflow 2.0 implement faceboxes. And some changes has been made in RDCL module, to achieve a better performance and run faster:

   1. input size is 512 (1024 in the paper), then the first conv stride is 2, kernel size 7x7x12.
   2. replace the first maxpool by conv 3x3x24 stride 2
   3. replace the second 5x5 stride2 conv and maxpool by two 3x3 stride 2 conv
   4. anchor based sample is used in data augmentaion.
   
   
   codes like below
   ```
       with tf.name_scope('RDCL'):
        net = conv2d(net_in, 12, [7, 7], stride=2,activation_fn=tf.nn.relu, scope='init_conv1')
        net = conv2d(net, 24, [3, 3], stride=2, activation_fn=tf.nn.crelu, scope='init_conv2')
       
        net = conv2d(net, 32, [3, 3], stride=2,activation_fn=tf.nn.relu,scope='conv1x1_before1')
        net = conv2d(net, 64, [3, 3], stride=2, activation_fn=tf.nn.crelu, scope='conv1x1_before2')
        
        return net

   ```
**I want to name it faceboxes++ ,if u don't mind**


Pretrained model can be download from:

+ [baidu disk](https://pan.baidu.com/s/1DzbFYjcjcbXO4C494IB2TA) (code eb6b )

+ [google drive](https://drive.google.com/drive/folders/1mV7I9UR_DjF91Wd2P6TqMQhMIOpcBWRJ?usp=sharing)

Evaluation result on fddb

 ![fddb](https://github.com/610265158/faceboxes-tensorflow/blob/master/figures/Figure_1.png)

| fddb   |
| :------: | 
|  0.961 | 


 **Speed: it runs over 70FPS on cpu (i7-8700K), 30FPS (i5-7200U), 140fps on gpu (2080ti) with fixed input size 512, tf1.14, multi thread.**
 **And i think the input size, the time consume and the performance is very appropriate for application :)**
 
Hope the codes can help you, and i am struggling with the new tf api, contact me if u have any question,      2120140200@mail.nankai.edu.cn  .

## requirment

+ tensorflow2.0

+ opencv

+ python 3.6

## useage

### train
1. download widerface data from http://shuoyang1213.me/WIDERFACE/
and release the WIDER_train, WIDER_val and wider_face_split into ./WIDER, 
2. download fddb， and release FDDB-folds into ./FDDB ， 2002,2003 into ./FDDB/img
3. then run
   ```python prepare_data.py```it will produce train.txt and val.txt

    (if u like train u own data, u should prepare the data like this:
    `...../9_Press_Conference_Press_Conference_9_659.jpg| 483(xmin),195(ymin),735(xmax),543(ymax),1(class) ......` 
    one line for one pic, **caution! class should start from 1, 0 means bg**)

4. then, run:

    `python train.py`

    and if u want to check the data when training, u could set vis in train_config.py as True

    
### finetune
1.  (if u like train u own data, u should prepare the data like this:
    `...../9_Press_Conference_Press_Conference_9_659.jpg| 483(xmin),195(ymin),735(xmax),543(ymax),1(class) ......` 
    one line for one pic, **caution! class should start from 1, 0 means bg**)
2.  set config.MODEL.pretrained_model='your model pretrained', in train_config.py

3. `python train.py`
### evaluation

```
    python test/fddb.py [--model [TRAINED_MODEL]] [--data_dir [DATA_DIR]]
                          [--split_dir [SPLIT_DIR]] [--result [RESULT_DIR]]
    --model              Path of the saved model,default ./model/detector.pb
    --data_dir           Path of fddb all images
    --split_dir          Path of fddb folds
    --result             Path to save fddb results
 ```
    
example `python model_eval/fddb.py --model model/epoch_36_val_loss2.926745_keras.h5 
                                    --data_dir 'FDDB/img/' 
                                    --split_dir FDDB/FDDB-folds/ 
                                    --result 'result/' `


### visualization
![A demo](https://github.com/610265158/faceboxes-tensorflow/blob/master/figures/example2.png)

2. `python vis.py  --img_dir 'your images dir ,by default it detect pics with .jpg `

3. or use a camera:
`python vis.py --cam_id 0`

You can check the code in vis.py to make it runable, it's simple.
### reference
# [FaceBoxes: A CPU Real-time Face Detector with High Accuracy](https://arxiv.org/abs/1708.05234)

