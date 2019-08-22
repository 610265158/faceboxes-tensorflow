# [faceboxes](https://arxiv.org/abs/1708.05234)

## introduction

A tensorflow implement faceboxes, and the input size is 512 (1024 in the paper).

It‘s a ssd-like object detect framework, but slightly different,

ps, the code maybe not that clear, please be patience, i am still working on it, and forgive me for my poor english :)

pretrained model can be download from
https://pan.baidu.com/s/1cUqnf9BwUVkCy0iT6EczKA ( password ty4d )


evaluation result on fddb

| fddb   |
| :------: | 
|  0.948 | 



## requirment

+ tensorflow1.12  

+ tensorpack (for data provider)

+ opencv

+ python 3.6

## useage

### train
1. download widerface data from http://shuoyang1213.me/WIDERFACE/
and release the WIDER_train, WIDER_val and wider_face_split into ./WIDER, 
2. download fddb and release the data into ./FDDB
3. then run
   ```python prepare_data.py```it will produce train.txt and val.txt

    (if u like train u own data, u should prepare the data like this:
    `...../9_Press_Conference_Press_Conference_9_659.jpg| 483(xmin),195(ymin),735(xmax),543(ymax),1(class) ......` 
    one line for one pic, **caution! class should start from 1, 0 means bg**)

4. then, run:

    `python train.py`

    and if u want to check the data when training, u could set vis in train_config.py as True
5. after training ,convert to pb file:
    `python tools/auto_freeze.py`

### evaluation

```
    python test/fddb.py [--model [TRAINED_MODEL]] [--data_dir [DATA_DIR]]
                          [--split_dir [SPLIT_DIR]] [--result [RESULT_DIR]]
    --model              Path of the saved model,default ./model/detector.pb
    --data_dir           Path of fddb all images
    --split_dir          Path of fddb folds
    --result             Path to save fddb results
 ```
    
example `python model_eval/fddb.py --model model/detector.pb 
                                    --data_dir 'fddb/img/' 
                                    --split_dir fddb/FDDB-folds/ 
                                    --result 'result/' `


### visualization
![A demo]()

if u get a trained model, run `python tools/auto_freeze.py`, it will read the checkpoint file in ./model, and produce detector.pb, then

`python vis.py`

u can check the code in vis.py to make it runable, it's simple.
### reference
# [FaceBoxes: A CPU Real-time Face Detector with High Accuracy](https://arxiv.org/abs/1708.05234)

