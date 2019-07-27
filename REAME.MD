# dsfd

## introduction

A tensorflow implement faceboxes, and there is something different with the origin paper.

It‘s a ssd-like object detect framework, but slightly different,
combines lots of tricks for face detection, such as dual-shot, dense anchor match, FPN,FEM and so on.

now it is mainly optimised about face detection,
and borrows some codes from other repos

ps, the code maybe not that clear, please be patience, and i am still working on it, and forgive me for my poor english :)


the evaluation results are based on vgg with batchsize(32),
pretrained model can be download from
https://pan.baidu.com/s/1cUqnf9BwUVkCy0iT6EczKA ( password ty4d )


evaluation result on fddb

| fddb   |
| :------: | 
|  0.95 | 



## requirment

+ tensorflow1.12

+ tensorpack (for data provider)

+ opencv

+ python 3.6

## useage

### train
1. download widerface data from http://shuoyang1213.me/WIDERFACE/
and release the WIDER_train, WIDER_val and wider_face_split into ./WIDER, 
2. download fddb and release alll into ./FDDB
3. then run
```python prepare_data.py```it will produce train.txt and val.txt

(if u like train u own data, u should prepare the data like this:
`...../9_Press_Conference_Press_Conference_9_659.jpg| 483(xmin),195(ymin),735(xmax),543(ymax),1(class) ......` 
one line for one pic, **caution! class should start from 1, 0 means bg**)

4. then, run:

`python train.py`


and if u want to check the data when training, u could set vis in train_config.py as True


#### ** CAUTION， WHEN USE TENSORPACK FOR DATA PROVIDER, some change is needed. **
#### in lib/python3.6/site-packages/tensorpack/dataflow/raw.py ,line 71-96. to make the iterator unstoppable, change it as below. so that we can keep trainning when the iter was over. contact me if u have problem about the codes : )
```
 71 class DataFromList(RNGDataFlow):
 72     """ Wrap a list of datapoints to a DataFlow"""
 73 
 74     def __init__(self, lst, shuffle=True):
 75         """
 76         Args:
 77             lst (list): input list. Each element is a datapoint.
 78             shuffle (bool): shuffle data.
 79         """
 80         super(DataFromList, self).__init__()
 81         self.lst = lst
 82         self.shuffle = shuffle
 83     
 84     #def __len__(self):
 85     #    return len(self.lst)
 86 
 87     def __iter__(self):
 88         if not self.shuffle:
 89             for k in self.lst:
 90                 yield k
 91         else:
 92             while True:
 93                 idxs = np.arange(len(self.lst))
 94                 self.rng.shuffle(idxs)
 95                 for k in idxs:
 96                     yield self.lst[k]
```



### evaluation

```
    python model_eval/fddb.py [--model [TRAINED_MODEL]] [--data_dir [DATA_DIR]]
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
![A demo](https://github.com/610265158/dsfd_tensorflow/blob/master/res_screenshot_11.05.2019.png)

if u get a trained model, run `python tools/auto_freeze.py`, it will read the checkpoint file in ./model, and produce detector.pb, then

`python vis.py`

u can check the code in vis.py to make it runable, it's simple.



