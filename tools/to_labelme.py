#-*-coding:utf-8-*-
import cv2
import os
import json
import random




def to_json(file_path,landmark):
    labelme_json_str = {
        "flags": {},
        "shapes": [

        ],
        "lineColor": [
            0,
            255,
            0,
            128
        ],
        "fillColor": [
            255,
            0,
            0,
            128
        ],
        "imagePath": "PUB_DATA/IBUG/ibug/image_003_1.jpg",
        "imageData": None
    }


    ###if there is no landmark do nothing
    if len(landmark)==0:
        return


    filename=file_path.rsplit('/',1)[-1]
    #img_raw=cv2.imread(file_path)
    labelme_json_str["imagePath"]=filename


    how_many_landmark=landmark.shape[0]
    index=random.randint(0,how_many_landmark-1)

    one_landmark=landmark[index]

    tmp_shape = {'label': str(0), 'line_color': None, 'fill_color': None, \
                 "points": []}

    for i in range(one_landmark.shape[0]):
        x_y=one_landmark[i]

        tmp_shape['points'].append([int(x_y[0]),int(x_y[1])])
    labelme_json_str['shapes'].append(tmp_shape)


    with open(file_path.replace('jpg','json'),mode='w') as json_f:
        json.dump(labelme_json_str,json_f,indent=2)








