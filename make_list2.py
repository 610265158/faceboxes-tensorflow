import os
import random
import json


def GetFileList(dir, fileList):
    newDir = dir
    if os.path.isfile(dir):
        fileList.append(dir)
    elif os.path.isdir(dir):
        for s in os.listdir(dir):
            #如果需要忽略某些文件夹，使用以下代码
            # if s == "pts":
            #     continue
            newDir=os.path.join(dir,s)
            GetFileList(newDir, fileList)
    return fileList
data_dir='/home/lz/coco_data/fddb_facetrain/annotations'


pic_list=[]
GetFileList(data_dir,pic_list)
pic_list=[x for x in pic_list if 'json' in x]
random.shuffle(pic_list)


ratio=1
train_list=pic_list[:int(ratio*len(pic_list))]
val_list=pic_list[int(ratio*len(pic_list)):]


train_file=open('./val.txt',mode='w')
#val_file=open('./val.txt',mode='w')


for _json in train_list:

    print(_json)
    content=json.load(open(_json,'r'))

    tmp_str=''
    image_path =os.path.join(_json.rsplit('/',1)[0].replace('annotations','images'),content['filename'].rsplit('/',1)[-1])

    tmp_str=image_path+'|'
    bboxes=content['object']
    if len(bboxes)>0:
        for bbox in bboxes:
            xmin = bbox['bndbox']['xmin']
            ymin = bbox['bndbox']['ymin']
            xmax = bbox['bndbox']['xmax']
            ymax = bbox['bndbox']['ymax']
            tmp_str=tmp_str+' %d,%d,%d,%d'%(xmin,ymin,xmax,ymax)
        tmp_str=tmp_str+'\n'
        train_file.write(tmp_str)
train_file.close()


# for _json in val_list:
#
#     content=json.load(open(_json,'r'))
#     tmp_str=''
#     image_path =os.path.join(_json.rsplit('/',1)[0].replace('annotations','images'),content['filename'].rsplit('/',1)[-1])
#     tmp_str=image_path+'|'
#     bboxes=content['object']
#     if len(bboxes) > 0:
#         for bbox in bboxes:
#             xmin = bbox['bndbox']['xmin']
#             ymin = bbox['bndbox']['ymin']
#             xmax = bbox['bndbox']['xmax']
#             ymax = bbox['bndbox']['ymax']
#         tmp_str=tmp_str+' %d,%d,%d,%d'%(xmin,ymin,xmax,ymax)
#         tmp_str=tmp_str+'\n'
#         val_file.write(tmp_str)
# val_file.close()


