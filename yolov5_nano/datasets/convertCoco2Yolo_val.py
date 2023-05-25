import os
import json
from tqdm import tqdm
import argparse

#COCO 格式的数据集转化为 YOLO 格式的数据集
def convert(size, box):
    dw = 1. / (size[0])
    dh = 1. / (size[1])
    x = box[0] + box[2] / 2.0
    y = box[1] + box[3] / 2.0
    w = box[2]
    h = box[3]
#round函数确定(xmin, ymin, xmax, ymax)的小数位数
    x = round(x * dw, 6)
    w = round(w * dw, 6)
    y = round(y * dh, 6)
    h = round(h * dh, 6)
    return (x, y, w, h)

mode = "val"
root_path = '/group/dphi_algo_scratch_13/fangyuan/internal-cooperation-models/pytorch/yolov5-m/datasets/coco'
image_path = '/group/dphi_algo/coco/images/images_2017/{}2017/'.format(mode)
json_file = '/group/dphi_algo/coco/annotations/annotations_2017/instances_{}2017.json'.format(mode)
ana_txt_save_path = root_path + '/labels/{}2017'.format(mode)

data = json.load(open(json_file, 'r'))
if not os.path.exists(ana_txt_save_path):
    os.makedirs(ana_txt_save_path)
id_map = {} # coco数据集的id不连续！重新映射一下再输出！
with open(os.path.join(root_path, 'classes.txt'), 'w') as f:
    # 写入classes.txt
    for i, category in enumerate(data['categories']):
        f.write(f"{category['name']}\n")
        id_map[category['id']] = i

#这里需要根据自己的需要，更改写入图像相对路径的文件位置。
list_file = open(os.path.join(root_path, '{}2017.txt'.format(mode)), 'w')
for img in tqdm(data['images']):
    filename = img["file_name"]
    img_width = img["width"]
    img_height = img["height"]
    img_id = img["id"]

    head, tail = os.path.splitext(filename)
    ana_txt_name = head + ".txt"  # 对应的txt名字，与jpg一致
    f_txt = open(os.path.join(ana_txt_save_path, ana_txt_name), 'w')
    for ann in data['annotations']:
        if ann['image_id'] == img_id:
            box = convert((img_width, img_height), ann["bbox"])
            f_txt.write("%s %s %s %s %s\n" % (id_map[ann["category_id"]], box[0], box[1], box[2], box[3]))
    f_txt.close()
    #将图片的相对路径写入train2017或val2017的路径
    list_file.write(image_path + '%s.jpg\n' %(head))
list_file.close()
