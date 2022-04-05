import os
from tqdm import tqdm
import random

# 数据集路径
dataset_root_dir = 'E:/00_graduation project/DataSet/isic2017/ISIC-Archive-Downloader-master/Data'
ratio = 0.1     # 验证集比率

if __name__ =='__main__':

    train_txt = open(os.path.join(dataset_root_dir, 'train.txt'), 'w')
    val_txt = open(os.path.join(dataset_root_dir, 'val.txt'), 'w')
    img_dir = os.listdir(os.path.join(dataset_root_dir, 'Images'))

    random.shuffle(img_dir)
    cut_poin = len(img_dir) * ratio
    count = 0
    for i in tqdm(img_dir):
        i = i.split('.')[0] # 删掉后缀
        if count >= cut_poin:
            train_txt.write(i + '\n')
        else:
            val_txt.write(i + '\n')
        count += 1

    ''' # 从官网下载的数据集需要处理分割集文件名，删去精细度后缀
    count = 0
    seg_dir = os.listdir(os.path.join(dataset_root_dir, 'Segmentation'))
    for i in tqdm(seg_dir):
        before_name = os.path.join(os.path.join(dataset_root_dir, 'Segmentation'), seg_dir[count])
        seg_dir[count] = 'ISIC_' + i.split('_')[1]
        after_name = os.path.join(os.path.join(dataset_root_dir, 'Segmentation'), seg_dir[count] + '.png')
        os.rename(before_name, after_name)

        count += 1
    '''
