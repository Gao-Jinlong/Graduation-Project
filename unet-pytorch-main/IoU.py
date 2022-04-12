from PIL import Image
import os
import numpy as np
from tqdm import tqdm

if __name__ == '__main__':
    eval_path = './eval/'
    label_dir = 'label'
    pre_dir = 'UNet_resnet_pre'

    label_list = os.listdir(os.path.join(eval_path,label_dir))
    pre_list = os.listdir(os.path.join(eval_path,pre_dir))

    # # 删除标签精细度后缀
    # count = 0
    # seg_dir = os.listdir(os.path.join(eval_path, label_dir))
    # for i in seg_dir:
    #     before_name = os.path.join(os.path.join(eval_path, label_dir), seg_dir[count])
    #     seg_dir[count] = 'ISIC_' + i.split('_')[1]
    #     after_name = os.path.join(os.path.join(eval_path, label_dir), seg_dir[count] + '.png')
    #     os.rename(before_name, after_name)
    #
    #     count += 1

    count = 1
    sum_iou = 0
    bar = tqdm(zip(label_list,pre_list))
    for label, pre in bar:
        if label.split('.')[0] != pre.split('.')[0]:
            exit('标签与预测结果匹配错误！')
        t = Image.open(os.path.join(eval_path,label_dir,label))
        p = Image.open(os.path.join(eval_path,pre_dir,pre))
        if p.mode != 'L':   # 转为灰度图
            p = p.convert('L')
        t = np.array(t)
        p = np.array(p)
        t = t/255   # 归一化
        p = p/255

        intersection = np.logical_and(t, p)
        union = np.logical_or(t, p)
        iou = np.sum(intersection) / np.sum(union)
        sum_iou = (sum_iou + iou)
        avg_iou = sum_iou/count
        count += 1
        bar.set_postfix({'avg_iou ': f'{avg_iou:.4f}'})