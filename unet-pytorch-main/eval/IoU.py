from PIL import Image
import os
import numpy as np
from tqdm import tqdm

def get_mse(pre, label):
    if len(pre) == len(label):
        return np.array([(pre - label) ** 2]).sum() / (label.shape[0] * label.shape[1])
    else:
        return None


if __name__ == '__main__':
    eval_path = './'
    label_dir = 'GroundTruth'
    pre_dir = 'MODNet_isic_v06'

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
    iou_sum = 0
    precision_sum = 0
    sensitivity_sum = 0
    accuracy_sum = 0
    mse_sum = 0
    bar = tqdm(zip(label_list,pre_list))
    for label, pre in bar:
        # if label.split('.')[0] != pre.split('.')[0]:
        if label.split('_')[1] != pre.split('_')[1].split('.')[0]:
            print(label.split('_')[1],pre.split('_')[1].split('.')[0])
            exit('标签与预测结果匹配错误！')
        t = Image.open(os.path.join(eval_path,label_dir,label))
        p = Image.open(os.path.join(eval_path,pre_dir,pre))
        # t.show()
        # p.show()
        if p.mode != 'L':   # 转为灰度图
            p = p.convert('L')
        t = np.array(t)
        p = np.array(p)
        t = t/255   # 归一化
        p = p/255
        mse = get_mse(p, t)
        mse_sum += mse
        # mes_vison = Image.fromarray(mse[0]*255)
        # mes_vison.show()
        # exit()

        TP = (t * p).sum()
        FP = np.float(np.sum((p >= 0.5) & (t == 0)))
        FN = np.float(np.sum((p == 0) & (t == 1)))
        TN = np.float(np.sum((p == 0) & (t == 0)))
        # intersection = np.logical_and(t, p)
        union = t.sum() + p.sum() - TP
        # union = np.logical_or(t, p)
        iou = TP/union

        # 精准率
        precision_sum += float(TP) / (float(TP + FP) + 1e-6)    #  +1e-6 防止分母为0
        # 召回率 敏感度
        sensitivity_sum += float(TP) / (float(TP + FN) + 1e-6)
        # 准确率
        accuracy_sum += float(TP + TN) / (float(TP + TN + FP + FN) + 1e-6)
        # iou = np.sum(intersection) / np.sum(union)
        iou_sum = (iou_sum + iou)
        avg_iou = iou_sum / count
        avg_precision = precision_sum / count
        avg_sensitivity = sensitivity_sum / count
        avg_accuracy = accuracy_sum / count
        avg_mse = mse_sum / count
        count += 1
        bar.set_postfix({'avg_iou ': f'{avg_iou:.4f}','avg_precision':f'{avg_precision:.4f}','avg_sensitivity':f'{avg_sensitivity:.4f}','avg_accuracy':f'{avg_accuracy:.4f}','avg_mse':f'{avg_mse:.4f}'})