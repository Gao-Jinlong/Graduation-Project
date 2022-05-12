import cv2
import numpy as np
import os
from tqdm import tqdm
from PIL import Image
def rand(a=0, b=1):
    return np.random.rand() * (b - a) + a

def cvtColor(image):
    if len(np.shape(image)) == 3 and np.shape(image)[-2] == 3:
        return image
    else:
        image = image.convert('RGB')
        return image

def get_random_data(image, label, input_shape, jitter=.3, hue=.1, sat=1.5, val=1.5, random=True):
    image = cvtColor(image)
    label = Image.fromarray(np.array(label))
    h, w = input_shape

    if not random:
        iw, ih = image.size
        scale = min(w / iw, h / ih)
        nw = int(iw * scale)
        nh = int(ih * scale)

        image = image.resize((nw, nh), Image.BICUBIC)
        new_image = Image.new('RGB', [w, h], (128, 128, 128))
        new_image.paste(image, ((w - nw) // 2, (h - nh) // 2))

        label = label.resize((nw, nh), Image.NEAREST)
        new_label = Image.new('L', [w, h], (0))
        new_label.paste(label, ((w - nw) // 2, (h - nh) // 2))
        return new_image, new_label

    # resize image
    rand_jit1 = rand(1 - jitter, 1 + jitter)  # 0.7 - 1.3
    rand_jit2 = rand(1 - jitter, 1 + jitter)
    new_ar = w / h * rand_jit1 / rand_jit2  # 原始长宽比进行一定拉伸

    scale = rand(0.25, 2)  # 缩放系数
    if new_ar < 1:  # 随机拉伸垂直或水平方向
        nh = int(scale * h)
        nw = int(nh * new_ar)
    else:
        nw = int(scale * w)
        nh = int(nw / new_ar)

    image = image.resize((nw, nh), Image.BICUBIC)
    label = label.resize((nw, nh), Image.NEAREST)

    flip = rand() < .5
    if flip:  # 水平反转
        image = image.transpose(Image.FLIP_LEFT_RIGHT)
        label = label.transpose(Image.FLIP_LEFT_RIGHT)


    # # distort image # 色相变换
    hue = rand(-hue, hue)
    sat = rand(1, sat) if rand() < .5 else 1 / rand(1, sat)
    val = rand(1, val) if rand() < .5 else 1 / rand(1, val)
    x = cv2.cvtColor(np.array(image, np.float32) / 255, cv2.COLOR_RGB2HSV)
    # x[..., 0] += hue * 360
    x[..., 0][x[..., 0] > 1] -= 1
    x[..., 0][x[..., 0] < 0] += 1
    x[..., 1] *= sat
    x[..., 2] *= val
    x[x[:, :, 0] > 360, 0] = 360
    x[:, :, 1:][x[:, :, 1:] > 1] = 1
    x[x < 0] = 0
    image_data = cv2.cvtColor(x, cv2.COLOR_HSV2RGB) * 255
    image_data = np.array(image_data, 'uint8')
    image_data = Image.fromarray(image_data)
    # --------------------------------------------
    # image_data = cv2.cvtColor(image_data, cv2.COLOR_RGB2BGR)
    # image_data = np.array(image_data/255,np.float32)
    # cv2.imshow('img',image_data)  # cv2默认BGR顺序输出 此时图像是RGB顺序
    # cv2.waitKey(0)
    # exit()

    # place image   将图像贴到[512,512]的灰底上
    dx = int(rand(0, w - nw))
    dy = int(rand(0, h - nh))
    new_image = Image.new('RGB', (w, h), (128, 128, 128))
    new_label = Image.new('L', (w, h), (0))
    new_image.paste(image_data, (dx, dy))
    new_label.paste(label, (dx, dy))
    image = new_image
    label = new_label

    return image, label

# 增加倍数
times = 30

if __name__ =='__main__':
    dataset_root_dir = 'E:/00_graduation project/DataSet/isic2017'
    # 原始文件位置
    image_path = os.path.join(dataset_root_dir, 'ISBI2016_ISIC_Part1_Training_Data')
    gt_path = os.path.join(dataset_root_dir, 'ISBI2016_ISIC_Part1_Training_GroundTruth')
    # 新生成文件位置
    new_image_path = os.path.join(dataset_root_dir, f'Augmentation/ISBI2016_ISIC_Part1_Training_Data_{times}x')
    new_gt_path = os.path.join(dataset_root_dir, f'Augmentation/ISBI2016_ISIC_Part1_Training_GroundTruth_{times}x')

    txt = open(os.path.join(dataset_root_dir, f'Augmentation/Training_{times}x.txt'), 'w')
    for i in range(times):
        for index in tqdm(os.listdir(image_path)):
            id = index.split('.')[0]

            image = Image.open(os.path.join(image_path, index))
            gt = Image.open(os.path.join(gt_path, id + '_Segmentation.png'))

            new_iamge, new_gt = get_random_data(image, gt, [512,512], random = True)

            # 路径不存在则创建路径
            if not os.path.exists(new_image_path):
                os.makedirs(new_image_path)
            if not os.path.exists(new_gt_path):
                os.makedirs(new_gt_path)
            new_iamge.save(os.path.join(new_image_path, id + f'_{i}.jpg'))
            new_gt.save(os.path.join(new_gt_path, id + f'_{i}_Segmentation.png'))
            txt.write(id + f'_{i}\n')