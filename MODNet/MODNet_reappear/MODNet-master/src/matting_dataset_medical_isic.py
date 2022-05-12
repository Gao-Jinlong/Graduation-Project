from torch.utils.data import Dataset
import numpy as np
import random
import cv2
from glob import glob
import torch
from torchvision.transforms import functional as F
from torchvision import transforms
from PIL import Image
import os

# 数据集加载类
times = 30  # 增强数据集
class MattingDataset(Dataset):
    def __init__(self,
                 dataset_root_dir='E:/00_graduation project/DataSet/isic2017/Augmentation/',
                 data_type='Training',
                 transform=None):
        self.dataset_root_dir = dataset_root_dir
        self.data_type = data_type
        image_path = dataset_root_dir + self.data_type + f'_{times}x.txt'  # '_test.txt 读取调试demo'
        if self.data_type == 'Test':
            image_path = dataset_root_dir + self.data_type + '.txt'  # '_test.txt 读取调试demo'
        img_lines = []
        with open(image_path, "r") as f:
            for line in f.readlines():
                line = line.strip('\n')
                img_lines.append(line)  # 读入每行数据
        f.close()
        self.img_lines = img_lines
        self.transform = transform

    def __len__(self):
        return len(self.img_lines)

    # 每次读取数据集元素时访问此方法
    def __getitem__(self, index):
        image_file_name = self.dataset_root_dir + 'ISBI2016_ISIC_Part1_' + self.data_type + f'_Data_{times}x/' + self.img_lines[index] + '.jpg'
        matte_file_name = self.dataset_root_dir + 'ISBI2016_ISIC_Part1_' + self.data_type + f'_GroundTruth_{times}x/' + self.img_lines[index] + '_Segmentation' + '.png'
        if self.data_type == 'Test':
            image_file_name = self.dataset_root_dir + 'ISBI2016_ISIC_Part1_' + self.data_type + f'_Data/' + \
                              self.img_lines[index] + '.jpg'
            matte_file_name = self.dataset_root_dir + 'ISBI2016_ISIC_Part1_' + self.data_type + f'_GroundTruth/' + \
                              self.img_lines[index] + '_Segmentation' + '.png'
        image = Image.open(image_file_name)
        matte = Image.open(matte_file_name)

        # image.show()
        # matte.show()

        data = {'image': image, 'gt_matte': matte}

        # 根据传入列表 变换图像
        if self.transform:
            data = self.transform(data)
            #--------------------------------------------------------------------------
            # 经过各种变换后
            # data = {'image': image, 'trimap': trimap, 'gt_matte': gt_matte}
            #--------------------------------------------------------------------------
        # 显示图像
        # a = data[2].cpu().detach().numpy()
        # b = a[0]
        # # b = b.numpy() * 255
        # b = b * 255
        # image2 = Image.fromarray(b)
        # image2.show()
        return data

# 数据增强
class GetRandomData(object):
    def __init__(self, input_shape = [512, 512],jitter = .3, hue = .1,sat = 1.5, val = 1.5, random = False):
        self.input_shape  = input_shape
        self.jitter = jitter
        self.hue = hue
        self.sat = sat
        self.val = val
        self.random = random
    def __call__(self, sample):
        image = cvtColor(sample['image'])
        gt_matte = Image.fromarray(np.array(sample['gt_matte']))
        h, w = self.input_shape

        if not self.random:
            iw, ih = image.size
            scale = min(w/iw, h/ih)
            nw = int(scale * iw)
            nh = int(scale * ih)

            image = image.resize((nw, nh), Image.BICUBIC)
            new_image = Image.new('RGB',[w, h], (128, 128, 128))
            new_image.paste(image, ((w-nw)//2, (h-nh)//2))

            gt_matte = gt_matte.resize((nw, nh), Image.NEAREST)
            new_matte = Image.new('L', [w, h], (0))
            new_matte.paste(gt_matte, ((w-nw)//2, (h-nh)//2))
            return {'image': new_image, 'gt_matte': new_matte}

        rand_jit1 = self.rand(1-self.jitter, 1+self.jitter)
        rand_jit2 = self.rand(1-self.jitter, 1+self.jitter)
        new_ar = w/h * rand_jit1/rand_jit2  # 扭曲长宽比

        scale = self.rand(0.25, 2)
        if new_ar < 1:
            nh = int(scale * h)
            nw = int(nh * new_ar)
        else:
            nw = int(scale * w)
            nh = int(nw/new_ar)

        image = image.resize((nw, nh), Image.BICUBIC)
        gt_matte = gt_matte.resize((nw, nh), Image.NEAREST)

        flip = self.rand() < .5
        if flip:    # 水平翻转
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
            gt_matte = gt_matte.transpose(Image.FLIP_LEFT_RIGHT)

        flip = self.rand() < .5
        if flip:    # 垂直翻转
            image = image.transpose(Image.FLIP_TOP_BOTTOM)
            gt_matte = gt_matte.transpose(Image.FLIP_TOP_BOTTOM)

        # place image
        dx = int(self.rand(0, w-nw))
        dy = int(self.rand(0, h-nh))
        new_image = Image.new('RGB', (w, h), (128,128,128))
        new_gt_matte = Image.new('L', (w, h), (0))
        new_image.paste(image, (dx, dy))
        new_gt_matte.paste(gt_matte, (dx, dy))
        image = new_image
        gt_matte = new_gt_matte

        # distort color
        hue = self.rand(-self.hue, self.hue)
        sat = self.rand(1, self.sat) if self.rand() < .5 else 1/self.rand(1, self.sat)
        val = self.rand(1, self.val) if self.rand() < .5 else 1/self.rand(1, self.val)
        x = cv2.cvtColor(np.array(image, np.float32) / 255, cv2.COLOR_RGB2HSV)
        # x[..., 0] += hue * 360
        x[..., 0][x[..., 0] > 1] -= 1
        x[..., 0][x[..., 0] < 0] += 1
        x[..., 1] *= sat
        x[..., 2] *= val
        x[:, :, 0] [x[:, :, 0] > 360] = 360
        x[:, :, 1:][x[:, :, 1:] > 1] = 1
        x[x < 0] = 0
        image_data = cv2.cvtColor(x, cv2.COLOR_HSV2RGB) * 255
        image_data = Image.fromarray(np.uint8(image_data))
        # image_data.show()
        # gt_matte.show()
        return {'image': image_data, 'gt_matte': gt_matte}

    def rand(self, a=0, b=1):
        return np.random.rand() * (b - a) + a


# resize
class Rescale(object):
    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        image, gt_matte = sample['image'], sample['gt_matte']

        new_h, new_w = int(self.output_size), int(self.output_size)
        new_img = F.resize(image, (new_h, new_w))
        new_gt_matte = F.resize(gt_matte, (new_h, new_w))

        return {'image': new_img,'gt_matte': new_gt_matte}

# 创建三元图
class GenTrimap(object):
    def __call__(self, sample):     # sample = {'image': new_img,'gt_matte': new_gt_matte}
        gt_matte = sample['gt_matte']
        trimap = self.gen_trimap(gt_matte)
        sample['trimap'] = trimap
        return sample

    @staticmethod   # 静态方法 无需声明类便可调用
    def gen_trimap(matte):
        """
        根据归matte生成归一化的trimap
        trimap三元图，图像分割领域用来标注前景(Foreground)背景(Background)和待确认(Unknown)的标注图
        图像分割就是判断Unknown为Foreground还是Background的问题
        """
        matte = np.array(matte)
        k_size = random.choice(range(2, 5))     # 随机卷积核大小
        iterations = np.random.randint(5, 15)   # 随机迭代次数
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,   # 椭圆形结构元素
                                           (k_size, k_size))  # cv2.MORPH_RECT, cv2.MORPH_CROSS
        dilated = cv2.dilate(matte, kernel, iterations=iterations)  # 膨胀
        eroded = cv2.erode(matte, kernel, iterations=iterations)    # 腐蚀

        trimap = np.zeros(matte.shape)
        trimap.fill(0.5)
        trimap[eroded > 254.5] = 1
        trimap[dilated < 0.5] = 0.0
        trimap = Image.fromarray(trimap)
        return trimap   # 1为前景 0为背景 0.5为边缘

# 数组转tensor
class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image, trimap, gt_matte = sample['image'], sample['trimap'], sample['gt_matte']
        image = F.pil_to_tensor(image)
        trimap = F.pil_to_tensor(trimap)
        gt_matte = F.pil_to_tensor(gt_matte)


        return {'image': image,
                'trimap': trimap,
                'gt_matte': gt_matte}

# 转换数据类型
class ConvertImageDtype(object):
    def __call__(self, sample):
        image, trimap, gt_matte = sample['image'], sample['trimap'], sample['gt_matte']
        image = F.convert_image_dtype(image, torch.float)
        trimap = F.convert_image_dtype(trimap, torch.float)
        gt_matte = F.convert_image_dtype(gt_matte, torch.float)

        return {'image': image, 'trimap': trimap, 'gt_matte': gt_matte}

# 标准化
class Normalize(object):
    def __init__(self, mean, std, inplace=False):
        self.mean = mean    # 均值
        self.std = std      # 方差
        self.inplace = inplace  # 原地

    def __call__(self, sample):
        image, trimap, gt_matte = sample['image'], sample['trimap'], sample['gt_matte']
        image = image.type(torch.FloatTensor)
        image = F.normalize(image, self.mean, self.std, self.inplace)
        sample['image'] = image
        sample['gt_matte'] = sample['gt_matte']
        return sample

# 规整结构
class ToTrainArray(object):
    def __call__(self, sample):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        image = sample['image'].to(device)
        trimap = sample['trimap'].to(device)
        gt_matte = sample['gt_matte'].to(device)

        return [image, trimap, gt_matte]
        # return [sample['image'], sample['trimap'], sample['gt_matte']]

def cvtColor(image):
    if len(np.shape(image)) == 3 and np.shape(image)[-2] == 3:
        return image
    else:
        image = image.convert('RGB')
        return image

if __name__ == '__main__':

    # test MattingDataset.gen_trimap
    matte = Image.open('src/datasets/PPM-100/train/alpha/6146816_556eaff97f_o.jpg')
    trimap1 = GenTrimap().gen_trimap(matte)
    trimap1 = np.array(trimap1) * 255
    trimap1 = np.uint8(trimap1)
    trimap1 = Image.fromarray(trimap1)
    trimap1.save('test_trimap.png')

    # test MattingDataset
    transform = transforms.Compose([
        Rescale(512),
        GenTrimap(),
        ToTensor(),
        # Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    mattingDataset = MattingDataset(transform=transform)

    import matplotlib.pyplot as plt

    fig = plt.figure()

    for i in range(len(mattingDataset)):
        sample = mattingDataset[i]
        print(mattingDataset.image_file_name_list[i])
        # print(sample)
        print(i, sample['image'].shape, sample['trimap'].shape, sample['gt_matte'].shape)

        # break

        ax = plt.subplot(4, 3, 3 * i + 1)
        plt.tight_layout()
        ax.set_title('image #{}'.format(i))
        ax.axis('off')
        img = transforms.ToPILImage()(sample['image'])
        plt.imshow(img)

        ax = plt.subplot(4, 3, 3 * i + 2)
        plt.tight_layout()
        ax.set_title('gt_matte #{}'.format(i))
        ax.axis('off')
        img = transforms.ToPILImage()(sample['gt_matte'])
        plt.imshow(img)

        ax = plt.subplot(4, 3, 3 * i + 3)
        plt.tight_layout()
        ax.set_title('trimap #{}'.format(i))
        ax.axis('off')
        img = transforms.ToPILImage()(sample['trimap'])
        plt.imshow(img)

        if i == 3:
            plt.show()
            break
