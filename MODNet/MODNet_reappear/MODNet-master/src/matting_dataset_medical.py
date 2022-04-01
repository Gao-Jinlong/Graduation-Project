from torch.utils.data import Dataset
import numpy as np
import random
import cv2
from glob import glob
import torch
from torchvision.transforms import functional as F
from torchvision import transforms
from PIL import Image

# 数据集加载类
class MattingDataset(Dataset):
    def __init__(self,
                 dataset_root_dir='datasets/PPM-100',
                 transform=None):
        image_path = dataset_root_dir + '/train/fg/*'
        matte_path = dataset_root_dir + '/train/alpha/*'
        image_file_name_list = glob(image_path)     # 全局搜索路径
        matte_file_name_list = glob(matte_path)
        self.image_file_name_list = sorted(image_file_name_list)    # 排序列表
        self.matte_file_name_list = sorted(matte_file_name_list)
        for img, mat in zip(self.image_file_name_list, self.matte_file_name_list):  # zip配对元组
            img_name = img.split("\\")[-1]  # 截取文件名
            mat_name = mat.split('\\')[-1]
            assert img_name == mat_name

        self.transform = transform

    def __len__(self):
        return len(self.image_file_name_list)

    # 每次读取数据集元素时访问此方法
    def __getitem__(self, index):
        image_file_name = self.image_file_name_list[index]
        matte_file_name = self.matte_file_name_list[index]

        image = Image.open(image_file_name)
        matte = Image.open(matte_file_name)

        data = {'image': image, 'gt_matte': matte}

        # 根据传入列表 变换图像
        if self.transform:
            data = self.transform(data)
            #--------------------------------------------------------------------------
            # 经过各种变换后
            # data = {'image': image, 'trimap': trimap, 'gt_matte': gt_matte}
            #--------------------------------------------------------------------------
        return data

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
        sample['gt_matte'] = sample['gt_matte'] / 255
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
