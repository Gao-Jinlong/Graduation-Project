import math
import json
import time
import scipy    # 数值计算库
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import grey_dilation, grey_erosion

from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F


__all__ = [
    'supervised_training_iter',
    'soc_adaptation_iter',
]


# ----------------------------------------------------------------------------------
# Tool Classes/Functions
# ----------------------------------------------------------------------------------

class GaussianBlurLayer(nn.Module):
    """ Add Gaussian Blur to a 4D tensors
    This layer takes a 4D tensor of {N, C, H, W} as input.
    The Gaussian blur will be performed in given channel number (C) splitly.
    """

    def __init__(self, channels, kernel_size):
        """ 
        Arguments:
            channels (int): Channel for input tensor
            kernel_size (int): Size of the kernel used in blurring
        """

        super(GaussianBlurLayer, self).__init__()
        self.channels = channels
        self.kernel_size = kernel_size
        assert self.kernel_size % 2 != 0

        self.op = nn.Sequential(
            nn.ReflectionPad2d(math.floor(self.kernel_size / 2)),  # 镜像边缘
            nn.Conv2d(channels, channels, self.kernel_size, 
                      stride=1, padding=0, bias=None, groups=channels)
        )
        # 初始化卷积核
        self._init_kernel()

    def forward(self, x):
        """
        Arguments:
            x (torch.Tensor): input 4D tensor
        Returns:
            torch.Tensor: Blurred version of the input 
        """

        if not len(list(x.shape)) == 4:
            print('\'GaussianBlurLayer\' requires a 4D tensor as input\n')
            exit()
        elif not x.shape[1] == self.channels:
            print('In \'GaussianBlurLayer\', the required channel ({0}) is'
                  'not the same as input ({1})\n'.format(self.channels, x.shape[1]))
            exit()
            
        return self.op(x)

    # 高斯卷积核
    def _init_kernel(self):
        sigma = 0.3 * ((self.kernel_size - 1) * 0.5 - 1) + 0.8  # 标准差

        n = np.zeros((self.kernel_size, self.kernel_size))
        i = math.floor(self.kernel_size / 2)    # 向下取整
        n[i, i] = 1     # 卷积核中心为1
        kernel = scipy.ndimage.gaussian_filter(n, sigma)    # 图像高斯滤波
        # print(f"the gaussian kernel is\n{kernel}")
        for name, param in self.named_parameters():
            param.data.copy_(torch.from_numpy(kernel))

# ----------------------------------------------------------------------------------


# ----------------------------------------------------------------------------------
# MODNet Training Functions
# ----------------------------------------------------------------------------------

blurer = GaussianBlurLayer(1, 3)  # channel = 1 kernel_size = 3 的高斯卷积层

if torch.cuda.is_available():
    blurer.cuda()

# 监督训练迭代器
def supervised_training_iter(
    modnet, optimizer, image, trimap, gt_matte,
    semantic_scale=10.0, detail_scale=10.0, matte_scale=1.0):
    """ Supervised training iteration of MODNet
    This function trains MODNet for one iteration in a labeled dataset.

    Arguments:
        modnet (torch.nn.Module): instance of MODNet
        optimizer (torch.optim.Optimizer): optimizer for supervised training
        image (torch.autograd.Variable): input RGB image
                                         its pixel values should be normalized
        trimap (torch.autograd.Variable): trimap used to calculate the losses
                                          its pixel values can be 0, 0.5, or 1
                                          (foreground=1, background=0, unknown=0.5)
        gt_matte (torch.autograd.Variable): ground truth alpha matte
                                            its pixel values are between [0, 1]
        semantic_scale (float): scale of the semantic loss
                                NOTE: please adjust according to your dataset
        detail_scale (float): scale of the detail loss
                              NOTE: please adjust according to your dataset
        matte_scale (float): scale of the matte loss
                             NOTE: please adjust according to your dataset
    
    Returns:
        semantic_loss (torch.Tensor): loss of the semantic estimation [Low-Resolution (LR) Branch]
        detail_loss (torch.Tensor): loss of the detail prediction [High-Resolution (HR) Branch]
        matte_loss (torch.Tensor): loss of the semantic-detail fusion [Fusion Branch]

    Example:
        import torch
        from src.models.modnet import MODNet
        from src.trainer import supervised_training_iter

        bs = 16         # batch size
        lr = 0.01       # learn rate
        epochs = 40     # total epochs

        modnet = torch.nn.DataParallel(MODNet()).cuda()
        optimizer = torch.optim.SGD(modnet.parameters(), lr=lr, momentum=0.9)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=int(0.25 * epochs), gamma=0.1)

        dataloader = CREATE_YOUR_DATALOADER(bs)     # NOTE: please finish this function

        for epoch in range(0, epochs):
            for idx, (image, trimap, gt_matte) in enumerate(dataloader):
                semantic_loss, detail_loss, matte_loss = \
                    supervised_training_iter(modnet, optimizer, image, trimap, gt_matte)
            lr_scheduler.step()
    """

    global blurer   # 全局化高斯卷积层

    # set the model to train mode and clear the optimizer
    # 启用训练模式，确保Batch Normalization 和 Dropout 层正常工作。 评价模型时用model.eval()
    modnet.train()  # 修改父类的self.training属性为True 某些类执行时会调用这个属性
    optimizer.zero_grad()   # 梯度归零

    # forward the model
    pred_semantic, pred_detail, pred_matte = modnet(image, False)

    # calculate the boundary mask from the trimap
    boundaries = (trimap < 0.5) + (trimap > 0.5)    # 只有 trimap = 0.5 即边缘才会被标记为False
    ''' # 显示结果  边界部分为黑色 其余为白色
    from PIL import Image
    a = boundaries.cpu()
    b = a[0][0]
    b = b.numpy()
    image2 = Image.fromarray(b)
    image2.show()
    exit()
    '''
    # calculate the semantic loss
    gt_semantic = F.interpolate(gt_matte, scale_factor=1/16, mode='bilinear')   # 下采样 双线性插值
    gt_semantic = blurer(gt_semantic)   # 高斯模糊 匹配预测蒙版
    semantic_loss = torch.mean(F.mse_loss(pred_semantic, gt_semantic))  # 综合每个元素计算均方差后返回平均值
    semantic_loss = semantic_scale * semantic_loss

    # calculate the detail loss
    # 融合boundaries为True的部分(前景+背景)返回trimap值(0)，False则返回pred_detail（预测边缘）值。即非边缘部分返回0 边缘部分返回预测值
    pred_boundary_detail = torch.where(boundaries, trimap, pred_detail)     # attention 预测边界
    gt_detail = torch.where(boundaries, trimap, gt_matte)                   # attention 标注边界
    detail_loss = torch.mean(F.l1_loss(pred_boundary_detail, gt_detail))    # 绝对值差异的平均值
    detail_loss = detail_scale * detail_loss

    # calculate the matte loss
    pred_boundary_matte = torch.where(boundaries, trimap, pred_matte)   # 融合预测蒙版的边界
    matte_l1_loss = F.l1_loss(pred_matte, gt_matte) + 4.0 * F.l1_loss(pred_boundary_matte, gt_matte)    # 蒙版综合损失
    matte_compositional_loss = F.l1_loss(image * pred_matte, image * gt_matte) \
        + 4.0 * F.l1_loss(image * pred_boundary_matte, image * gt_matte)    # 合成图像综合损失
    matte_loss = torch.mean(matte_l1_loss + matte_compositional_loss)
    matte_loss = matte_scale * matte_loss

    # calculate the final loss, backward the loss, and update the model 
    loss = semantic_loss + detail_loss + matte_loss
    loss.backward()
    optimizer.step()    # 执行一次优化

    # for test
    return semantic_loss, detail_loss, matte_loss


def val_iter(
        modnet, image, trimap, gt_matte,
        semantic_scale=10.0, detail_scale=10.0, matte_scale=1.0):

    global blurer  # 全局化高斯卷积层

    # set the model to train mode and clear the optimizer
    # 启用训练模式，确保Batch Normalization 和 Dropout 层正常工作。 评价模型时用model.eval()
    modnet.eval()  # 修改父类的self.training属性为False 某些类执行时会调用这个属性
    # optimizer.zero_grad()  # 梯度归零

    # forward the model
    pred_semantic, pred_detail, pred_matte = modnet(image, False)

    # calculate the boundary mask from the trimap
    boundaries = (trimap < 0.5) + (trimap > 0.5)  # 只有 trimap = 0.5 即边缘才会被标记为False
    ''' # 显示结果  边界部分为黑色 其余为白色
    from PIL import Image
    a = boundaries.cpu()
    b = a[0][0]
    b = b.numpy()
    image2 = Image.fromarray(b)
    image2.show()
    exit()
    '''
    # calculate the semantic loss
    gt_semantic = F.interpolate(gt_matte, scale_factor=1 / 16, mode='bilinear')  # 下采样 双线性插值
    gt_semantic = blurer(gt_semantic)  # 高斯模糊 匹配预测蒙版
    semantic_loss = torch.mean(F.mse_loss(pred_semantic, gt_semantic))  # 综合每个元素计算均方差后返回平均值
    semantic_loss = semantic_scale * semantic_loss

    # calculate the detail loss
    # 融合boundaries为True的部分(前景+背景)返回trimap值(0)，False则返回pred_detail（预测边缘）值。即非边缘部分返回0 边缘部分返回预测值
    pred_boundary_detail = torch.where(boundaries, trimap, pred_detail)  # attention 预测边界
    gt_detail = torch.where(boundaries, trimap, gt_matte)  # attention 标注边界
    detail_loss = torch.mean(F.l1_loss(pred_boundary_detail, gt_detail))  # 绝对值差异的平均值
    detail_loss = detail_scale * detail_loss

    # calculate the matte loss
    pred_boundary_matte = torch.where(boundaries, trimap, pred_matte)  # 融合预测蒙版的边界
    matte_l1_loss = F.l1_loss(pred_matte, gt_matte) + 4.0 * F.l1_loss(pred_boundary_matte, gt_matte)  # 蒙版综合损失
    matte_compositional_loss = F.l1_loss(image * pred_matte, image * gt_matte) \
                               + 4.0 * F.l1_loss(image * pred_boundary_matte, image * gt_matte)  # 合成图像综合损失
    matte_loss = torch.mean(matte_l1_loss + matte_compositional_loss)
    matte_loss = matte_scale * matte_loss
    '''
    # calculate the final loss, backward the loss, and update the model
    loss = semantic_loss + detail_loss + matte_loss
    loss.backward()
    optimizer.step()  # 执行一次优化
    '''
    # for test
    return semantic_loss, detail_loss, matte_loss


def soc_adaptation_iter(
    modnet, backup_modnet, optimizer, image,
    soc_semantic_scale=100.0, soc_detail_scale=1.0):
    """ Self-Supervised sub-objective consistency (SOC) adaptation iteration of MODNet
    This function fine-tunes MODNet for one iteration in an unlabeled dataset.
    Note that SOC can only fine-tune a converged MODNet, i.e., MODNet that has been 
    trained in a labeled dataset.

    Arguments:
        modnet (torch.nn.Module): instance of MODNet
        backup_modnet (torch.nn.Module): backup of the trained MODNet
        optimizer (torch.optim.Optimizer): optimizer for self-supervised SOC 
        image (torch.autograd.Variable): input RGB image
                                         its pixel values should be normalized
        soc_semantic_scale (float): scale of the SOC semantic loss 
                                    NOTE: please adjust according to your dataset
        soc_detail_scale (float): scale of the SOC detail loss
                                  NOTE: please adjust according to your dataset
    
    Returns:
        soc_semantic_loss (torch.Tensor): loss of the semantic SOC
        soc_detail_loss (torch.Tensor): loss of the detail SOC

    Example:
        import copy
        import torch
        from src.models.modnet import MODNet
        from src.trainer import soc_adaptation_iter

        bs = 1          # batch size
        lr = 0.00001    # learn rate
        epochs = 10     # total epochs

        modnet = torch.nn.DataParallel(MODNet()).cuda()
        modnet = LOAD_TRAINED_CKPT()    # NOTE: please finish this function

        optimizer = torch.optim.Adam(modnet.parameters(), lr=lr, betas=(0.9, 0.99))
        dataloader = CREATE_YOUR_DATALOADER(bs)     # NOTE: please finish this function

        for epoch in range(0, epochs):
            backup_modnet = copy.deepcopy(modnet)
            for idx, (image) in enumerate(dataloader):
                soc_semantic_loss, soc_detail_loss = \
                    soc_adaptation_iter(modnet, backup_modnet, optimizer, image)
    """

    global blurer

    # set the backup model to eval mode
    backup_modnet.eval()

    # set the main model to train mode and freeze its norm layers
    modnet.train()
    modnet.module.freeze_norm()

    # clear the optimizer
    optimizer.zero_grad()

    # forward the main model
    pred_semantic, pred_detail, pred_matte = modnet(image, False)

    # forward the backup model
    with torch.no_grad():
        _, pred_backup_detail, pred_backup_matte = backup_modnet(image, False)

    # calculate the boundary mask from `pred_matte` and `pred_semantic`
    pred_matte_fg = (pred_matte.detach() > 0.1).float()
    pred_semantic_fg = (pred_semantic.detach() > 0.1).float()
    pred_semantic_fg = F.interpolate(pred_semantic_fg, scale_factor=16, mode='bilinear')
    pred_fg = pred_matte_fg * pred_semantic_fg

    n, c, h, w = pred_matte.shape
    np_pred_fg = pred_fg.data.cpu().numpy()
    np_boundaries = np.zeros([n, c, h, w])
    for sdx in range(0, n):
        sample_np_boundaries = np_boundaries[sdx, 0, ...]
        sample_np_pred_fg = np_pred_fg[sdx, 0, ...]

        side = int((h + w) / 2 * 0.05)
        dilated = grey_dilation(sample_np_pred_fg, size=(side, side))
        eroded = grey_erosion(sample_np_pred_fg, size=(side, side))

        sample_np_boundaries[np.where(dilated - eroded != 0)] = 1
        np_boundaries[sdx, 0, ...] = sample_np_boundaries

    boundaries = torch.tensor(np_boundaries).float().cuda()

    # sub-objectives consistency between `pred_semantic` and `pred_matte`
    # generate pseudo ground truth for `pred_semantic`
    downsampled_pred_matte = blurer(F.interpolate(pred_matte, scale_factor=1/16, mode='bilinear'))
    pseudo_gt_semantic = downsampled_pred_matte.detach()
    pseudo_gt_semantic = pseudo_gt_semantic * (pseudo_gt_semantic > 0.01).float()
    
    # generate pseudo ground truth for `pred_matte`
    pseudo_gt_matte = pred_semantic.detach()
    pseudo_gt_matte = pseudo_gt_matte * (pseudo_gt_matte > 0.01).float()

    # calculate the SOC semantic loss
    soc_semantic_loss = F.mse_loss(pred_semantic, pseudo_gt_semantic) + F.mse_loss(downsampled_pred_matte, pseudo_gt_matte)
    soc_semantic_loss = soc_semantic_scale * torch.mean(soc_semantic_loss)

    # NOTE: using the formulas in our paper to calculate the following losses has similar results
    # sub-objectives consistency between `pred_detail` and `pred_backup_detail` (on boundaries only)
    backup_detail_loss = boundaries * F.l1_loss(pred_detail, pred_backup_detail, reduction='none')
    backup_detail_loss = torch.sum(backup_detail_loss, dim=(1,2,3)) / torch.sum(boundaries, dim=(1,2,3))
    backup_detail_loss = torch.mean(backup_detail_loss)

    # sub-objectives consistency between pred_matte` and `pred_backup_matte` (on boundaries only)
    backup_matte_loss = boundaries * F.l1_loss(pred_matte, pred_backup_matte, reduction='none')
    backup_matte_loss = torch.sum(backup_matte_loss, dim=(1,2,3)) / torch.sum(boundaries, dim=(1,2,3))
    backup_matte_loss = torch.mean(backup_matte_loss)

    soc_detail_loss = soc_detail_scale * (backup_detail_loss + backup_matte_loss)

    # calculate the final loss, backward the loss, and update the model 
    loss = soc_semantic_loss + soc_detail_loss

    loss.backward()
    optimizer.step()

    return soc_semantic_loss, soc_detail_loss

# ----------------------------------------------------------------------------------


if __name__ == '__main__':
    from matting_dataset_medical import MattingDataset, Rescale, \
        ToTensor, Normalize, ToTrainArray, \
        ConvertImageDtype, GenTrimap
    from torchvision import transforms
    from torch.utils.data import DataLoader
    from models.modnet import MODNet
    from setting import BS, LR, EPOCHS, SEMANTIC_SCALE, DETAIL_SCALE, MATTE_SCALE, SAVE_EPOCH_STEP

    # 图像变换
    transform = transforms.Compose([
        Rescale(512),   # 缩放图片
        GenTrimap(),
        ToTensor(),
        ConvertImageDtype(),
        Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),    # 标准化
        ToTrainArray()
    ])
    # 数据集对象
    # mattingDataset = MattingDataset(data_type='train',transform=transform)
    train_dataset = MattingDataset(data_type='train',transform=transform)
    val_dataset = MattingDataset(data_type='val',transform=transform)

    modnet = torch.nn.DataParallel(MODNet())  # 多卡训练
    if torch.cuda.is_available():   # 转移到GPU （要在构建优化器前执行）
        modnet = modnet.cuda()

    optimizer = torch.optim.SGD(modnet.parameters(), lr=LR, momentum=0.9)   # 随机梯度下降
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=int(0.25 * EPOCHS), gamma=0.1)  # 随epoch衰减学习率

    # 数据加载对象
    train_data = DataLoader(train_dataset,
                            batch_size=BS,  # how many samples per batch to load
                            shuffle=True)   # 乱序
    val_data = DataLoader(val_dataset,
                            batch_size=BS,  # how many samples per batch to load
                            shuffle=True)   # 乱序
    '''
        增加验证集
        增强数据集
    '''
    # 损失记录
    illustration_data = {
        's_loss':[],
        'd_loss':[],
        'm_loss':[],
        'v_s_loss': [],
        'v_d_loss': [],
        'v_m_loss': [],
    }
    # 加载权重
    # modnet.load_state_dict(torch.load('../pretrained/modnet_custom_portrait_matting_last_epoch_weight_epoch60.ckpt', map_location='cpu'))
    for epoch in range(0, EPOCHS):  # 开始训练
        print(f'epoch: {epoch}/{EPOCHS-1}')
        # train
        for idx, (image, trimap, gt_matte) in enumerate(tqdm(train_data)):    # 读入sample
            semantic_loss, detail_loss, matte_loss = \
                supervised_training_iter(modnet, optimizer, image, trimap, gt_matte,    # 调用训练迭代器
                                    semantic_scale=SEMANTIC_SCALE,
                                    detail_scale=DETAIL_SCALE,
                                    matte_scale=MATTE_SCALE)

        lr_scheduler.step()     # 缩小学习率
        # val
        with torch.no_grad():
            for idx, (image, trimap, gt_matte) in enumerate(tqdm(val_data)):
                val_semantic_loss, val_detail_loss, val_matte_loss = \
                    val_iter(modnet, image, trimap, gt_matte,
                             semantic_scale=SEMANTIC_SCALE,
                             detail_scale=DETAIL_SCALE,
                             matte_scale=MATTE_SCALE)

        # 记录损失
        illustration_data['s_loss'].append(float(f'{semantic_loss:.5f}'))
        illustration_data['d_loss'].append(float(f'{detail_loss:.5f}'))
        illustration_data['m_loss'].append(float(f'{matte_loss:.5f}'))
        illustration_data['v_s_loss'].append(float(f'{val_semantic_loss:.5f}'))
        illustration_data['v_d_loss'].append(float(f'{val_detail_loss:.5f}'))
        illustration_data['v_m_loss'].append(float(f'{val_matte_loss:.5f}'))
        with open('../pretrained/loss_data.json', 'w') as f:
            json.dump(illustration_data,f)

        # 保存中间训练结果
        if epoch % SAVE_EPOCH_STEP == 0 and epoch > 1:  # 每一定阶段保存一次
            torch.save({
                'epoch': epoch,     # 保存epoch值
                'model_state_dict': modnet.state_dict(),    # 保存网络状态
                'optimizer_state_dict': optimizer.state_dict(),     # 保存优化器状态
                'loss': {'semantic_loss': semantic_loss, 'detail_loss': detail_loss, 'matte_loss': matte_loss}, # 保存损失值
                'val_loss': {'val_semantic_loss': val_semantic_loss, 'val_detail_loss': val_detail_loss, 'val_matte_loss': val_matte_loss},
            }, f'../pretrained/modnet_custom_portrait_{epoch:2f}_th_loss_{matte_loss:.4f}_val_loss_{val_matte_loss:.4f}.ckpt')     # 文件保存路径
            # 绘图
            plt.figure(figsize=(12, 9))
            plt.plot(illustration_data['s_loss'], 'r-p', label='s_loss')
            plt.plot(illustration_data['d_loss'], 'g-p', label='d_loss')
            plt.plot(illustration_data['m_loss'], 'b-p', label='m_loss')
            plt.plot(illustration_data['v_s_loss'], 'r:p', label='v_s_loss')
            plt.plot(illustration_data['v_d_loss'], 'g:p', label='v_d_loss')
            plt.plot(illustration_data['v_m_loss'], 'b:p', label='v_m_loss')
            plt.legend()
            plt.xlabel('epoch')
            plt.ylabel('loss')
            plt.title('loss chart')
            plt.savefig(f'../pretrained/loss_chart_{epoch:02d}.jpg')
            plt.show()
        print(f'\n{len(train_data)}/{len(train_data)} --- '
              f'semantic_loss: {semantic_loss:f}, detail_loss: {detail_loss:f}, matte_loss: {matte_loss:f}\n'
              f'val_semantic_loss:{val_semantic_loss:f},val_detail_loss: {val_detail_loss:f}, val_matte_loss: {val_matte_loss:f}\n')

    # 绘图
    plt.figure()
    plt.figure(figsize=(12, 9))
    plt.plot(illustration_data['s_loss'],'r-p',label = 's_loss')
    plt.plot(illustration_data['d_loss'],'g-p',label = 'd_loss')
    plt.plot(illustration_data['m_loss'],'b-p',label = 'm_loss')
    plt.plot(illustration_data['v_s_loss'],'r:p',label = 'v_s_loss')
    plt.plot(illustration_data['v_d_loss'],'g:p',label = 'v_d_loss')
    plt.plot(illustration_data['v_m_loss'],'b:p',label = 'v_m_loss')
    plt.legend()
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.title('loss chart')
    plt.savefig('../pretrained/loss_chart.jpg')
    plt.show()
    plt.close('all')

    # 仅保存模型权重参数
    torch.save(modnet.state_dict(), f'../pretrained/modnet_custom_portrait_matting_last_epoch_weight.ckpt')
