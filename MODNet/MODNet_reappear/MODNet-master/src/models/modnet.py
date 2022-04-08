import torch
import torch.nn as nn
import torch.nn.functional as F

from .backbones import SUPPORTED_BACKBONES


#------------------------------------------------------------------------------
#  MODNet Basic Modules
#------------------------------------------------------------------------------

class IBNorm(nn.Module):
    """ Combine Instance Norm and Batch Norm into One Layer
    """

    def __init__(self, in_channels):
        super(IBNorm, self).__init__()
        in_channels = in_channels
        self.bnorm_channels = int(in_channels / 2)
        self.inorm_channels = in_channels - self.bnorm_channels

        self.bnorm = nn.BatchNorm2d(self.bnorm_channels, affine=True)
        self.inorm = nn.InstanceNorm2d(self.inorm_channels, affine=False)
        
    def forward(self, x):
        bn_x = self.bnorm(x[:, :self.bnorm_channels, ...].contiguous())
        in_x = self.inorm(x[:, self.bnorm_channels:, ...].contiguous())

        return torch.cat((bn_x, in_x), 1)

# 2d空洞卷积 实例标准化 激活 三合一
class Conv2dIBNormRelu(nn.Module):
    """ Convolution + IBNorm + ReLu
    """

    def __init__(self, in_channels, out_channels, kernel_size, 
                 stride=1, padding=0, dilation=1, groups=1, bias=True, 
                 with_ibn=True, with_relu=True):
        super(Conv2dIBNormRelu, self).__init__()

        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size, 
                      stride=stride, padding=padding, dilation=dilation, 
                      groups=groups, bias=bias)     # 通过groups = in_channels 可以实现深度可分离卷积
        ]

        if with_ibn:       
            layers.append(IBNorm(out_channels))
        if with_relu:
            layers.append(nn.ReLU(inplace=True))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


# 增强图像通道间相关性
class SEBlock(nn.Module):
    """ SE Block Proposed in https://arxiv.org/pdf/1709.01507.pdf 
    """
    def __init__(self, in_channels, out_channels, reduction=1):
        super(SEBlock, self).__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)     # 池化最后一个维度， 保持前面维度形状
        self.fc = nn.Sequential(
            nn.Linear(in_channels, int(in_channels // reduction), bias=False),  # 320个 线性变换 y = xA^T + b 压缩维度
            nn.ReLU(inplace=True),  # 激活函数max(0,x)
            nn.Linear(int(in_channels // reduction), out_channels, bias=False), # 1280个 线性变换展开维度
            nn.Sigmoid()    # 重映射到(0,1)     1/1+exp(-x)
        )
    
    def forward(self, x):
        b, c, _, _ = x.size()

        #=============================================================
        # print(f'b = {b},c = {c}, x.size = {x.size()}')
        #=============================================================

        w = self.pool(x).view(b, c)         # 池化后重新整合结构为[1, 1280]
        # print(f'pool and view(b, c) = {w} ')
        w = self.fc(w).view(b, c, 1, 1)     # 全连接层 通过线性变换增强所有维度间的关系  扩展维度

        return x * w.expand_as(x)       # 将w复制到与x的size一致。作为x的系数


#------------------------------------------------------------------------------
#  MODNet Branches
#------------------------------------------------------------------------------

class LRBranch(nn.Module):
    """ Low Resolution Branch of MODNet
    """

    def __init__(self, backbone):
        super(LRBranch, self).__init__()
        enc_channels = backbone.enc_channels    # 主干网络各阶段通道数  [16, 24, 32, 96, 1280]
        self.backbone = backbone
        self.se_block = SEBlock(enc_channels[4], enc_channels[4], reduction=4)  # 增强通道间联系
        self.conv_lr16x = Conv2dIBNormRelu(enc_channels[4], enc_channels[3], 5, stride=1, padding=2)    # 5*5空洞卷积 缩减通道数
        self.conv_lr8x = Conv2dIBNormRelu(enc_channels[3], enc_channels[2], 5, stride=1, padding=2)      # 5*5空洞卷积 缩减通道数
        self.conv_lr = Conv2dIBNormRelu(enc_channels[2], 1, kernel_size=3, stride=2, padding=1, with_ibn=False, with_relu=False)    # 压缩为单通道

    def forward(self, img, inference):
        enc_features = self.backbone.forward(img)   # 执行主干网络，即 mobilenet_v2 返回五个阶段的tensor
        # #==========================================================================
        # import numpy as np
        # a        = np.array(enc_features)
        # print(f'enc_features.shape = {a[0].shape}')
        # # [1, 16, 256, *]
        # #=========================================================================
        enc2x, enc4x, enc32x = enc_features[0], enc_features[1], enc_features[4]
        # ================================================================================
        #   e-ASPP  efficiency - Atrous Spatial Pyramid Pooling 高效空间金字塔空洞卷积
        # ================================================================================
        enc32x = self.se_block(enc32x)  # 增强第五阶段结果通道间的联系
        lr16x = F.interpolate(enc32x, scale_factor=2, mode='bilinear', align_corners=False) # 扩展数据量
        lr16x = self.conv_lr16x(lr16x)
        lr8x = F.interpolate(lr16x, scale_factor=2, mode='bilinear', align_corners=False)
        lr8x = self.conv_lr8x(lr8x)
        # tensor ---> numpy ---> Image.show()
        # from PIL import Image
        # pred_semantic = pred_semantic[0][0].data.cpu().numpy()
        # Image.fromarray(((pred_semantic * 255).astype('uint8')), mode='L').show()

        # 获取pred_semantic结果
        pred_semantic = None
        if not inference:   # 非预测时
            lr = self.conv_lr(lr8x)     # 压缩通道信息
            pred_semantic = torch.sigmoid(lr)   # 激活语义预分割

            # # tensor ---> numpy ---> Image.show()
            # from PIL import Image
            # pred_semantic = pred_semantic[0][0].data.cpu().numpy()
            # Image.fromarray(((pred_semantic * 255).astype('uint8')), mode='L').show()


        return pred_semantic, lr8x, [enc2x, enc4x]  # 返回 预处理语义分割， e-ASPP结果, [主干网络第1层结果， 主干网络第2层结果]


class HRBranch(nn.Module):
    """ High Resolution Branch of MODNet
    """
    # hr_channels = 32      enc_channels = [16, 24, 32, 96, 1280]
    def __init__(self, hr_channels, enc_channels):
        super(HRBranch, self).__init__()
        # 通道数中的+3 是拼接了原图的RGB通道
        # 卷积 16 ---> 32
        self.tohr_enc2x = Conv2dIBNormRelu(enc_channels[0], hr_channels, 1, stride=1, padding=0)    # 升维
        self.conv_enc2x = Conv2dIBNormRelu(hr_channels + 3, hr_channels, 3, stride=2, padding=1)

        # 24 ---> 36
        self.tohr_enc4x = Conv2dIBNormRelu(enc_channels[1], hr_channels, 1, stride=1, padding=0)        # 升维
        self.conv_enc4x = Conv2dIBNormRelu(2 * hr_channels, 2 * hr_channels, 3, stride=1, padding=1)    # 卷积

        # 通道数的倍数取决于拼接的层数
        self.conv_hr4x = nn.Sequential(
            Conv2dIBNormRelu(3 * hr_channels + 3, 2 * hr_channels, 3, stride=1, padding=1),
            Conv2dIBNormRelu(2 * hr_channels, 2 * hr_channels, 3, stride=1, padding=1),
            Conv2dIBNormRelu(2 * hr_channels, hr_channels, 3, stride=1, padding=1),
        )
        # 64 ---> 32
        self.conv_hr2x = nn.Sequential(
            Conv2dIBNormRelu(2 * hr_channels, 2 * hr_channels, 3, stride=1, padding=1),
            Conv2dIBNormRelu(2 * hr_channels, hr_channels, 3, stride=1, padding=1),
            Conv2dIBNormRelu(hr_channels, hr_channels, 3, stride=1, padding=1),
            Conv2dIBNormRelu(hr_channels, hr_channels, 3, stride=1, padding=1),
        )

        self.conv_hr = nn.Sequential(
            Conv2dIBNormRelu(hr_channels + 3, hr_channels, 3, stride=1, padding=1),
            Conv2dIBNormRelu(hr_channels, 1, kernel_size=1, stride=1, padding=0, with_ibn=False, with_relu=False),  # 降维，压缩单通道输出
        )
    # detail分支的通道数始终维持在32以减少运算开销 并降低到了原始分辨率的1/4
    def forward(self, img, enc2x, enc4x, lr8x, inference):
        img2x = F.interpolate(img, scale_factor=1/2, mode='bilinear', align_corners=False)  # 压缩size
        img4x = F.interpolate(img, scale_factor=1/4, mode='bilinear', align_corners=False)  # 压缩size

        # 将semantic分支传递来的第二个特征层上采样拼接到当前分支
        enc2x = self.tohr_enc2x(enc2x)
        hr4x = self.conv_enc2x(torch.cat((img2x, enc2x), dim=1))
        # 将semantic分支传递来的第三个特征层上采样拼接到当前分支
        enc4x = self.tohr_enc4x(enc4x)
        hr4x = self.conv_enc4x(torch.cat((hr4x, enc4x), dim=1))
        # 将semantic分支的预测结果上采样结果和原图1/4拼接到当前分支
        lr4x = F.interpolate(lr8x, scale_factor=2, mode='bilinear', align_corners=False)
        hr4x = self.conv_hr4x(torch.cat((hr4x, lr4x, img4x), dim=1))
        # 还原size将semantic分支传递来的第二个特征层上采样拼接到当前分支并卷积
        hr2x = F.interpolate(hr4x, scale_factor=2, mode='bilinear', align_corners=False)
        hr2x = self.conv_hr2x(torch.cat((hr2x, enc2x), dim=1))

        # 为训练返回detail的预测
        pred_detail = None
        if not inference:
            hr = F.interpolate(hr2x, scale_factor=2, mode='bilinear', align_corners=False)  # 还原size
            hr = self.conv_hr(torch.cat((hr, img), dim=1))  # Skip Link
            pred_detail = torch.sigmoid(hr)     # 激活

        return pred_detail, hr2x


class FusionBranch(nn.Module):
    """ Fusion Branch of MODNet
    """
    #   hr_channels = 32    enc_channels = [16, 24, 32, 96, 1280]
    def __init__(self, hr_channels, enc_channels):
        super(FusionBranch, self).__init__()
        # 统一LR和HR的通道数并连接
        self.conv_lr4x = Conv2dIBNormRelu(enc_channels[2], hr_channels, 5, stride=1, padding=2)
        self.conv_f2x = Conv2dIBNormRelu(2 * hr_channels, hr_channels, 3, stride=1, padding=1)
        self.conv_f = nn.Sequential(
            Conv2dIBNormRelu(hr_channels + 3, int(hr_channels / 2), 3, stride=1, padding=1),
            # 降通道数
            Conv2dIBNormRelu(int(hr_channels / 2), 1, 1, stride=1, padding=0, with_ibn=False, with_relu=False),
        )

    def forward(self, img, lr8x, hr2x):
        lr4x = F.interpolate(lr8x, scale_factor=2, mode='bilinear', align_corners=False)    # 上采样
        lr4x = self.conv_lr4x(lr4x) # 卷积 统一通道数
        lr2x = F.interpolate(lr4x, scale_factor=2, mode='bilinear', align_corners=False)    # 上采样

        f2x = self.conv_f2x(torch.cat((lr2x, hr2x), dim=1)) # 拼接LR与HR并卷积
        f = F.interpolate(f2x, scale_factor=2, mode='bilinear', align_corners=False)    # 上采样
        f = self.conv_f(torch.cat((f, img), dim=1)) # 拼接f和原图 卷积
        pred_matte = torch.sigmoid(f)   # 激活

        return pred_matte   # 返回预测


#------------------------------------------------------------------------------
#  MODNet
#------------------------------------------------------------------------------

class MODNet(nn.Module):
    """ Architecture of MODNet
    """
#=========================================================
#   在其他domain训练时不要加载预训练参数
#   主干网络通过backbones的初始化文件__init__.py加载并传递初始化参数
#   包含 backbone.enc_channels，   enc*x
#=========================================================
    def __init__(self, in_channels=3, hr_channels=32, backbone_arch='mobilenetv2', backbone_pretrained=False):
        super(MODNet, self).__init__()

        self.in_channels = in_channels
        self.hr_channels = hr_channels
        self.backbone_arch = backbone_arch
        self.backbone_pretrained = backbone_pretrained

        self.backbone = SUPPORTED_BACKBONES[self.backbone_arch](self.in_channels)

        self.lr_branch = LRBranch(self.backbone)                                    # 初始化低分辨率分支
        self.hr_branch = HRBranch(self.hr_channels, self.backbone.enc_channels)     # 初始化高分辨率分支
        self.f_branch = FusionBranch(self.hr_channels, self.backbone.enc_channels)  # 初始化融合分支

        # 初始化权重
        for m in self.modules():
            if isinstance(m, nn.Conv2d):    # 卷积层
                self._init_conv(m)
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.InstanceNorm2d): # 标准化层
                self._init_norm(m)

        # 加载预训练
        if self.backbone_pretrained:
            self.backbone.load_pretrained_ckpt()    # 再wrapper中设置预训练模型路径

    def forward(self, img, inference):
        pred_semantic, lr8x, [enc2x, enc4x] = self.lr_branch(img, inference)    # 低分辨率语义分割分支
        pred_detail, hr2x = self.hr_branch(img, enc2x, enc4x, lr8x, inference)  # 高分辨率细节处理分支
        pred_matte = self.f_branch(img, lr8x, hr2x)                             # 融合分支
        # print(pred_semantic)
        return pred_semantic, pred_detail, pred_matte
    
    def freeze_norm(self):
        norm_types = [nn.BatchNorm2d, nn.InstanceNorm2d]
        for m in self.modules():
            for n in norm_types:
                if isinstance(m, n):
                    m.eval()
                    continue

    def _init_conv(self, conv):
        nn.init.kaiming_uniform_(
            conv.weight, a=0, mode='fan_in', nonlinearity='relu')
        if conv.bias is not None:
            nn.init.constant_(conv.bias, 0)

    def _init_norm(self, norm):
        if norm.weight is not None:
            nn.init.constant_(norm.weight, 1)
            nn.init.constant_(norm.bias, 0)
