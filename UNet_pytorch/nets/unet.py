import torch
import torch.nn as nn

# 下采样块
class ConvBNReLUDown(nn.Module):
    def __init__(self, in_ch, out_ch, groups = 1, k_size = 3, stride = 1, padding = 1):
        super().__init__()
        self.Conv_BN_ReLU = nn.Sequential(
            nn.Conv2d(in_channels=in_ch,
                      out_channels=out_ch,
                      kernel_size=k_size,
                      stride=stride,
                      padding=1,
                      groups=groups),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(),
            nn.Conv2d(in_channels=out_ch,
                      out_channels=out_ch,
                      kernel_size=k_size,
                      stride=stride,
                      padding=1,
                      groups=groups),
            nn.BatchNorm2d(out_ch),
            nn.ReLU()
        )
        self.Down_sample = nn.Sequential(
            nn.Conv2d(in_channels=out_ch,out_channels=out_ch,kernel_size=3,stride=2,padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU()
        )

    def forwad(self, img):
        feature = self.Conv_BN_ReLU(img)
        result = self.Down_sample(feature)
        return feature, result

class UNet(nn.Module):
    def __init__(self, in_ch):
        super().__init__()
        self.channels = [64, 128, 256, 512, 1024]
        # Down-sample
        self.conv_1 = ConvBNReLUDown(in_ch, self.channels[0])
        self.conv_2 = ConvBNReLUDown(self.channels[0], self.channels[1])
        self.conv_3 = ConvBNReLUDown(self.channels[1], self.channels[2])
        self.conv_4 = ConvBNReLUDown(self.channels[2], self.channels[3])

        self.features = []  # 存储特各层结果

    def forwad(self, img):
