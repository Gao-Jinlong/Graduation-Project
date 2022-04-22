#--------------------------------------------#
#   该部分代码只用于看网络结构，并非测试代码
#--------------------------------------------#
from torchsummary import summary
from thop import profile
import torch
from models.modnet_sum import MODNet

if __name__ == "__main__":
    model = MODNet().train().cuda()
    # print("model have {} paramerters in total".format(sum(x.numel() for x in model.parameters())))
    summary(model, (3, 512, 512))

    inputs = torch.randn(1, 3, 512,512).cuda()
    flops, params = profile(model, (inputs,))
    print('flops: ', flops, 'params: ', params)