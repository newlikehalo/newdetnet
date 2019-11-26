import torch
import torch.nn as nn
import torch.nn.functional as F
import ipdb
from torch.autograd import Variable


class RES_BLOCK(nn.Module):
    def __init__(self, in_channels, filters_list, strides=1, use_bias=True):
        super(RES_BLOCK, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=filters_list[0], kernel_size=1, stride=1,
                               bias=False)
        self.conv1_b = nn.BatchNorm2d(filters_list[0])
        self.conv1_r = nn.ReLU()

        self.conv2 = nn.Conv2d(in_channels=filters_list[0], out_channels=filters_list[1], kernel_size=3, stride=1,padding=1,
                               bias=False)
        self.conv2_b = nn.BatchNorm2d(filters_list[1])
        self.conv2_r = nn.ReLU()

        self.conv3 = nn.Conv2d(in_channels=filters_list[1], out_channels=filters_list[2], kernel_size=1, stride=1,
                               bias=False)
        self.conv3_b = nn.BatchNorm2d(filters_list[2])

        # 输入x

        self.out = nn.ReLU()

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv1_b(x1)
        x3 = self.conv1_r(x2)
        x4 = self.conv2(x3)
        x5 = self.conv2_b(x4)
        x6 = self.conv2_r(x5)
        x7 = self.conv3(x6)
        x8 = self.conv3_b(x7)
        out = torch.add(x8, x)
        out = self.out(out)
        return out


class RES_BLOCK_PROJ(nn.Module):
    def __init__(self, in_channels, filters_list, strides=2):
        super(RES_BLOCK_PROJ, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=filters_list[0], kernel_size=1, stride=strides,
                               bias=False)
        self.conv1_b = nn.BatchNorm2d(filters_list[0])
        self.conv1_r = nn.ReLU()

        self.conv2 = nn.Conv2d(in_channels=filters_list[0], out_channels=filters_list[1], kernel_size=3, stride=1,
                               padding=1,
                               bias=False)
        self.conv2_b = nn.BatchNorm2d(filters_list[1])
        self.conv2_r = nn.ReLU()

        self.conv3 = nn.Conv2d(in_channels=filters_list[1], out_channels=filters_list[2], kernel_size=1, stride=1,
                               bias=False)
        self.conv3_b = nn.BatchNorm2d(filters_list[2])

        # 输入x
        self.x1 = nn.Conv2d(in_channels=in_channels, out_channels=filters_list[2], kernel_size=1, stride=strides,
                            bias=False)
        self.x2 = nn.BatchNorm2d(filters_list[2])

        self.out = nn.ReLU()

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv1_b(x1)
        x3 = self.conv1_r(x2)
        x4 = self.conv2(x3)
        x5 = self.conv2_b(x4)
        x6 = self.conv2_r(x5)
        x7 = self.conv3(x6)
        x8 = self.conv3_b(x7)
        x9 = self.x1(x)
        x10 = self.x2(x9)
        out = torch.add(x8, x10)
        out = self.out(out)
        return out


class DILATED_RES_BLOCK(nn.Module):
    def __init__(self, in_channels, filters_list, strides=1, use_bias=True, name=None):
        super(DILATED_RES_BLOCK, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=filters_list[0], kernel_size=1, stride=1,
                               bias=False)
        self.conv1_b = nn.BatchNorm2d(filters_list[0])
        self.conv1_r = nn.ReLU()

        self.conv2 = nn.Conv2d(in_channels=filters_list[0], out_channels=filters_list[1], kernel_size=3, dilation=2,
                               stride=1,
                               padding=2,
                               bias=False)
        self.conv2_b = nn.BatchNorm2d(filters_list[1])
        self.conv2_r = nn.ReLU()

        self.conv3 = nn.Conv2d(in_channels=filters_list[1], out_channels=filters_list[2], kernel_size=1, stride=1,
                               bias=False)
        self.conv3_b = nn.BatchNorm2d(filters_list[2])

        self.out = nn.ReLU()

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv1_b(x1)
        x3 = self.conv1_r(x2)
        x4 = self.conv2(x3)
        x5 = self.conv2_b(x4)
        x6 = self.conv2_r(x5)
        x7 = self.conv3(x6)
        x8 = self.conv3_b(x7)

        out = torch.add(x8, x)
        out = self.out(out)
        return out


class DILATED_RES_BLOCK_PROJ(nn.Module):
    def __init__(self, in_channels, filters_list, strides=2, use_bias=True, name=None):
        super(DILATED_RES_BLOCK_PROJ, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=filters_list[0], kernel_size=1, stride=1,
                               bias=False)
        self.conv1_b = nn.BatchNorm2d(filters_list[0])
        self.conv1_r = nn.ReLU()

        self.conv2 = nn.Conv2d(in_channels=filters_list[0], out_channels=filters_list[1], kernel_size=3, dilation=2,
                               stride=1,
                               padding=2,
                               bias=False)
        self.conv2_b = nn.BatchNorm2d(filters_list[1])
        self.conv2_r = nn.ReLU()

        self.conv3 = nn.Conv2d(in_channels=filters_list[1], out_channels=filters_list[2], kernel_size=1, stride=1,
                               bias=False)
        self.conv3_b = nn.BatchNorm2d(filters_list[1])

        # 输入x
        self.x1 = nn.Conv2d(in_channels=in_channels, out_channels=filters_list[2], kernel_size=1, stride=1,
                            bias=False)
        self.x2 = nn.BatchNorm2d(filters_list[2])

        self.out = nn.ReLU()

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv1_b(x1)
        x3 = self.conv1_r(x2)
        x4 = self.conv2(x3)
        x5 = self.conv2_b(x4)
        x6 = self.conv2_r(x5)
        x7 = self.conv3(x6)
        x8 = self.conv3_b(x7)
        x9 = self.x1(x)
        x10 = self.x2(x9)
        out = torch.add(x8, x10)
        out = self.out(out)
        return out


class ResNet_Body(nn.Module):
    def __init__(self, in_channels, filters_list, num_blocks, strides):
        super(ResNet_Body, self).__init__()
        blocks_list = [1, 3, 4, 6, 3, 3]
        self.input = input
        self.filters_list = filters_list
        self.stride = strides
        self.num=num_blocks
        self.num_blocks = blocks_list[num_blocks]

        self.res_block_proj = RES_BLOCK_PROJ(in_channels=in_channels, filters_list=filters_list, strides=2)
        if num_blocks == 1:
            self.res_block1 = RES_BLOCK(in_channels=256, filters_list=filters_list, strides=2)
            self.res_block2 = RES_BLOCK(in_channels=256, filters_list=filters_list, strides=2)
            self.res_block3 = RES_BLOCK(in_channels=256, filters_list=filters_list, strides=2)

        elif num_blocks == 2:
            self.res_block1 = RES_BLOCK(in_channels=512, filters_list=filters_list, strides=2)
            self.res_block2 = RES_BLOCK(in_channels=512, filters_list=filters_list, strides=2)
            self.res_block3 = RES_BLOCK(in_channels=512, filters_list=filters_list, strides=2)
            self.res_block4 = RES_BLOCK(in_channels=512, filters_list=filters_list, strides=2)
        elif num_blocks == 3:
            self.res_block1 = RES_BLOCK(in_channels=1024, filters_list=filters_list, strides=2)
            self.res_block2 = RES_BLOCK(in_channels=1024, filters_list=filters_list, strides=2)
            self.res_block3 = RES_BLOCK(in_channels=1024, filters_list=filters_list, strides=2)
            self.res_block4 = RES_BLOCK(in_channels=1024, filters_list=filters_list, strides=2)
            self.res_block5 = RES_BLOCK(in_channels=1024, filters_list=filters_list, strides=2)
            self.res_block6 = RES_BLOCK(in_channels=1024, filters_list=filters_list, strides=2)
        elif num_blocks == 4:
            self.res_block1 = RES_BLOCK(in_channels=256, filters_list=filters_list, strides=2)
            self.res_block2 = RES_BLOCK(in_channels=256, filters_list=filters_list, strides=2)
            self.res_block3 = RES_BLOCK(in_channels=256, filters_list=filters_list, strides=2)
        elif num_blocks == 5:
            self.res_block1 = RES_BLOCK(in_channels=256, filters_list=filters_list, strides=2)
            self.res_block2 = RES_BLOCK(in_channels=256, filters_list=filters_list, strides=2)
            self.res_block3 = RES_BLOCK(in_channels=256, filters_list=filters_list, strides=2)

    def forward(self, x):
        out = self.res_block_proj(x)
        if self.num == 1:
            x1 = self.res_block1(out)
            x2 = self.res_block2(x1)
            x3 = self.res_block3(x2)
            return x3

        elif self.num == 2:
            x1 = self.res_block1(out)
            x2 = self.res_block2(x1)
            x3 = self.res_block3(x2)
            x4 = self.res_block4(x3)
            return x4
        elif self.num == 3:
            x1 = self.res_block1(out)
            x2 = self.res_block2(x1)
            x3 = self.res_block3(x2)
            x4 = self.res_block4(x3)
            x5 = self.res_block5(x4)
            x6 = self.res_block6(x5)
            return x6
        elif self.num == 4:
            x1 = self.res_block1(out)
            x2 = self.res_block2(x1)
            x3 = self.res_block3(x2)
            return x3

        elif self.num == 5:
            x1 = self.res_block1(out)
            x2 = self.res_block2(x1)
            x3 = self.res_block3(x2)
            return x3


class DETNET_BODY(nn.Module):
    def __init__(self, in_channels, filters_list, num_blocks, strides):
        super(DETNET_BODY, self).__init__()
        blocks_list = [1, 3, 4, 6, 3, 3]
        self.input = input
        self.filters_list = filters_list
        self.stride = strides
        self.num = num_blocks
        self.num_blocks = blocks_list[num_blocks]

        self.res_block_proj = DILATED_RES_BLOCK_PROJ(in_channels=in_channels, filters_list=filters_list, strides=1)
        if num_blocks == 1:
            self.res_block1 = DILATED_RES_BLOCK(in_channels=256, filters_list=filters_list, strides=2)
            self.res_block2 = DILATED_RES_BLOCK(in_channels=256, filters_list=filters_list, strides=2)
            self.res_block3 = DILATED_RES_BLOCK(in_channels=256, filters_list=filters_list, strides=2)

        elif num_blocks == 2:
            self.res_block1 = DILATED_RES_BLOCK(in_channels=512, filters_list=filters_list, strides=2)
            self.res_block2 = DILATED_RES_BLOCK(in_channels=512, filters_list=filters_list, strides=2)
            self.res_block3 = DILATED_RES_BLOCK(in_channels=512, filters_list=filters_list, strides=2)
            self.res_block4 = DILATED_RES_BLOCK(in_channels=512, filters_list=filters_list, strides=2)
        elif num_blocks == 3:
            self.res_block1 = DILATED_RES_BLOCK(in_channels=1024, filters_list=filters_list, strides=2)
            self.res_block2 = DILATED_RES_BLOCK(in_channels=1024, filters_list=filters_list, strides=2)
            self.res_block3 = DILATED_RES_BLOCK(in_channels=1024, filters_list=filters_list, strides=2)
            self.res_block4 = DILATED_RES_BLOCK(in_channels=1024, filters_list=filters_list, strides=2)
            self.res_block5 = DILATED_RES_BLOCK(in_channels=1024, filters_list=filters_list, strides=2)
            self.res_block6 = DILATED_RES_BLOCK(in_channels=1024, filters_list=filters_list, strides=2)
        elif num_blocks == 4:
            self.res_block1 = DILATED_RES_BLOCK(in_channels=256, filters_list=filters_list, strides=2)
            self.res_block2 = DILATED_RES_BLOCK(in_channels=256, filters_list=filters_list, strides=2)
            self.res_block3 = DILATED_RES_BLOCK(in_channels=256, filters_list=filters_list, strides=2)
        elif num_blocks == 5:
            self.res_block1 = DILATED_RES_BLOCK(in_channels=256, filters_list=filters_list, strides=2)
            self.res_block2 = DILATED_RES_BLOCK(in_channels=256, filters_list=filters_list, strides=2)
            self.res_block3 = DILATED_RES_BLOCK(in_channels=256, filters_list=filters_list, strides=2)

    def forward(self, x):
        out = self.res_block_proj(x)
        if self.num == 1:
            x1 = self.res_block1(out)
            x2 = self.res_block2(x1)
            x3 = self.res_block3(x2)
            return x3

        elif self.num == 2:
            x1 = self.res_block1(out)
            x2 = self.res_block2(x1)
            x3 = self.res_block3(x2)
            x4 = self.res_block4(x3)
            return x4
        elif self.num == 3:
            x1 = self.res_block1(out)
            x2 = self.res_block2(x1)
            x3 = self.res_block3(x2)
            x4 = self.res_block4(x3)
            x5 = self.res_block5(x4)
            x6 = self.res_block6(x5)
            return x6
        elif self.num == 4:
            x1 = self.res_block1(out)
            x2 = self.res_block2(x1)
            x3 = self.res_block3(x2)
            return x3

        elif self.num == 5:
            x1 = self.res_block1(out)
            x2 = self.res_block2(x1)
            x3 = self.res_block3(x2)
            return x3


class detnet_59(nn.Module):
    def __init__(self):
        super(detnet_59, self).__init__()
        filters_list = [[64],
                        [64, 64, 256],
                        [128, 128, 512],
                        [256, 256, 1024],
                        [256, 256, 256],
                        [256, 256, 256]]

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=filters_list[0][0], kernel_size=7, padding=3, stride=2,
                               bias=False)
        self.conv1_b = nn.BatchNorm2d(64)
        self.conv1_r = nn.ReLU()
        self.conv1_pool = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)

        self.conv2_x = ResNet_Body(in_channels=64, filters_list=filters_list[1], num_blocks=1, strides=2)
        self.conv3_x = ResNet_Body(in_channels=256, filters_list=filters_list[2], num_blocks=2, strides=2)
        self.conv4_x = ResNet_Body(in_channels=512, filters_list=filters_list[3], num_blocks=3, strides=2)
        self.conv5_x = DETNET_BODY(in_channels=1024, filters_list=filters_list[4], num_blocks=4, strides=2)
        self.conv6_x = DETNET_BODY(in_channels=256, filters_list=filters_list[5], num_blocks=5, strides=2)

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv1_b(x1)
        x3 = self.conv1_r(x2)
        x4 = self.conv1_pool(x3)
        x5 = self.conv2_x(x4)
        x6 = self.conv3_x(x5)
        x7 = self.conv4_x(x6)
        x8 = self.conv5_x(x7)
        x9 = self.conv6_x(x8)
        x10=x8+x9
        # ipdb.set_trace()
        return x10


if __name__ == '__main__':
    net = detnet_59()
    ipdb.set_trace()
