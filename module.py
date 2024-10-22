import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


import torch.nn as nn
import torch.nn.functional as F


# 定义优化后的残差块 ResBlock
class ResBlock(nn.Module):
    def __init__(self, in_channel, out_channel, stride=1):
        super().__init__()
        # 残差块的连续卷积层
        self.left = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=stride, padding=(0, 1), bias=False),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channel, out_channel, kernel_size=3, stride=1, padding=(0, 1), bias=False),
            nn.BatchNorm2d(out_channel),
        )

        if stride != 1 or in_channel != out_channel:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channel),  # 加入 BatchNorm 提高稳定性
            )
        else:
            self.shortcut = nn.Identity()  # 不需要处理时直接输出

    def forward(self, x):
        out = self.left(x)
        out += self.shortcut(x)  # 残差连接
        out = F.relu(out)  # 保持原有结构的激活函数

        return out


class ResNet(nn.Module):
    def __init__(self, ResBlock, num_classes=1000):
        super().__init__()
        self.in_channel = 64
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )
        self.layer1 = self.make_layer(ResBlock, 64, 2, stride=1)
        self.layer2 = self.make_layer(ResBlock, 128, 2, stride=2)
        self.layer3 = self.make_layer(ResBlock, 256, 2, stride=2)
        self.layer4 = self.make_layer(ResBlock, 512, 2, stride=2)
        self.fc = nn.Linear(512, num_classes)

    # 重复同一个残差块
    def make_layer(self, block, channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channel, channels, stride))
            self.in_channel = channels
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)  # 全局平均池化层
        out = out.view(out.size(0), -1)
        return out


class Demo(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.resnet_block = ResBlock(3, 3)
        self.fc = nn.Linear(3, 1)

    def forward(self, x):
        out = self.resnet_block(x)
        out = self.fc(out)
        return out
