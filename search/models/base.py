import torch.nn as nn
import torch
import torch.nn.functional as F


class Identity(nn.Module):
    "A skip layer"
    def __init__(self):
        super(Identity, self).__init__()
    def forward(self, x):
        return x

class Flatten(nn.Module):
    "A flatten layer, save the first row shape"
    def __init__(self):
        super(Flatten, self).__init__()
    def forward(self, x):
        return x.reshape(x.size(0), -1)

def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)

class InvertedResBlock(nn.Module):
    '''expand + depthwise + pointwise'''
    def __init__(self, in_planes, out_planes, stride, kernel, expansion):
        super(InvertedResBlock, self).__init__()
        self.identity = stride == 1 and in_planes == out_planes
        self.stride = stride
        self.multiplier = 1.0
        self.lat = 0
        self.flops = 0
        self.params = 0

        # expand 
        planes = int(round(expansion * in_planes * self.multiplier))
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)

        # depthwise (group conv, 1->1, one channel feature extraction)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=kernel, stride=stride, padding=kernel//2, groups=planes, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        # pointwise (channels feature fusion)
        self.conv3 = nn.Conv2d(planes, out_planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = nn.BatchNorm2d(out_planes)

        # no use, the identity map is controled by self.identity and forward 
        # self.shortcut = nn.Sequential()
        # if stride == 1 and in_planes != out_planes:
        #     self.shortcut = nn.Sequential(
        #         nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, padding=0, bias=False),
        #         nn.BatchNorm2d(out_planes),
        #     )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        if self.identity:
            out = out + x
        return out

class BasicBlock(nn.Module):
    "double conv3 block"
    expansion = 1

    def __init__(self, inp, oup, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inp, oup, stride)
        self.bn1 = nn.BatchNorm2d(oup)
        self.conv2 = conv3x3(oup, oup)
        self.bn2 = nn.BatchNorm2d(oup)
        self.relu = nn.ReLU6(inplace=True)
        self.lat = 0
        self.flops = 0
        self.params = 0

    def forward(self, x):

        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    "? idontknow, this module don't use"
    expansion = 4

    def __init__(self, inplanes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * Bottleneck.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * Bottleneck.expansion)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)
        out = F.relu(out)

        return out

class DownsampleB(nn.Module):
    "downsample in the number of out channels"
    def __init__(self, nIn, nOut, stride):
        super(DownsampleB, self).__init__()
        self.avg = nn.AvgPool2d(stride)
        self.expand_ratio = nOut // nIn

    def forward(self, x):
        x = self.avg(x)
        print(self.expand_ratio)
        return torch.cat([x] + [x.mul(0)] * (self.expand_ratio - 1), 1)



