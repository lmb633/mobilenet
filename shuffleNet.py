import torch
import torch.nn as nn


def channel_shuffle(x, groups):
    batch_size, channels, h, w = x.data.size()
    channels_per_group = channels // groups
    x = x.view(batch_size, groups, channels_per_group, h, w)
    x = torch.transpose(x, 1, 2).contiguous()
    x = x.view(batch_size, -1, h, w)
    return x


def depthwise_conv(i, o, kernel_size, stride=1, padding=0, bias=False):
    return nn.Conv2d(i, o, kernel_size, stride, padding, bias=bias, groups=i)


class InvertedRessidual(nn.Module):
    def __init__(self, inchannel, outchannel, stride):
        super(InvertedRessidual, self).__init__()
        branch_feature = outchannel // 2
        self.stride = stride
        if self.stride > 1:
            self.branch1 = nn.Sequential(
                depthwise_conv(inchannel,inchannel,kernel_size=3,stride=stride,padding=1),
                nn.BatchNorm2d(inchannel),
                nn.Conv2d(inchannel,branch_feature,kernel_size=1,stride=1,padding=0,bias=False),
                nn.BatchNorm2d(outchannel),
                nn.ReLU(inplace=True)
            )
        else:
            self.branch1=nn.Sequential()

        self.branch2=nn.Sequential(
            
        )
