import torch
import torch.nn as nn


def conv_dw(inchannel, outchannel, stride, leak=0.1):
    return nn.Sequential(
        nn.Conv2d(inchannel, inchannel, 3, stride, 1, groups=inchannel, bias=False),
        nn.BatchNorm2d(inchannel),
        nn.LeakyReLU(leak, True),

        nn.Conv2d(inchannel, outchannel, 1, 1, 0, bias=False),
        nn.BatchNorm2d(outchannel),
        nn.LeakyReLU(leak, True)

    )


def conv_bn(inchannel, outchannel, stride, leak=0.1):
    return nn.Sequential(
        nn.Conv2d(inchannel, outchannel, 3, stride, 1, bias=False),
        nn.BatchNorm2d(outchannel),
        nn.LeakyReLU(leak, inplace=True)
    )
