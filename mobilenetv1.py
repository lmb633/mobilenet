import torch
import torch.nn as nn


def conv_bn(inchannel, outchannel, stride, leak=0.1):
    return nn.Sequential(
        nn.Conv2d(inchannel, outchannel, 3, stride, 1, bias=False),
        nn.BatchNorm2d(outchannel),
        nn.LeakyReLU(leak, inplace=True)
    )


def conv_dw(inchannel, outchannel, stride, leak=0.1):
    return nn.Sequential(
        nn.Conv2d(inchannel, inchannel, 3, stride, 1, groups=inchannel, bias=False),
        nn.BatchNorm2d(inchannel),
        nn.LeakyReLU(leak, True),

        nn.Conv2d(inchannel, outchannel, 1, 1, 0, bias=False),
        nn.BatchNorm2d(outchannel),
        nn.LeakyReLU(leak, True)

    )


class MobileNetV1(nn.Module):
    def __init__(self, expand=1):
        super(MobileNetV1, self).__init__()
        self.expand = expand
        self.conv_bn = conv_bn(3, 8 * expand, 2)
        self.dw = nn.Sequential()
        dw_op = [
            [8 * expand, 16 * expand, 1],
            [16 * expand, 32 * expand, 2],
            [32 * expand, 32 * expand, 1],
            [32 * expand, 64 * expand, 2],
            [64 * expand, 64 * expand, 1],

            [64 * expand, 128 * expand, 2],
            [128 * expand, 128 * expand, 1],
            [128 * expand, 128 * expand, 1],
            [128 * expand, 128 * expand, 1],
            [128 * expand, 128 * expand, 1],
            [128 * expand, 128 * expand, 1],

            [128 * expand, 256 * expand, 2],
            [256 * expand, 256 * expand, 1]
        ]
        for i, o, s in dw_op:
            self.dw.add_module('conv_bn{0}_{1}_{2}'.format(i, o, s), conv_bn(i, o, s))
        self.avg = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(256 * expand, 1000)

    def forward(self, x):
        x = self.conv_bn(x)
        x = self.dw(x)
        x = self.avg(x)
        x = x.view(-1, 256 * self.expand)
        x = self.fc(x)
        return x


if __name__ == '__main__':
    mobilenet = MobileNetV1(2)
    x = torch.randn(1, 3, 128, 128)
    out = mobilenet(x)
    print(out.shape)
