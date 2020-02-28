import torch
import torch.nn as nn


class ConvDn(nn.Module):
    def __init__(self, inchannel, outchannel, stride, expand=6):
        super(ConvDn, self).__init__()
        self.use_res = stride == 1 and inchannel == outchannel
        hidden = inchannel * expand
        self.conv = nn.Sequential(
            nn.Conv2d(inchannel, hidden, 1, 1, 0, bias=False),
            nn.BatchNorm2d(hidden),
            nn.ReLU6(True),

            nn.Conv2d(hidden, hidden, 3, stride, 1, groups=hidden, bias=False),
            nn.BatchNorm2d(hidden),
            nn.LeakyReLU(True),

            nn.Conv2d(hidden, outchannel, 1, 1, 0, bias=False),
            nn.BatchNorm2d(outchannel)
        )

    def forward(self, x):
        if self.use_res:
            return x + self.conv(x)
        return self.conv(x)


def conv_bn(inchannel, outchannel, stride):
    return nn.Sequential(
        nn.Conv2d(inchannel, outchannel, 3, stride, 1, bias=False),
        nn.BatchNorm2d(outchannel),
        nn.ReLU6(inplace=True)
    )


def conv_1x1_bn(inchannel, outchannel):
    return nn.Sequential(
        nn.Conv2d(inchannel, outchannel, 1, 1, 0, bias=False),
        nn.BatchNorm2d(outchannel),
        nn.ReLU6(inplace=True)
    )


class MobileNetV2(nn.Module):
    def __init__(self, width_mult=1):
        super().__init__()
        inchannel = 32
        self.lastchannel = 1280
        self.conv = nn.Sequential()
        config = [
            [1, 16, 1, 1],
            [6, 24, 2, 2],
            [6, 32, 3, 2],
            [6, 64, 4, 2],
            [6, 96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1],
        ]
        self.conv.add_module('first conv', conv_bn(3, inchannel, 2))
        for t, c, n, s in config:
            outchannel = c * width_mult
            for i in range(n):
                if i == 0:
                    self.conv.add_module('conv_{0}_{1}_{2}'.format(inchannel, outchannel, i), ConvDn(inchannel, outchannel, s, expand=t))
                else:
                    self.conv.add_module('conv_{0}_{1}_{2}'.format(inchannel, outchannel, i), ConvDn(inchannel, outchannel, 1, expand=t))
                inchannel = outchannel
        self.conv.add_module('last conv', conv_1x1_bn(inchannel, self.lastchannel))
        self.avg = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(self.lastchannel, 1000)

    def forward(self, x):
        x = self.conv(x)
        x = self.avg(x)
        x = x.view(-1, self.lastchannel)
        x = self.fc(x)
        return x


model = MobileNetV2()
x = torch.rand(1, 3, 224, 224)
out = model(x)
print(out.shape)
