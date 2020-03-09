import torch
import torch.nn as nn


class Fire(nn.Module):
    def __init__(self, in_channel, squeeze_channel, expand1x1, expand3x3):
        super().__init__()
        self.squeeze = nn.Conv2d(in_channel, squeeze_channel, 1)
        self.squeeze_act = nn.ReLU(inplace=True)
        self.expand1x1 = nn.Conv2d(squeeze_channel, expand1x1, 1)
        self.expand1x1_act = nn.ReLU(inplace=True)
        self.expand3x3 = nn.Conv2d(squeeze_channel, expand3x3, 1, padding=1)
        self.expand3x3_act = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.squeeze_act(self.squeeze(x))
        return torch.cat([
            self.expand1x1_act(self.expand1x1(x)),
            self.expand3x3_act(self.expand1x1(x))
        ], 1)


class SqueezeNet(nn.Module):
    def __init__(self, num_class=1000):
        super().__init__()
        self.feature = nn.Sequential(
            nn.Conv2d(3, 96, kernel_size=7, stride=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
            Fire(96, 16, 64, 64),
            Fire(128, 16, 64, 64),
            Fire(128, 32, 128, 128),
            nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
            Fire(256, 32, 128, 128),
            Fire(256, 48, 192, 192),
            Fire(384, 48, 192, 192),
            Fire(384, 64, 256, 256),
            nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
            Fire(512, 64, 256, 256)
        )
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Conv2d(512, num_class, 1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        for m in self.modules():
            # print("===========================")
            # print(m)
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, mean=0.0, std=0.01)
                # else:
                #     nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.feature(x)
        x = self.classifier(x)
        return x.reshape(-1)


if __name__ == '__main__':
    squeezenet = SqueezeNet()
    x = torch.randn(1, 3, 128, 128)
    out = squeezenet(x)
    print(out.shape)
