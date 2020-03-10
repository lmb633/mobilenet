import torch
import torch.nn as nn
import torch.nn.functional as F


def channel_shuffle(x, groups):
    batch_size, channels, h, w = x.data.size()
    channels_per_group = channels // groups
    x = x.view(batch_size, groups, channels_per_group, h, w)
    x = torch.transpose(x, 1, 2).contiguous()
    x = x.view(batch_size, -1, h, w)
    return x


class ShuffleUnit(nn.Module):
    def __init__(self, inchannels, outchannels, kernel, stride, bottleneck, first_group, group):
        super().__init__()
        self.group = group
        self.stride = stride
        pad = kernel // 2
        if stride == 2:
            outchannels = outchannels - inchannels
        else:
            outchannels = outchannels
        self.conv1 = nn.Sequential(
            nn.Conv2d(inchannels, bottleneck, 1, 1, 0, groups=1 if first_group else group, bias=False),
            nn.BatchNorm2d(bottleneck),
            nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(bottleneck, bottleneck, kernel, stride, pad, groups=bottleneck, bias=False),
            nn.BatchNorm2d(bottleneck),
            nn.Conv2d(bottleneck, outchannels, 1, 1, 0, groups=group),
            nn.BatchNorm2d(outchannels)
        )
        if stride == 2:
            self.branch_proj = nn.AvgPool2d(3, 2, 1)

    def forward(self, x):
        residual = x
        x = self.conv1(x)
        if self.group > 1:
            x = channel_shuffle(x, self.group)
        x = self.conv2(x)
        if self.stride == 1:
            return F.relu(residual + x)
        elif self.stride == 2:
            return torch.cat((self.branch_proj(residual), F.relu(x)), 1)


class ShuffleNetV1(nn.Module):
    def __init__(self, n_class=1000, model_size='2.0x', group=3):
        super().__init__()
        print('model size is ', model_size)
        stage_repeat = [4, 8, 4]
        if group == 3:
            if model_size == '0.5x':
                stage_out_channels = [-1, 12, 120, 240, 480]
            elif model_size == '1.0x':
                stage_out_channels = [-1, 24, 240, 480, 960]
            elif model_size == '1.5x':
                stage_out_channels = [-1, 24, 360, 720, 1440]
            elif model_size == '2.0x':
                stage_out_channels = [-1, 48, 480, 960, 1920]
            else:
                raise NotImplementedError
        elif group == 8:
            if model_size == '0.5x':
                stage_out_channels = [-1, 16, 192, 384, 768]
            elif model_size == '1.0x':
                stage_out_channels = [-1, 24, 384, 768, 1536]
            elif model_size == '1.5x':
                stage_out_channels = [-1, 24, 576, 1152, 2304]
            elif model_size == '2.0x':
                stage_out_channels = [-1, 48, 768, 1536, 3072]
            else:
                raise NotImplementedError
        self.last_channel = stage_out_channels[-1]
        inchannel = stage_out_channels[1]
        self.first_conv = nn.Sequential(
            nn.Conv2d(3, inchannel, 3, 2, 1, bias=False),
            nn.BatchNorm2d(inchannel),
            nn.ReLU(inplace=True)
        )
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.feature = nn.Sequential()
        for idx in range(len(stage_repeat)):
            repeat = stage_repeat[idx]
            outchannels = stage_out_channels[idx + 2]
            for i in range(repeat):
                stride = 2 if i == 0 else 1
                first_group = idx == 0 and i == 0
                self.feature.add_module('conv_{0}_{1}'.format(idx, i), ShuffleUnit(inchannel, outchannels, 3, stride, outchannels // 4, first_group, group))
                inchannel = outchannels
        self.globalpool = nn.AdaptiveAvgPool2d((1, 1))
        self.leaner = nn.Linear(stage_out_channels[-1], n_class, bias=False)
        self.init_weight()

    def forward(self, x):
        x = self.first_conv(x)
        x = self.maxpool(x)
        x = self.feature(x)
        x = self.globalpool(x)
        x = x.view(-1, self.last_channel)
        x = self.leaner(x)
        return x

    def init_weight(self):
        for name, m in self.named_modules():
            if isinstance(m, nn.Conv2d):
                if 'first' in name:
                    nn.init.normal_(m.weight, 0, 0.01)
                else:
                    nn.init.normal_(m.weight, 0, 1.0 / m.weight.shape[1])
                if m.bias is not None:
                    nn.init.constant_(m.weight, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0001)
                nn.init.constant_(m.running_mean, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0001)
                nn.init.constant_(m.running_mean, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)


if __name__ == '__main__':
    shuffleNetV1 = ShuffleNetV1()
    x = torch.randn(1, 3, 224, 224)
    out = shuffleNetV1(x)
    print(out.data.size())
