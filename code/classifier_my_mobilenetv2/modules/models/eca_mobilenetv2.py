from torch import nn
from .eca_module import eca_layer
import math
import config


def _make_divisible(v, divisor, min_value=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    :param v:
    :param divisor:
    :param min_value:
    :return:
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v



class ConvBNReLU(nn.Sequential):
    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, groups=1):
        padding = (kernel_size - 1) // 2
        super(ConvBNReLU, self).__init__(
            nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding, groups=groups, bias=False),
            nn.BatchNorm2d(out_planes),
            nn.ReLU6(inplace=True)
        )


class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio, k_size):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        hidden_dim = int(round(inp * expand_ratio))
        self.use_res_connect = self.stride == 1 and inp == oup

        layers = []
        if expand_ratio != 1:
            # pw
            layers.append(ConvBNReLU(inp, hidden_dim, kernel_size=1))
        layers.extend([
            # dw
            ConvBNReLU(hidden_dim, hidden_dim, stride=stride, groups=hidden_dim),
            # pw-linear
            nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
            nn.BatchNorm2d(oup),
        ])
        layers.append(eca_layer(oup, k_size))
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class ECA_MobileNetV2(nn.Module):
    def __init__(self, num_classes=config.NUM_CLASSES, width_mult=1.0):
        super(ECA_MobileNetV2, self).__init__()
        block = InvertedResidual
        input_channel = 32
        # last_channel = 1280
        last_channel = _make_divisible(128 * width_mult, 4 if width_mult == 0.1 else 8) if width_mult > 1.0 else 128
        # inverted_residual_setting = [
        #     # t, c, n, s
        #     [1, 16, 1, 1],
        #     [6, 24, 2, 2],
        #     [6, 32, 3, 2],
        #     [6, 64, 4, 2],
        #     [6, 96, 3, 1],
        #     [6, 160, 3, 2],
        #     [6, 320, 1, 1],
        # ]
        inverted_residual_setting = [
            # t, c, n, s
            [1,  8, 1, 1],
            [2,  16, 2, 2],
            [2,  24, 2, 2],
            [4,  32, 3, 2],
            [2,  64, 2, 1],
            # [6, 160, 2, 2],
            # [6, 320, 1, 1],
        ]

        # building first layer
        input_channel = int(input_channel * width_mult)
        # input_channel = _make_divisible(32 * width_mult, 4 if width_mult == 0.1 else 8)
        self.last_channel = int(last_channel * max(1.0, width_mult))
        features = [ConvBNReLU(3, input_channel, stride=2)]
        # building inverted residual blocks
        for t, c, n, s in inverted_residual_setting:
            output_channel = int(c * width_mult)
            for i in range(n):
                # if c < 96:
                #     ksize = 1
                # else:
                #     ksize = 3
                if c < 24:
                    ksize = 3
                else:
                    ksize = 5
                stride = s if i == 0 else 1
                features.append(block(input_channel, output_channel, stride, expand_ratio=t, k_size=ksize))
                input_channel = output_channel
        # building last several layers
        features.append(ConvBNReLU(input_channel, self.last_channel, kernel_size=1))
        # make it nn.Sequential
        self.features = nn.Sequential(*features)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # building classifier
        self.classifier = nn.Sequential(
            nn.Dropout(0.25),
            nn.Linear(self.last_channel, num_classes),
        )

        # weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


