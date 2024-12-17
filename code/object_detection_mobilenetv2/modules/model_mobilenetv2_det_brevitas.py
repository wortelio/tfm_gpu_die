import torch
import torch.nn as nn 
import math

from brevitas.quant_tensor import QuantTensor
from brevitas.core.restrict_val import RestrictValueType

from brevitas.nn import QuantConv2d
from brevitas.nn import QuantLinear
from brevitas.nn import QuantReLU
from brevitas.nn import QuantIdentity
from brevitas.nn import TruncAvgPool2d
from brevitas.quant import Int32Bias

from .common import CommonIntWeightPerChannelQuant
from .common import CommonIntWeightPerTensorQuant
from .common import CommonUintActQuant
from .common import CommonIntActQuant


#####################################
#             ReLU6                 #
#####################################
from typing import Optional
import torch.nn.functional as F
from brevitas.nn.quant_layer import QuantNonLinearActLayer as QuantNLAL
from brevitas.nn.quant_layer import ActQuantType
from brevitas.inject.defaults import Int8ActPerTensorFloatMinMaxInit
class myact_relu6(nn.Hardtanh):
    def __init__(self):
        super().__init__(0.0, 6.0)

    def forward(self,x):
        return F.hardtanh(x, 0.0, 6.0)
class QuantRelu6(QuantNLAL):
    def __init__(
            self,
            act_quant: Optional[ActQuantType] = Int8ActPerTensorFloatMinMaxInit,
            input_quant: Optional[ActQuantType] = None,
            return_quant_tensor: bool = False,
            **kwargs):
        QuantNLAL.__init__(
            self,
            act_impl=myact_relu6,
            passthrough_act=True,
            input_quant=input_quant,
            act_quant=act_quant,
            return_quant_tensor=return_quant_tensor,
            **kwargs)



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


def conv_3x3_bn(inp, oup, stride, weight_bit_width, act_bit_width):
    # return nn.Sequential(
    #     nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
    #     nn.BatchNorm2d(oup),
    #     nn.ReLU()
    # )
    return nn.Sequential(
        QuantConv2d(
            in_channels=inp,
            out_channels=oup,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
            weight_quant=CommonIntWeightPerChannelQuant,
            weight_bit_width=weight_bit_width),
        nn.BatchNorm2d(oup),
        # QuantReLU(
        #     act_quant=CommonUintActQuant,
        #     bit_width=act_bit_width,
        #     return_quant_tensor=True),
        QuantRelu6(
            min_val=0.0, 
            max_val=6.0, 
            bit_width=act_bit_width,
            signed=False,
            return_quant_tensor=True
        ),
    )
            
        


def conv_1x1_bn(inp, oup, weight_bit_width, act_bit_width):
    # return nn.Sequential(
    #     nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
    #     nn.BatchNorm2d(oup),
    #     nn.ReLU()
    # )
    return nn.Sequential(
        QuantConv2d(
            in_channels=inp,
            out_channels=oup,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False,
            weight_quant=CommonIntWeightPerChannelQuant,
            weight_bit_width=weight_bit_width),
        nn.BatchNorm2d(oup),
        # QuantReLU(
        #     act_quant=CommonUintActQuant,
        #     bit_width=act_bit_width,
        #     return_quant_tensor=True),
        QuantRelu6(
            min_val=0.0, 
            max_val=6.0, 
            bit_width=act_bit_width,
            signed=False,
            return_quant_tensor=True
        ),
    )


class InvertedBlock(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio, weight_bit_width, act_bit_width, shared_quant):
        super(InvertedBlock, self).__init__()
        assert stride in [1, 2]

        hidden_dim = round(inp * expand_ratio)
        self.identity = stride == 1 and inp == oup
        if self.identity:
            self.pw_linear_quant = shared_quant.act_quant
            # Add quantizer only needed if block is a Resnet one
            self.quant_identity_out = QuantIdentity(
                act_quant=CommonIntActQuant,
                bit_width=act_bit_width,
                return_quant_tensor=True)
        else:
            self.pw_linear_quant = CommonIntActQuant    

        if expand_ratio == 1:
            # self.conv = nn.Sequential(
            #     # dw
            #     nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
            #     nn.BatchNorm2d(hidden_dim),
            #     nn.ReLU(),
            #     # pw-linear
            #     nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
            #     nn.BatchNorm2d(oup),
            # )
            self.conv = nn.Sequential(
                # dw
                QuantConv2d(
                    in_channels=hidden_dim,
                    out_channels=hidden_dim,
                    kernel_size=3,
                    stride=stride,
                    padding=1,
                    groups=hidden_dim,
                    bias=False,
                    weight_quant=CommonIntWeightPerChannelQuant,
                    weight_bit_width=weight_bit_width),
                nn.BatchNorm2d(hidden_dim),
                # QuantReLU(
                #     act_quant=CommonUintActQuant,
                #     bit_width=act_bit_width),
                QuantRelu6(
                    min_val=0.0, 
                    max_val=6.0, 
                    bit_width=act_bit_width,
                    signed=False),
                # pw-linear
                QuantConv2d(
                    in_channels=hidden_dim,
                    out_channels=oup,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                    bias=False,
                    weight_quant=CommonIntWeightPerChannelQuant,
                    weight_bit_width=weight_bit_width),
                nn.BatchNorm2d(oup),
                QuantIdentity(
                    act_quant=self.pw_linear_quant,
                    bit_width=act_bit_width,
                    return_quant_tensor=True),
                )
        else:
            # self.conv = nn.Sequential(
            #     # pw
            #     nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
            #     nn.BatchNorm2d(hidden_dim),
            #     nn.ReLU(),
            #     # dw
            #     nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
            #     nn.BatchNorm2d(hidden_dim),
            #     nn.ReLU(),
            #     # pw-linear
            #     nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
            #     nn.BatchNorm2d(oup),
            # )
            self.conv = nn.Sequential(
                # pw
                QuantConv2d(
                    in_channels=inp,
                    out_channels=hidden_dim,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                    bias=False,
                    weight_quant=CommonIntWeightPerChannelQuant,
                    weight_bit_width=weight_bit_width),
                nn.BatchNorm2d(hidden_dim),
                # QuantReLU(
                #     act_quant=CommonUintActQuant,
                #     bit_width=act_bit_width),
                QuantRelu6(
                    min_val=0.0, 
                    max_val=6.0, 
                    bit_width=act_bit_width,
                    signed=False),
                # dw
                QuantConv2d(
                    in_channels=hidden_dim,
                    out_channels=hidden_dim,
                    kernel_size=3,
                    stride=stride,
                    padding=1,
                    groups=hidden_dim,
                    bias=False,
                    weight_quant=CommonIntWeightPerChannelQuant,
                    weight_bit_width=weight_bit_width),
                nn.BatchNorm2d(hidden_dim),
                # QuantReLU(
                #     act_quant=CommonUintActQuant,
                #     bit_width=act_bit_width),
                QuantRelu6(
                    min_val=0.0, 
                    max_val=6.0, 
                    bit_width=act_bit_width,
                    signed=False),
                # pw-linear
                QuantConv2d(
                    in_channels=hidden_dim,
                    out_channels=oup,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                    bias=False,
                    weight_quant=CommonIntWeightPerChannelQuant,
                    weight_bit_width=weight_bit_width),
                nn.BatchNorm2d(oup),
                QuantIdentity(
                    act_quant=self.pw_linear_quant,
                    bit_width=act_bit_width,
                    return_quant_tensor=True),
                )

        
    def forward(self, x):
        if self.identity:
            conv_out = self.conv(x)
            assert isinstance(conv_out, QuantTensor), "Perform add among QuantTensors"
            assert isinstance(x, QuantTensor), "Perform add among QuantTensors"
            x = x + conv_out
            x = self.quant_identity_out(x)
            return x 
        else:
            return self.conv(x)

    # Return the Quantizer of the Output, to use it in next block if Resnet Connection is applied
        # If current block is a Resnet one, return Add Quantizer
        # If current block is not Resnet, return pw-linear quantizer
    def get_shared_quant(self):
        if self.identity:
            return self.quant_identity_out
        else:
            return self.conv[-1] 



class MobileNetV2_DET_BREVITAS(nn.Module):
    def __init__(self, num_classes=2, 
                 width_mult=1., 
                 in_bit_width=8,
                 weight_bit_width=4,
                 act_bit_width=4):
        super(MobileNetV2_DET_BREVITAS, self).__init__()
        # setting of inverted residual blocks
        ### 180K params
        # self.cfgs = [
        #     # t, c, n, s
        #     [1,  16, 1, 1],
        #     [2,  24, 2, 2],
        #     [2,  32, 3, 2],
        #     [2,  64, 3, 2],
        #     [2,  96, 2, 1],
        #     [2, 128, 1, 2],
        #     # [6, 160, 2, 2],
        #     # [6, 320, 1, 1],
        # ]
        ### 307K params
        self.cfgs = [
            # t, c, n, s
            [1,  16, 1, 1],
            [2,  24, 2, 2],
            [2,  32, 3, 2],
            [2,  64, 4, 2],
            [2,  96, 3, 1],
            [2, 128, 2, 2],
            # [2, 320, 1, 1],
        ]
        
        # Input 224x224x3
        layers = [
                QuantIdentity( # for Q1.7 input format -> sign.7bits
                act_quant = CommonIntActQuant,
                bit_width = in_bit_width,
                min_val = -1.0,
                max_val = 1.0 - 2.0 ** (-7),
                narrow_range = False, 
                restrict_scaling_type = RestrictValueType.POWER_OF_TWO)
        ]
        
        # building first layer
        input_channel = _make_divisible(32 * width_mult, 4 if width_mult == 0.1 else 8)
        layers.append(conv_3x3_bn(3, input_channel, 2, weight_bit_width, act_bit_width))
        
        # building inverted residual blocks
        block = InvertedBlock
        shared_quant = None
        for t, c, n, s in self.cfgs:
            output_channel = _make_divisible(c * width_mult, 4 if width_mult == 0.1 else 8)
            for i in range(n):
                new_block = block(input_channel, output_channel, s if i == 0 else 1, t, weight_bit_width, act_bit_width, shared_quant)
                layers.append(new_block)
                shared_quant = new_block.get_shared_quant()
                input_channel = output_channel
        self.features = nn.Sequential(*layers)
        
        out_convs = []
        out_convs.append(conv_1x1_bn(output_channel, 64, weight_bit_width, act_bit_width))
        out_convs.append(conv_1x1_bn(64, 16, weight_bit_width, act_bit_width))
        out_convs.append(conv_1x1_bn(16, 16, weight_bit_width, act_bit_width))
        out_convs.append(QuantConv2d(
            in_channels=16,
            out_channels=12,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False,
            weight_quant=CommonIntWeightPerChannelQuant,
            weight_bit_width=8))
        self.detector = nn.Sequential(*out_convs)
        
    def forward(self, x):
        x = 2.0 * x - torch.tensor([1.0], device=x.device)
        x = self.features(x)
        x = self.detector(x)
        return x



                



                
