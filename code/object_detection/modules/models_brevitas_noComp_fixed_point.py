import collections
import torch
import torch.nn as nn 
import brevitas.nn as qnn

from brevitas.quant import Int8WeightPerChannelFixedPointMSE
from brevitas.quant import Int8BiasPerTensorFixedPointInternalScaling
from brevitas.quant import Uint8ActPerTensorFixedPoint
from brevitas.inject.enum import ScalingImplType, RestrictValueType

import config

class MyQuantIdFixedPoint(Uint8ActPerTensorFixedPoint):
    bit_width = 8
    scaling_impl_type = ScalingImplType.CONST
    restrict_value_impl = RestrictValueType.POWER_OF_TWO
    scaling_init = 1 

class MyIntBias(Int8BiasPerTensorFixedPointInternalScaling):
    """
    8-bit signed int bias quantizer with scale factor equal to the scale factor of the accumulator
    the bias is added to, so typically quant_input_scale * quant_weight_scale.
    With Fixed Point and Internal Scale, return_quant_tensor=False, as you do not need previous layer scale.
    """
    bit_width = config.BIAS_BIT_WIDTH
    #pass

class MyInt8Weight(Int8WeightPerChannelFixedPointMSE):
    pass

class MyUInt8Act(Uint8ActPerTensorFixedPoint):
    pass


# ______________________________________________________________ #
# __________________ SIMPLE_BED_DETECTOR MODEL _________________ #
# __________________  NO SIGMOID, SOFTMAX  _____________________ #
# ______________________________________________________________ #
class FIXED_POINT_QUANT_SIMPLE_BED_DETECTOR(nn.Module):
    
    def __init__(self, 
                 weight_bw,
                 big_layers_weight_bw,
                 head_weight_bw,
                 act_bw, 
                 bias_bw, 
                 in_channels=3):
        
        super(FIXED_POINT_QUANT_SIMPLE_BED_DETECTOR, self).__init__()
        self.in_channels = in_channels
        self.B = config.B
        self.C = config.C

        self.weight_bw = weight_bw
        self.big_layers_weight_bw = big_layers_weight_bw
        self.head_weight_bw = head_weight_bw
        self.act_bw = act_bw
        self.bias_bw = bias_bw
        
        self.model = self.__create_BED__()
        
    def __create_BED__(self):
        BED_model = nn.Sequential(

            collections.OrderedDict(
                [
                    ("input0", qnn.QuantIdentity(
                                act_quant=MyQuantIdFixedPoint,
                                return_quant_tensor=False)),
           
                # Conv2d [in_channels, out_channels, kernel_size, stride, padding, bias]

                    # CNNBlock 224x224
                    ("conv1", qnn.QuantConv2d(
                                self.in_channels, 32, 
                                kernel_size=3, stride=1, padding=1, 
                                weight_bit_width=self.weight_bw,
                                weight_quant=MyInt8Weight,
                                bias=True, 
                                bias_quant=MyIntBias)),
                    ("relu1", qnn.QuantReLU(
                                bit_width=self.act_bw, 
                                act_quant=MyUInt8Act,
                                return_quant_tensor=False)),  
                    ("dropout1", nn.Dropout2d(p=0.3)),
        
                    # CNNBlock 112x112
                    ("maxpool2", qnn.QuantMaxPool2d(
                                kernel_size=2, stride=2,
                                return_quant_tensor=False)),
                    ("conv2", qnn.QuantConv2d(
                                32, 16, 
                                kernel_size=3, stride=1, padding=1, 
                                weight_bit_width=self.weight_bw,
                                weight_quant=MyInt8Weight,
                                bias=True, 
                                bias_quant=MyIntBias)),
                    ("relu2", qnn.QuantReLU(
                                bit_width=self.act_bw, 
                                act_quant=MyUInt8Act,
                                return_quant_tensor=False)),
                    ("dropout2",nn.Dropout2d(p=0.3)),
        
                    # CNNBlock 56x56
                    ("maxpool3", qnn.QuantMaxPool2d(
                                kernel_size=2, stride=2,
                                return_quant_tensor=False)), # Next Layer has No Bias
                    ("conv31", qnn.QuantConv2d(
                                16, 16, 
                                kernel_size=1, stride=1, 
                                weight_bit_width=self.weight_bw,
                                weight_quant=MyInt8Weight,
                                bias=True, 
                                bias_quant=MyIntBias)),
                    ("relu31", qnn.QuantReLU(
                                bit_width=self.act_bw, 
                                act_quant=MyUInt8Act,
                                return_quant_tensor=False)),
                    
                    ("conv32", qnn.QuantConv2d(
                                16, 32, 
                                kernel_size=3, stride=1, padding=1, 
                                weight_bit_width=self.weight_bw,
                                weight_quant=MyInt8Weight,
                                bias=True, 
                                bias_quant=MyIntBias)),
                    ("relu32", qnn.QuantReLU(
                                bit_width=self.act_bw, 
                                act_quant=MyUInt8Act,
                                return_quant_tensor=True)),

                    ("conv33", qnn.QuantConv2d(32, 32, 
                                kernel_size=1, stride=1, 
                                weight_bit_width=self.weight_bw,
                                weight_quant=MyInt8Weight,
                                bias=True, 
                                bias_quant=MyIntBias)),
                    ("relu33", qnn.QuantReLU(
                                bit_width=self.act_bw, 
                                act_quant=MyUInt8Act,
                                return_quant_tensor=False)),

                    ("conv34", qnn.QuantConv2d(
                                32, 64, 
                                kernel_size=3, stride=1, padding=1, 
                                weight_bit_width=self.big_layers_weight_bw,
                                weight_quant=MyInt8Weight,
                                bias=True, 
                                bias_quant=MyIntBias)),
                    ("relu34", qnn.QuantReLU(
                        bit_width=self.act_bw, 
                        act_quant=MyUInt8Act,
                        return_quant_tensor=False)),
        
                    # CNNBlock 28x28
                    ("maxpool4", qnn.QuantMaxPool2d(
                                kernel_size=2, stride=2,
                                return_quant_tensor=False)),
                    ("conv41", qnn.QuantConv2d(64, 32, 
                                kernel_size=1, stride=1, 
                                weight_bit_width=self.weight_bw,
                                weight_quant=MyInt8Weight,
                                bias=True, 
                                bias_quant=MyIntBias)),
                    ("relu41", qnn.QuantReLU(
                                bit_width=self.act_bw, 
                                act_quant=MyUInt8Act,
                                return_quant_tensor=False)),
                    
                    ("conv42", qnn.QuantConv2d(
                                32, 64, 
                                kernel_size=3, stride=1, padding=1, 
                                weight_bit_width=self.big_layers_weight_bw,
                                weight_quant=MyInt8Weight,
                                bias=True, 
                                bias_quant=MyIntBias)),
                    ("relu42", qnn.QuantReLU(
                                bit_width=self.act_bw, 
                                act_quant=MyUInt8Act,
                                return_quant_tensor=False)),

                    ("conv43", qnn.QuantConv2d(64, 32, 
                                kernel_size=1, stride=1, 
                                weight_bit_width=self.weight_bw,
                                weight_quant=MyInt8Weight,
                                bias=True, 
                                bias_quant=MyIntBias)),
                    ("relu43", qnn.QuantReLU(
                                bit_width=self.act_bw, 
                                act_quant=MyUInt8Act,
                                return_quant_tensor=False)),
                    
                    ("conv44", qnn.QuantConv2d(
                                32, 64, 
                                kernel_size=3, stride=1, padding=1, 
                                weight_bit_width=self.big_layers_weight_bw,
                                weight_quant=MyInt8Weight,
                                bias=True, 
                                bias_quant=MyIntBias)),
                    ("relu44", qnn.QuantReLU(
                                bit_width=self.act_bw, 
                                act_quant=MyUInt8Act,
                                return_quant_tensor=False)),

                    ("conv45", qnn.QuantConv2d(
                                64, 32, 
                                kernel_size=1, stride=1, 
                                weight_bit_width=self.weight_bw,
                                weight_quant=MyInt8Weight,
                                bias=True, 
                                bias_quant=MyIntBias)),
                    ("relu45", qnn.QuantReLU(
                                bit_width=self.act_bw, 
                                act_quant=MyUInt8Act,
                                return_quant_tensor=False)),
                    
                    ("conv46", qnn.QuantConv2d(
                                32, 64, 
                                kernel_size=3, stride=1, padding=1, 
                                weight_bit_width=self.big_layers_weight_bw,
                                weight_quant=MyInt8Weight,
                                bias=True, 
                                bias_quant=MyIntBias)),
                    ("relu46", qnn.QuantReLU(
                                bit_width=self.act_bw, 
                                act_quant=MyUInt8Act,
                                return_quant_tensor=False)),
                                        
                    # CNNBlock 14x14
                    ("maxpool5", qnn.QuantMaxPool2d(
                                kernel_size=2, stride=2,
                                return_quant_tensor=False)),
                    ("conv51", qnn.QuantConv2d(
                                64, 32, 
                                kernel_size=1, stride=1, 
                                weight_bit_width=self.weight_bw,
                                weight_quant=MyInt8Weight,
                                bias=True, 
                                bias_quant=MyIntBias)),
                    ("relu51", qnn.QuantReLU(
                                bit_width=self.act_bw, 
                                act_quant=MyUInt8Act,
                                return_quant_tensor=False)),
                    
                    ("conv52", qnn.QuantConv2d(
                                32, 64, 
                                kernel_size=3, stride=1, padding=1, 
                                weight_bit_width=self.big_layers_weight_bw,
                                weight_quant=MyInt8Weight,
                                bias=True, 
                                bias_quant=MyIntBias)),
                    ("relu52", qnn.QuantReLU(
                                bit_width=self.act_bw, 
                                act_quant=MyUInt8Act,
                                return_quant_tensor=False)),
                    
                    ("conv53", qnn.QuantConv2d(64, 32, 
                                kernel_size=1, stride=1, 
                                weight_bit_width=self.weight_bw,
                                weight_quant=MyInt8Weight,
                                bias=True, 
                                bias_quant=MyIntBias)),
                    ("relu53", qnn.QuantReLU(
                                bit_width=self.act_bw, 
                                act_quant=MyUInt8Act,
                                return_quant_tensor=False)),
                    
                    ("conv54", qnn.QuantConv2d(
                                32, 64, 
                                kernel_size=3, stride=1, padding=1, 
                                weight_bit_width=self.big_layers_weight_bw,
                                weight_quant=MyInt8Weight,
                                bias=True, 
                                bias_quant=MyIntBias)),
                    ("relu54", qnn.QuantReLU(
                                bit_width=self.act_bw, 
                                act_quant=MyUInt8Act,
                                return_quant_tensor=False)),
                    
                    ("conv55", qnn.QuantConv2d(
                                64, 64, 
                                kernel_size=3, stride=1, padding=1, 
                                weight_bit_width=self.big_layers_weight_bw,
                                weight_quant=MyInt8Weight,
                                bias=True, 
                                bias_quant=MyIntBias)),
                    ("relu55", qnn.QuantReLU(
                                bit_width=self.act_bw, 
                                act_quant=MyUInt8Act,
                                return_quant_tensor=False)),

                    ("conv56", qnn.QuantConv2d(
                                64, 64, 
                                kernel_size=3, stride=1, padding=1, 
                                weight_bit_width=self.big_layers_weight_bw,
                                weight_quant=MyInt8Weight,
                                bias=True, 
                                bias_quant=MyIntBias)),
                    ("relu56", qnn.QuantReLU(
                                bit_width=self.act_bw, 
                                act_quant=MyUInt8Act,
                                return_quant_tensor=False)),
            
                    # CNNBlock 7x7
                    ("maxpool6", qnn.QuantMaxPool2d(
                                kernel_size=2, stride=2,
                                return_quant_tensor=False)),
                    ("conv61", qnn.QuantConv2d(
                                64, 64, 
                                kernel_size=3, stride=1, padding=1, 
                                weight_bit_width=self.big_layers_weight_bw,
                                weight_quant=MyInt8Weight,
                                bias=True,
                                bias_quant=MyIntBias)),
                    ("relu61", qnn.QuantReLU(
                                bit_width=self.act_bw, 
                                act_quant=MyUInt8Act,
                                return_quant_tensor=False)),

                    ("conv62", qnn.QuantConv2d(
                                64, 64, 
                                kernel_size=3, stride=1, padding=1, 
                                weight_bit_width=self.big_layers_weight_bw,
                                weight_quant=MyInt8Weight,
                                bias=True,
                                bias_quant=MyIntBias)),
                    ("relu62", qnn.QuantReLU(
                                bit_width=self.act_bw, 
                                act_quant=MyUInt8Act,
                                return_quant_tensor=False)),

                    # CNNBlock Out
                    ("conv71", qnn.QuantConv2d(
                                64, 64, 
                                kernel_size=1, stride=1, 
                                weight_bit_width=self.head_weight_bw,
                                weight_quant=MyInt8Weight,
                                bias=True,
                                bias_quant=MyIntBias)),
                    ("relu71", qnn.QuantReLU(
                                bit_width=self.act_bw, 
                                act_quant=MyUInt8Act,
                                return_quant_tensor=False)),
                    
                    ("conv72", qnn.QuantConv2d(
                                64, 16, 
                                kernel_size=1, stride=1, 
                                weight_bit_width=self.head_weight_bw,
                                weight_quant=MyInt8Weight,
                                bias=True,
                                bias_quant=MyIntBias)),
                    ("relu72", qnn.QuantReLU(
                                bit_width=self.act_bw, 
                                act_quant=MyUInt8Act,
                                return_quant_tensor=False)),
                    
                    ("conv73", qnn.QuantConv2d(
                                16, 16, 
                                kernel_size=1, stride=1, 
                                weight_bit_width=self.head_weight_bw,
                                weight_quant=MyInt8Weight,
                                bias=True,
                                bias_quant=MyIntBias)),
                    ("relu73", qnn.QuantReLU(
                                bit_width=self.act_bw, 
                                act_quant=MyUInt8Act,
                                return_quant_tensor=False)),

                    ("conv74", qnn.QuantConv2d(
                                16, self.B*5 + self.C, 
                                kernel_size=1, stride=1, 
                                weight_bit_width=self.head_weight_bw,
                                weight_quant=MyInt8Weight,
                                bias=True,
                                bias_quant=MyIntBias)),
                ]
            )
        )
        return BED_model
    
    
    def forward(self, x):
        return self.model(x) 