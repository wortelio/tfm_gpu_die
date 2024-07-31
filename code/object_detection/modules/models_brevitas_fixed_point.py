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
# ___________ PRUNED_AFTER_SVD_BED_DETECTOR MODEL ______________ #
# __________________  NO SIGMOID, SOFTMAX  _____________________ #
# ______________________________________________________________ #
class FIXED_POINT_QUANT_PRUNED_AFTER_SVD_BED_DETECTOR(nn.Module):
    
    def __init__(self, 
                 weight_bw,
                 big_layers_weight_bw,
                 head_weight_bw,
                 act_bw, 
                 bias_bw, 
                 in_channels=3):
        
        super(FIXED_POINT_QUANT_PRUNED_AFTER_SVD_BED_DETECTOR, self).__init__()
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
                    ("conv1", nn.Sequential(
                        qnn.QuantConv2d(
                                self.in_channels, 5, 
                                kernel_size=(3, 1), stride=(1, 1), padding=(1,0), 
                                weight_bit_width=self.weight_bw,
                                weight_quant=MyInt8Weight,
                                bias=False,
                                return_quant_tensor=False),
                        qnn.QuantConv2d(
                                5, 32, 
                                kernel_size=(1, 3), stride=(1, 1), padding=(0,1), 
                                weight_bit_width=self.weight_bw,
                                weight_quant=MyInt8Weight,
                                bias=True, 
                                bias_quant=MyIntBias))),
                    ("relu1", qnn.QuantReLU(
                                bit_width=self.act_bw, 
                                act_quant=MyUInt8Act,
                                return_quant_tensor=False)),  
                    ("dropout1", nn.Dropout2d(p=0.3)),
        
                    # CNNBlock 112x112
                    ("maxpool2", qnn.QuantMaxPool2d(
                                kernel_size=2, stride=2,
                                return_quant_tensor=False)),
                    ("conv2", nn.Sequential(
                        qnn.QuantConv2d(
                                32, 4, 
                                kernel_size=(3, 1), stride=(1, 1), padding=(1,0), 
                                weight_bit_width=self.weight_bw,
                                weight_quant=MyInt8Weight,
                                bias=False,
                                return_quant_tensor=False),
                        qnn.QuantConv2d(
                                4, 11, 
                                kernel_size=(1, 3), stride=(1, 1), padding=(0,1), 
                                weight_bit_width=self.weight_bw,
                                weight_quant=MyInt8Weight,
                                bias=True, 
                                bias_quant=MyIntBias))),
                    ("relu2", qnn.QuantReLU(
                                bit_width=self.act_bw, 
                                act_quant=MyUInt8Act,
                                return_quant_tensor=False)),
                    ("dropout2",nn.Dropout2d(p=0.3)),
        
                    # CNNBlock 56x56
                    ("maxpool3", qnn.QuantMaxPool2d(
                                kernel_size=2, stride=2,
                                return_quant_tensor=False)), # Next Layer has No Bias
                    ("conv31", nn.Sequential(
                        qnn.QuantConv2d(
                                11, 3, 
                                kernel_size=(1, 1), stride=(1, 1), 
                                weight_bit_width=self.weight_bw,
                                weight_quant=MyInt8Weight,
                                bias=False,
                                return_quant_tensor=False),
                        qnn.QuantConv2d(
                                3, 14, 
                                kernel_size=(1, 1), stride=(1, 1), 
                                weight_bit_width=self.weight_bw,
                                weight_quant=MyInt8Weight,
                                bias=True, 
                                bias_quant=MyIntBias))),
                    ("relu31", qnn.QuantReLU(
                                bit_width=self.act_bw, 
                                act_quant=MyUInt8Act,
                                return_quant_tensor=False)),
                    
                    ("conv32", nn.Sequential(
                        qnn.QuantConv2d(
                                14, 14, 
                                kernel_size=(3, 1), stride=(1, 1), padding=(1,0), 
                                weight_bit_width=self.weight_bw,
                                weight_quant=MyInt8Weight,
                                bias=False,
                                return_quant_tensor=False),
                        qnn.QuantConv2d(
                                14, 25, 
                                kernel_size=(1, 3), stride=(1, 1), padding=(0,1), 
                                weight_bit_width=self.weight_bw,
                                weight_quant=MyInt8Weight,
                                bias=True, 
                                bias_quant=MyIntBias))),
                    ("relu32", qnn.QuantReLU(
                                bit_width=self.act_bw, 
                                act_quant=MyUInt8Act,
                                return_quant_tensor=True)),

                    ("conv33", qnn.QuantConv2d(25, 22, 
                                kernel_size=(1, 1), stride=(1, 1), 
                                weight_bit_width=self.weight_bw,
                                weight_quant=MyInt8Weight,
                                bias=True, 
                                bias_quant=MyIntBias)),
                    ("relu33", qnn.QuantReLU(
                                bit_width=self.act_bw, 
                                act_quant=MyUInt8Act,
                                return_quant_tensor=False)),

                    ("conv34", nn.Sequential(
                        qnn.QuantConv2d(
                                22, 30, 
                                kernel_size=(3, 1), stride=(1, 1), padding=(1,0), 
                                weight_bit_width=self.weight_bw,
                                weight_quant=MyInt8Weight,
                                bias=False,
                                return_quant_tensor=False),
                        qnn.QuantConv2d(
                                30, 64, 
                                kernel_size=(1, 3), stride=(1, 1), padding=(0,1), 
                                weight_bit_width=self.weight_bw,
                                weight_quant=MyInt8Weight,
                                bias=True, 
                                bias_quant=MyIntBias))),
                    ("relu34", qnn.QuantReLU(
                        bit_width=self.act_bw, 
                        act_quant=MyUInt8Act,
                        return_quant_tensor=False)),
        
                    # CNNBlock 28x28
                    ("maxpool4", qnn.QuantMaxPool2d(
                                kernel_size=2, stride=2,
                                return_quant_tensor=False)),
                    ("conv41", qnn.QuantConv2d(64, 28, 
                                kernel_size=(1, 1), stride=(1, 1), 
                                weight_bit_width=self.weight_bw,
                                weight_quant=MyInt8Weight,
                                bias=True, 
                                bias_quant=MyIntBias)),
                    ("relu41", qnn.QuantReLU(
                                bit_width=self.act_bw, 
                                act_quant=MyUInt8Act,
                                return_quant_tensor=False)),
                    
                    ("conv42", nn.Sequential(
                        qnn.QuantConv2d(
                                28, 38, 
                                kernel_size=(3, 1), stride=(1, 1), padding=(1,0), 
                                weight_bit_width=self.weight_bw,
                                weight_quant=MyInt8Weight,
                                bias=False,
                                return_quant_tensor=False),
                        qnn.QuantConv2d(
                                38, 64, 
                                kernel_size=(1, 3), stride=(1, 1), padding=(0,1), 
                                weight_bit_width=self.weight_bw,
                                weight_quant=MyInt8Weight,
                                bias=True, 
                                bias_quant=MyIntBias))),
                    ("relu42", qnn.QuantReLU(
                                bit_width=self.act_bw, 
                                act_quant=MyUInt8Act,
                                return_quant_tensor=False)),

                    ("conv43", qnn.QuantConv2d(64, 28, 
                                kernel_size=(1, 1), stride=(1, 1), 
                                weight_bit_width=self.weight_bw,
                                weight_quant=MyInt8Weight,
                                bias=True, 
                                bias_quant=MyIntBias)),
                    ("relu43", qnn.QuantReLU(
                                bit_width=self.act_bw, 
                                act_quant=MyUInt8Act,
                                return_quant_tensor=False)),
                    
                    ("conv44", nn.Sequential(
                        qnn.QuantConv2d(
                                28, 25, 
                                kernel_size=(3, 1), stride=(1, 1), padding=(1,0), 
                                weight_bit_width=self.weight_bw,
                                weight_quant=MyInt8Weight,
                                bias=False,
                                return_quant_tensor=False),
                        qnn.QuantConv2d(
                                25, 57, 
                                kernel_size=(1, 3), stride=(1, 1), padding=(0,1), 
                                weight_bit_width=self.weight_bw,
                                weight_quant=MyInt8Weight,
                                bias=True, 
                                bias_quant=MyIntBias))),
                    ("relu44", qnn.QuantReLU(
                                bit_width=self.act_bw, 
                                act_quant=MyUInt8Act,
                                return_quant_tensor=False)),

                    ("conv45", nn.Sequential(
                        qnn.QuantConv2d(
                                57, 15, 
                                kernel_size=(1, 1), stride=(1, 1), 
                                weight_bit_width=self.weight_bw,
                                weight_quant=MyInt8Weight,
                                bias=False,
                                return_quant_tensor=True),
                        qnn.QuantConv2d(
                                15, 28, 
                                kernel_size=(1, 1), stride=(1, 1), 
                                weight_bit_width=self.weight_bw,
                                weight_quant=MyInt8Weight,
                                bias=True, 
                                bias_quant=MyIntBias))),
                    ("relu45", qnn.QuantReLU(
                                bit_width=self.act_bw, 
                                act_quant=MyUInt8Act,
                                return_quant_tensor=False)),
                    
                    ("conv46", nn.Sequential(
                        qnn.QuantConv2d(
                                28, 28, 
                                kernel_size=(3, 1), stride=(1, 1), padding=(1,0), 
                                weight_bit_width=self.weight_bw,
                                weight_quant=MyInt8Weight,
                                bias=False,
                                return_quant_tensor=False),
                        qnn.QuantConv2d(
                                28, 57, 
                                kernel_size=(1, 3), stride=(1, 1), padding=(0,1), 
                                weight_bit_width=self.weight_bw,
                                weight_quant=MyInt8Weight,
                                bias=True, 
                                bias_quant=MyIntBias))),
                    ("relu46", qnn.QuantReLU(
                                bit_width=self.act_bw, 
                                act_quant=MyUInt8Act,
                                return_quant_tensor=False)),
                                        
                    # CNNBlock 14x14
                    ("maxpool5", qnn.QuantMaxPool2d(
                                kernel_size=2, stride=2,
                                return_quant_tensor=False)),
                    ("conv51", nn.Sequential(
                        qnn.QuantConv2d(
                                57, 14, 
                                kernel_size=(1, 1), stride=(1, 1), 
                                weight_bit_width=self.weight_bw,
                                weight_quant=MyInt8Weight,
                                bias=False,
                                return_quant_tensor=False),
                        qnn.QuantConv2d(
                                14, 32, 
                                kernel_size=(1, 1), stride=(1, 1), 
                                weight_bit_width=self.weight_bw,
                                weight_quant=MyInt8Weight,
                                bias=True, 
                                bias_quant=MyIntBias))),
                    ("relu51", qnn.QuantReLU(
                                bit_width=self.act_bw, 
                                act_quant=MyUInt8Act,
                                return_quant_tensor=False)),
                    
                    ("conv52", nn.Sequential(
                        qnn.QuantConv2d(
                                32, 40, 
                                kernel_size=(3, 1), stride=(1, 1), padding=(1,0), 
                                weight_bit_width=self.weight_bw,
                                weight_quant=MyInt8Weight,
                                bias=False,
                                return_quant_tensor=False),
                        qnn.QuantConv2d(
                                40, 64, 
                                kernel_size=(1, 3), stride=(1, 1), padding=(0,1), 
                                weight_bit_width=self.weight_bw,
                                weight_quant=MyInt8Weight,
                                bias=True, 
                                bias_quant=MyIntBias))),
                    ("relu52", qnn.QuantReLU(
                                bit_width=self.act_bw, 
                                act_quant=MyUInt8Act,
                                return_quant_tensor=False)),
                    
                    ("conv53", qnn.QuantConv2d(64, 32, 
                                kernel_size=(1, 1), stride=(1, 1), 
                                weight_bit_width=self.weight_bw,
                                weight_quant=MyInt8Weight,
                                bias=True, 
                                bias_quant=MyIntBias)),
                    ("relu53", qnn.QuantReLU(
                                bit_width=self.act_bw, 
                                act_quant=MyUInt8Act,
                                return_quant_tensor=False)),
                    
                    ("conv54", nn.Sequential(
                        qnn.QuantConv2d(
                                32, 45, 
                                kernel_size=(3, 1), stride=(1, 1), padding=(1,0), 
                                weight_bit_width=self.weight_bw,
                                weight_quant=MyInt8Weight,
                                bias=False,
                                return_quant_tensor=False),
                        qnn.QuantConv2d(
                                45, 64, 
                                kernel_size=(1, 3), stride=(1, 1), padding=(0,1), 
                                weight_bit_width=self.weight_bw,
                                weight_quant=MyInt8Weight,
                                bias=True, 
                                bias_quant=MyIntBias))),
                    ("relu54", qnn.QuantReLU(
                                bit_width=self.act_bw, 
                                act_quant=MyUInt8Act,
                                return_quant_tensor=False)),
                    
                    ("conv55", nn.Sequential(
                        qnn.QuantConv2d(
                                64, 67, 
                                kernel_size=(3, 1), stride=(1, 1), padding=(1,0), 
                                weight_bit_width=self.big_layers_weight_bw,
                                weight_quant=MyInt8Weight,
                                bias=False,
                                return_quant_tensor=False),
                        qnn.QuantConv2d(
                                67, 64, 
                                kernel_size=(1, 3), stride=(1, 1), padding=(0,1), 
                                weight_bit_width=self.big_layers_weight_bw,
                                weight_quant=MyInt8Weight,
                                bias=True, 
                                bias_quant=MyIntBias))),
                    ("relu55", qnn.QuantReLU(
                                bit_width=self.act_bw, 
                                act_quant=MyUInt8Act,
                                return_quant_tensor=False)),

                    ("conv56", nn.Sequential(
                        qnn.QuantConv2d(
                                64, 77, 
                                kernel_size=(3, 1), stride=(1, 1), padding=(1,0), 
                                weight_bit_width=self.big_layers_weight_bw,
                                weight_quant=MyInt8Weight,
                                bias=False,
                                return_quant_tensor=False),
                        qnn.QuantConv2d(
                                77, 64, 
                                kernel_size=(1, 3), stride=(1, 1), padding=(0,1), 
                                weight_bit_width=self.big_layers_weight_bw,
                                weight_quant=MyInt8Weight,
                                bias=True, 
                                bias_quant=MyIntBias))),
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