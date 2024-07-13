import collections
import torch
import torch.nn as nn 
import brevitas.nn as qnn
from brevitas.quant import Int8WeightPerTensorFloat, Uint8ActPerTensorFloat, Int8ActPerTensorFloat
from brevitas.quant import Int8WeightPerTensorFixedPoint, Uint8ActPerTensorFixedPoint, Int8ActPerTensorFixedPoint
from brevitas.quant import Int32Bias, Int16Bias, Int8Bias, IntBias
from brevitas.quant import Int8BiasPerTensorFixedPointInternalScaling
from brevitas.quant import Int8WeightPerChannelFloat, Int8WeightPerChannelFixedPoint, Int8WeightPerChannelFixedPointMSE
from brevitas.inject.defaults import RoundTo8bit
from brevitas.inject.enum import ScalingImplType, RestrictValueType, BitWidthImplType
from brevitas.core.scaling import ConstScaling

import config

# ______________________________________________________________ #
# ____________________    QUANT WEIGHTS     ____________________ #
# ______________________________________________________________ #

class QUANT_BED_CLASSIFIER(nn.Module):
    def __init__(self, num_classes, weight_bw, act_bw, bias_bw, in_channels=3):
        super(QUANT_BED_CLASSIFIER, self).__init__()
        self.in_channels = in_channels
        self.last_channels = 64
        self.num_classes = num_classes

        self.weight_bw = weight_bw
        self.act_bw = act_bw
        self.bias_bw = bias_bw
        
        self.model = self.__create_BED__()
        
    def __create_BED__(self):
        BED_model = nn.Sequential(

            collections.OrderedDict(
                [   
                    ("input0", qnn.QuantIdentity(act_quant=Uint8ActPerTensorFloat)),
                    
                    # Conv2d [in_channels, out_channels, kernel_size, stride, padding, bias]
          
                    # CNNBlock 224x224
                    ("conv1", qnn.QuantConv2d(self.in_channels, 32, 
                                              kernel_size=3, stride=1, padding=1, 
                                              bias=False, weight_bit_width=self.weight_bw)),
                    #("bn1", nn.BatchNorm2d(32, affine=False)),
                    ("bn1", qnn.BatchNorm2dToQuantScaleBias(32)),
                    ("relu1", qnn.QuantReLU(bit_width=self.act_bw)),
                    ("dropout1", nn.Dropout2d(p=0.3)),
        
                    # CNNBlock 112x112
                    ("maxpool2", nn.MaxPool2d(kernel_size=2, stride=2)),
                    ("conv2", qnn.QuantConv2d(32, 16, 
                                              kernel_size=3, stride=1, padding=1, 
                                              bias=False, weight_bit_width=self.weight_bw)),
                    #("bn2", nn.BatchNorm2d(16, affine=False)),
                    ("bn2", qnn.BatchNorm2dToQuantScaleBias(16)),
                    ("relu2", qnn.QuantReLU(bit_width=self.act_bw)),
                    ("dropout2",nn.Dropout2d(p=0.3)),

                    # CNNBlock 56x56     
                    ("maxpool3", nn.MaxPool2d(kernel_size=2, stride=2)),
                    ("conv31", qnn.QuantConv2d(16, 16, 
                                               kernel_size=1, stride=1, padding=0, 
                                               bias=False, weight_bit_width=self.weight_bw)),
                    #("bn31", nn.BatchNorm2d(16, affine=False)),
                    ("bn31", qnn.BatchNorm2dToQuantScaleBias(16)),
                    ("relu31", qnn.QuantReLU(bit_width=self.act_bw)),
                    
                    ("conv32", qnn.QuantConv2d(16, 32, 
                                               kernel_size=3, stride=1, padding=1, 
                                               bias=False, weight_bit_width=self.weight_bw)),
                    #("bn32", nn.BatchNorm2d(32, affine=False)),
                    ("bn32", qnn.BatchNorm2dToQuantScaleBias(32)),
                    ("relu32", qnn.QuantReLU(bit_width=self.act_bw)),
                    
                    ("conv33", qnn.QuantConv2d(32, 32, 
                                               kernel_size=1, stride=1, padding=0, 
                                               bias=False, weight_bit_width=self.weight_bw)),
                    #("bn33", nn.BatchNorm2d(32, affine=False)),
                    ("bn33", qnn.BatchNorm2dToQuantScaleBias(32)),
                    ("relu33", qnn.QuantReLU(bit_width=self.act_bw)),
                    
                    ("conv34", qnn.QuantConv2d(32, 64, 
                                               kernel_size=3, stride=1, padding=1, 
                                               bias=False, weight_bit_width=self.weight_bw)),
                    #("bn34", nn.BatchNorm2d(64, affine=False)),
                    ("bn34", qnn.BatchNorm2dToQuantScaleBias(64)),
                    ("relu34", qnn.QuantReLU(bit_width=self.act_bw)),
        
                    # CNNBlock 28x28                                      
                    ("maxpool4", nn.MaxPool2d(kernel_size=2, stride=2)),
                    ("conv41", qnn.QuantConv2d(64, 32, 
                                               kernel_size=1, stride=1, padding=0, 
                                               bias=False, weight_bit_width=self.weight_bw)),
                    #("bn41", nn.BatchNorm2d(32, affine=False)),
                    ("bn41", qnn.BatchNorm2dToQuantScaleBias(32)),
                    ("relu41", qnn.QuantReLU(bit_width=self.act_bw)),
                    
                    ("conv42", qnn.QuantConv2d(32, 64, 
                                               kernel_size=3, stride=1, padding=1, 
                                               bias=False, weight_bit_width=self.weight_bw)),
                    #("bn42", nn.BatchNorm2d(64, affine=False)),
                    ("bn42", qnn.BatchNorm2dToQuantScaleBias(64)),
                    ("relu42", qnn.QuantReLU(bit_width=self.act_bw)),
                    
                    ("conv43", qnn.QuantConv2d(64, 32, 
                                               kernel_size=1, stride=1, padding=0, 
                                               bias=False, weight_bit_width=self.weight_bw)),
                    #("bn43", nn.BatchNorm2d(32, affine=False)),
                    ("bn43", qnn.BatchNorm2dToQuantScaleBias(32)),
                    ("relu43", qnn.QuantReLU(bit_width=self.act_bw)),
                    
                    ("conv44", qnn.QuantConv2d(32, 64, 
                                               kernel_size=3, stride=1, padding=1, 
                                               bias=False, weight_bit_width=self.weight_bw)),
                    #("bn44", nn.BatchNorm2d(64, affine=False)),
                    ("bn44", qnn.BatchNorm2dToQuantScaleBias(64)),
                    ("relu44", qnn.QuantReLU(bit_width=self.act_bw)),
                    
                    ("conv45", qnn.QuantConv2d(64, 32, 
                                               kernel_size=1, stride=1, padding=0, 
                                               bias=False, weight_bit_width=self.weight_bw)),
                    #("bn45", nn.BatchNorm2d(32, affine=False)),
                    ("bn45", qnn.BatchNorm2dToQuantScaleBias(32)),
                    ("relu45", qnn.QuantReLU(bit_width=self.act_bw)),
                    
                    ("conv46", qnn.QuantConv2d(32, self.last_channels, 
                                               kernel_size=3, stride=1, padding=1, 
                                               bias=False, weight_bit_width=self.weight_bw)),
                    #("bn46", nn.BatchNorm2d(self.last_channels, affine=False)),
                    ("bn46", qnn.BatchNorm2dToQuantScaleBias(self.last_channels)),
                    ("relu46", qnn.QuantReLU(
                                    bit_width=self.act_bw,
                                    return_quant_tensor=True)),
        
                    # Output One Head, 2 Neurons
                    # ("avgpool5", qnn.TruncAdaptiveAvgPool2d(
                    #                 (1, 1),
                    #                trunc_quant=RoundTo8bit,
                    #                return_quant_tensor=True)),
                    ("avgpool5", qnn.TruncAvgPool2d(                  # Adaptive Avg Pool replaced to avoid extra calculations
                                    kernel_size=(28, 28),
                                    stride=1,
                                    trunc_quant=RoundTo8bit,
                                    return_quant_tensor=True)),
                    ("flatten5", nn.Flatten(start_dim=1)),
                    ("dropout5", nn.Dropout(p=0.2)),
                    ("linear51", qnn.QuantLinear(
                                    in_features=self.last_channels, out_features=16, 
                                    bias=True,
                                    weight_bit_width=self.weight_bw,
                                    bias_quant=Int32Bias)),
                    ("relu5", qnn.QuantReLU(bit_width=self.act_bw,
                                           return_quant_tensor=True)),
                    ("linear52", qnn.QuantLinear(
                                    in_features=16, out_features=self.num_classes,
                                    bias=True,
                                    weight_bit_width=self.weight_bw,
                                    #weight_quant=Int8WeightPerTensorFloat,
                                    bias_quant=Int32Bias)),  
                ]
            )                    
        )
        
        
        return BED_model


    def forward(self, x):
        x = self.model(x)
        return x

# ______________________________________________________________ #
# ____________________    FIXED POINT Q     ____________________ #
# ______________________________________________________________ #
class QUANT_FixedPoint_BED_CLASSIFIER(nn.Module):
    def __init__(self, num_classes, weight_bw, act_bw, bias_bw, in_channels=3):
        super(QUANT_FixedPoint_BED_CLASSIFIER, self).__init__()
        self.in_channels = in_channels
        self.last_channels = 64
        self.num_classes = num_classes

        self.weight_bw = weight_bw
        self.act_bw = act_bw
        self.bias_bw = bias_bw
        
        self.model = self.__create_BED__()
        
    def __create_BED__(self):
        BED_model = nn.Sequential(

            collections.OrderedDict(
                [   
                    ("input0", qnn.QuantIdentity(act_quant=Uint8ActPerTensorFixedPoint)),
                    
                    # Conv2d [in_channels, out_channels, kernel_size, stride, padding, bias]
          
                    # CNNBlock 224x224
                    ("conv1", qnn.QuantConv2d(
                                self.in_channels, 32, 
                                kernel_size=3, stride=1, padding=1, 
                                bias=False, weight_bit_width=self.weight_bw,
                                weight_quant=Int8WeightPerTensorFixedPoint,
                                return_quant_tensor=True)),
                    ("bn1", qnn.BatchNorm2dToQuantScaleBias(
                                32,
                                weight_quant=Int8WeightPerTensorFixedPoint,
                                bias_bit_width = self.bias_bw,
                                bias_quant=Int8BiasPerTensorFixedPointInternalScaling)),
                    ("relu1", qnn.QuantReLU(bit_width=self.act_bw, act_quant=Uint8ActPerTensorFixedPoint)),
                    ("dropout1", nn.Dropout2d(p=0.3)),
        
                    # CNNBlock 112x112
                    ("maxpool2", nn.MaxPool2d(kernel_size=2, stride=2)),
                    ("conv2", qnn.QuantConv2d(
                                32, 16, 
                                kernel_size=3, stride=1, padding=1, 
                                bias=False, weight_bit_width=self.weight_bw,
                                weight_quant=Int8WeightPerTensorFixedPoint,
                                return_quant_tensor=True)),
                    ("bn2", qnn.BatchNorm2dToQuantScaleBias(
                                16,
                                weight_quant=Int8WeightPerTensorFixedPoint,
                                bias_bit_width = self.bias_bw,
                                bias_quant=Int8BiasPerTensorFixedPointInternalScaling)),
                    ("relu2", qnn.QuantReLU(bit_width=self.act_bw, act_quant=Uint8ActPerTensorFixedPoint)),
                    ("dropout2",nn.Dropout2d(p=0.3)),

                    # CNNBlock 56x56     
                    ("maxpool3", nn.MaxPool2d(kernel_size=2, stride=2)),
                    ("conv31", qnn.QuantConv2d(
                                16, 16, 
                                kernel_size=1, stride=1, padding=0, 
                                bias=False, weight_bit_width=self.weight_bw,
                                weight_quant=Int8WeightPerTensorFixedPoint,
                                return_quant_tensor=True)),
                    ("bn31", qnn.BatchNorm2dToQuantScaleBias(
                                16,
                                weight_quant=Int8WeightPerTensorFixedPoint,
                                bias_bit_width = self.bias_bw,
                                bias_quant=Int8BiasPerTensorFixedPointInternalScaling)),
                    ("relu31", qnn.QuantReLU(bit_width=self.act_bw, act_quant=Uint8ActPerTensorFixedPoint)),
                    
                    ("conv32", qnn.QuantConv2d(
                                16, 32, 
                                kernel_size=3, stride=1, padding=1, 
                                bias=False, weight_bit_width=self.weight_bw,
                                weight_quant=Int8WeightPerTensorFixedPoint,
                                return_quant_tensor=True)),
                    ("bn32", qnn.BatchNorm2dToQuantScaleBias(
                                32,
                                weight_quant=Int8WeightPerTensorFixedPoint,
                                bias_bit_width = self.bias_bw,
                                bias_quant=Int8BiasPerTensorFixedPointInternalScaling)),                        
                    ("relu32", qnn.QuantReLU(bit_width=self.act_bw, act_quant=Uint8ActPerTensorFixedPoint)),
                    
                    ("conv33", qnn.QuantConv2d(32, 32, 
                                kernel_size=1, stride=1, padding=0, 
                                bias=False, weight_bit_width=self.weight_bw,
                                weight_quant=Int8WeightPerTensorFixedPoint,
                                return_quant_tensor=True)),                                               
                    ("bn33", qnn.BatchNorm2dToQuantScaleBias(
                                32,
                                weight_quant=Int8WeightPerTensorFixedPoint,
                                bias_bit_width = self.bias_bw,
                                bias_quant=Int8BiasPerTensorFixedPointInternalScaling)),                          
                    ("relu33", qnn.QuantReLU(bit_width=self.act_bw, act_quant=Uint8ActPerTensorFixedPoint)),
                    
                    ("conv34", qnn.QuantConv2d(
                                32, 64, 
                                kernel_size=3, stride=1, padding=1, 
                                bias=False, weight_bit_width=self.weight_bw,
                                weight_quant=Int8WeightPerTensorFixedPoint,
                                return_quant_tensor=True)),                                                                       
                    ("bn34", qnn.BatchNorm2dToQuantScaleBias(
                                64,
                                weight_quant=Int8WeightPerTensorFixedPoint,
                                bias_quant=Int8BiasPerTensorFixedPointInternalScaling)),                        
                    ("relu34", qnn.QuantReLU(bit_width=self.act_bw, act_quant=Uint8ActPerTensorFixedPoint)),
        
                    # CNNBlock 28x28                                      
                    ("maxpool4", nn.MaxPool2d(kernel_size=2, stride=2)),
                    ("conv41", qnn.QuantConv2d(64, 32, 
                                kernel_size=1, stride=1, padding=0, 
                                bias=False, weight_bit_width=self.weight_bw,
                                weight_quant=Int8WeightPerTensorFixedPoint,
                                return_quant_tensor=True)),
                    ("bn41", qnn.BatchNorm2dToQuantScaleBias(
                                32,
                                weight_quant=Int8WeightPerTensorFixedPoint,
                                bias_bit_width = self.bias_bw,
                                bias_quant=Int8BiasPerTensorFixedPointInternalScaling)),                    
                    ("relu41", qnn.QuantReLU(bit_width=self.act_bw, act_quant=Uint8ActPerTensorFixedPoint)),
                    
                    ("conv42", qnn.QuantConv2d(32, 64, 
                                kernel_size=3, stride=1, padding=1, 
                                bias=False, weight_bit_width=self.weight_bw,
                                weight_quant=Int8WeightPerTensorFixedPoint,
                                return_quant_tensor=True)),
                    ("bn42", qnn.BatchNorm2dToQuantScaleBias(
                                64,
                                weight_quant=Int8WeightPerTensorFixedPoint,
                                bias_bit_width = self.bias_bw,
                                bias_quant=Int8BiasPerTensorFixedPointInternalScaling)),                        
                    ("relu42", qnn.QuantReLU(bit_width=self.act_bw, act_quant=Uint8ActPerTensorFixedPoint)),
                    
                    ("conv43", qnn.QuantConv2d(
                                64, 32, 
                                kernel_size=1, stride=1, padding=0, 
                                bias=False, weight_bit_width=self.weight_bw,
                                weight_quant=Int8WeightPerTensorFixedPoint,
                                return_quant_tensor=True)),
                    ("bn43", qnn.BatchNorm2dToQuantScaleBias(
                                32,
                                weight_quant=Int8WeightPerTensorFixedPoint,
                                bias_bit_width = self.bias_bw,
                                bias_quant=Int8BiasPerTensorFixedPointInternalScaling)),
                    ("relu43", qnn.QuantReLU(bit_width=self.act_bw, act_quant=Uint8ActPerTensorFixedPoint)),
                    
                    ("conv44", qnn.QuantConv2d(
                                32, 64, 
                                kernel_size=3, stride=1, padding=1, 
                                bias=False, weight_bit_width=self.weight_bw,
                                weight_quant=Int8WeightPerTensorFixedPoint,
                                return_quant_tensor=True)),
                    ("bn44", qnn.BatchNorm2dToQuantScaleBias(
                                64,
                                weight_quant=Int8WeightPerTensorFixedPoint,
                                bias_bit_width = self.bias_bw,
                                bias_quant=Int8BiasPerTensorFixedPointInternalScaling)),                        
                    ("relu44", qnn.QuantReLU(bit_width=self.act_bw, act_quant=Uint8ActPerTensorFixedPoint)),
                    
                    ("conv45", qnn.QuantConv2d(64, 32, 
                                kernel_size=1, stride=1, padding=0, 
                                bias=False, weight_bit_width=self.weight_bw,
                                weight_quant=Int8WeightPerTensorFixedPoint,
                                return_quant_tensor=True)),
                    ("bn45", qnn.BatchNorm2dToQuantScaleBias(
                                32,
                                weight_quant=Int8WeightPerTensorFixedPoint,
                                bias_bit_width = self.bias_bw,
                                bias_quant=Int8BiasPerTensorFixedPointInternalScaling)),                        
                    ("relu45", qnn.QuantReLU(bit_width=self.act_bw, act_quant=Uint8ActPerTensorFixedPoint)),
                    
                    ("conv46", qnn.QuantConv2d(
                                32, self.last_channels, 
                                kernel_size=3, stride=1, padding=1, 
                                bias=False, weight_bit_width=self.weight_bw,
                                weight_quant=Int8WeightPerTensorFixedPoint,
                                return_quant_tensor=True)),
                    ("bn46", qnn.BatchNorm2dToQuantScaleBias(
                                self.last_channels,
                                weight_quant=Int8WeightPerTensorFixedPoint,
                                bias_bit_width = self.bias_bw,
                                bias_quant=Int8BiasPerTensorFixedPointInternalScaling)),
                    ("relu46", qnn.QuantReLU(
                                    bit_width=self.act_bw,
                                    act_quant=Uint8ActPerTensorFixedPoint,
                                    return_quant_tensor=True)),
        
                    # Output One Head, 2 Neurons
                    ("avgpool5", qnn.TruncAvgPool2d(                  # Adaptive Avg Pool replaced to avoid extra calculations
                                    kernel_size=(28, 28),
                                    stride=1,
                                    trunc_quant=RoundTo8bit,
                                    return_quant_tensor=True)),
                    ("flatten5", nn.Flatten(start_dim=1)),
                    ("dropout5", nn.Dropout(p=0.2)),
                    ("linear51", qnn.QuantLinear(
                                    in_features=self.last_channels, out_features=16, 
                                    bias=True,
                                    weight_bit_width=self.weight_bw,
                                    bias_bit_width = self.bias_bw,
                                    weight_quant=Int8WeightPerTensorFixedPoint,
                                    bias_quant=Int8BiasPerTensorFixedPointInternalScaling)),
                    ("relu5", qnn.QuantReLU(
                                    bit_width=self.act_bw,
                                    act_quant=Uint8ActPerTensorFixedPoint, 
                                    return_quant_tensor=True)),
                    ("linear52", qnn.QuantLinear(
                                    in_features=16, out_features=self.num_classes,
                                    bias=True,
                                    weight_bit_width=self.weight_bw,
                                    bias_bit_width = self.bias_bw,
                                    weight_quant=Int8WeightPerTensorFixedPoint,
                                    bias_quant=Int8BiasPerTensorFixedPointInternalScaling)),  
                ]
            )                    
        )
        return BED_model


    def forward(self, x):
        x = self.model(x)
        return x

# ______________________________________________________________ #
# ____________________    FIXED POINT Q     ____________________ #
# ____________________    No Batch Norm     ____________________ #
# ______________________________________________________________ #
class MyQuantIdFixedPoint(Uint8ActPerTensorFixedPoint):
    bit_width = 8
    scaling_impl_type = ScalingImplType.CONST
    restrict_value_impl = RestrictValueType.POWER_OF_TWO
    scaling_init = 1 

class MyIntBias(Int8BiasPerTensorFixedPointInternalScaling):
#class MyIntBias(Int32Bias):    
    """
    8-bit signed int bias quantizer with scale factor equal to the scale factor of the accumulator
    the bias is added to, so typically quant_input_scale * quant_weight_scale.
    """
    bit_width = config.BIAS_BIT_WIDTH
    #pass

class MyInt8Weight(Int8WeightPerChannelFixedPointMSE):
    #scaling_per_output_channel = True
    #restrict_scaling_type = RestrictValueType.POWER_OF_TWO
    pass

   

class QUANT_FixedPoint_NoBN_BED_CLASSIFIER(nn.Module):
    def __init__(self, num_classes, 
                 weight_bw, 
                 big_layers_weight_bw,
                 act_bw, 
                 bias_bw, 
                 in_channels=3):
        super(QUANT_FixedPoint_NoBN_BED_CLASSIFIER, self).__init__()
        self.in_channels = in_channels
        self.last_channels = 64
        self.num_classes = num_classes

        self.weight_bw = weight_bw
        self.big_layers_weight_bw = big_layers_weight_bw
        self.act_bw = act_bw
        self.bias_bw = bias_bw
        
        self.model = self.__create_BED__()
        
    def __create_BED__(self):
        BED_model = nn.Sequential(

            collections.OrderedDict(
                [   
                    ("input0", qnn.QuantIdentity(
                                act_quant=MyQuantIdFixedPoint,        
                                return_quant_tensor=True)),
                    
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
                                act_quant=Uint8ActPerTensorFixedPoint,
                                return_quant_tensor=True)),
                    ("dropout1", nn.Dropout2d(p=0.3)),
        
                    # CNNBlock 112x112
                    ("maxpool2", nn.MaxPool2d(kernel_size=2, stride=2)),
                    ("conv2", qnn.QuantConv2d(
                                32, 16, 
                                kernel_size=3, stride=1, padding=1, 
                                weight_bit_width=self.weight_bw,
                                weight_quant=MyInt8Weight,
                                bias=True, 
                                bias_quant=MyIntBias)),
                    ("relu2", qnn.QuantReLU(
                                bit_width=self.act_bw, 
                                act_quant=Uint8ActPerTensorFixedPoint,
                                return_quant_tensor=True)),
                    ("dropout2",nn.Dropout2d(p=0.3)),

                    # CNNBlock 56x56     
                    ("maxpool3", nn.MaxPool2d(kernel_size=2, stride=2)),
                    ("conv31", qnn.QuantConv2d(
                                16, 16, 
                                kernel_size=1, stride=1, padding=0, 
                                weight_bit_width=self.weight_bw,
                                weight_quant=MyInt8Weight,
                                bias=True, 
                                bias_quant=MyIntBias)),
                    ("relu31", qnn.QuantReLU(
                                bit_width=self.act_bw, 
                                act_quant=Uint8ActPerTensorFixedPoint,
                                return_quant_tensor=True)),
                    
                    ("conv32", qnn.QuantConv2d(
                                16, 32, 
                                kernel_size=3, stride=1, padding=1, 
                                weight_bit_width=self.weight_bw,
                                weight_quant=MyInt8Weight,
                                bias=True,
                                bias_quant=MyIntBias)),
                    ("relu32", qnn.QuantReLU(
                                bit_width=self.act_bw, 
                                act_quant=Uint8ActPerTensorFixedPoint,
                                return_quant_tensor=True)),
                    
                    ("conv33", qnn.QuantConv2d(32, 32, 
                                kernel_size=1, stride=1, padding=0, 
                                weight_bit_width=self.weight_bw,
                                weight_quant=MyInt8Weight,
                                bias=True, 
                                bias_quant=MyIntBias)),
                    ("relu33", qnn.QuantReLU(
                                bit_width=self.act_bw, 
                                act_quant=Uint8ActPerTensorFixedPoint,
                                return_quant_tensor=True)),
                    
                    ("conv34", qnn.QuantConv2d(
                                32, 64, 
                                kernel_size=3, stride=1, padding=1, 
                                weight_bit_width=self.big_layers_weight_bw,
                                weight_quant=MyInt8Weight,
                                bias=True, 
                                bias_quant=MyIntBias)),
                    ("relu34", qnn.QuantReLU(
                                bit_width=self.act_bw, 
                                act_quant=Uint8ActPerTensorFixedPoint,
                                return_quant_tensor=True)),
        
                    # CNNBlock 28x28                                      
                    ("maxpool4", nn.MaxPool2d(kernel_size=2, stride=2)),
                    ("conv41", qnn.QuantConv2d(64, 32, 
                                kernel_size=1, stride=1, padding=0, 
                                weight_bit_width=self.weight_bw,
                                weight_quant=MyInt8Weight,
                                bias=True, 
                                bias_quant=MyIntBias)),
                    ("relu41", qnn.QuantReLU(
                                bit_width=self.act_bw, 
                                act_quant=Uint8ActPerTensorFixedPoint,
                                return_quant_tensor=True)),
                    
                    ("conv42", qnn.QuantConv2d(32, 64, 
                                kernel_size=3, stride=1, padding=1, 
                                weight_bit_width=self.big_layers_weight_bw,
                                weight_quant=MyInt8Weight,
                                bias=True, 
                                bias_quant=MyIntBias)),
                    ("relu42", qnn.QuantReLU(
                                bit_width=self.act_bw, 
                                act_quant=Uint8ActPerTensorFixedPoint,
                                return_quant_tensor=True)),
                    
                    ("conv43", qnn.QuantConv2d(
                                64, 32, 
                                kernel_size=1, stride=1, padding=0, 
                                weight_bit_width=self.weight_bw,
                                weight_quant=MyInt8Weight,
                                bias=True, 
                                bias_quant=MyIntBias)),
                    ("relu43", qnn.QuantReLU(
                                bit_width=self.act_bw, 
                                act_quant=Uint8ActPerTensorFixedPoint,
                                return_quant_tensor=True)),
                    
                    ("conv44", qnn.QuantConv2d(
                                32, 64, 
                                kernel_size=3, stride=1, padding=1, 
                                weight_bit_width=self.big_layers_weight_bw,
                                weight_quant=MyInt8Weight,
                                bias=True, 
                                bias_quant=MyIntBias)),
                    ("relu44", qnn.QuantReLU(
                                bit_width=self.act_bw, 
                                act_quant=Uint8ActPerTensorFixedPoint,
                                return_quant_tensor=True)),
                    
                    ("conv45", qnn.QuantConv2d(64, 32, 
                                kernel_size=1, stride=1, padding=0, 
                                weight_bit_width=self.weight_bw,
                                weight_quant=MyInt8Weight,
                                bias=True, 
                                bias_quant=MyIntBias)),
                    ("relu45", qnn.QuantReLU(
                                bit_width=self.act_bw, 
                                act_quant=Uint8ActPerTensorFixedPoint,
                                return_quant_tensor=True)),
                    
                    ("conv46", qnn.QuantConv2d(
                                32, self.last_channels, 
                                kernel_size=3, stride=1, padding=1, 
                                weight_bit_width=self.big_layers_weight_bw,
                                weight_quant=MyInt8Weight,
                                bias=True, 
                                bias_quant=MyIntBias)),
                    ("relu46", qnn.QuantReLU(
                                bit_width=self.act_bw,
                                act_quant=Uint8ActPerTensorFixedPoint,
                                return_quant_tensor=True)),
        
                    # Output One Head, 2 Neurons
                    ("avgpool5", qnn.TruncAvgPool2d(                  # Adaptive Avg Pool replaced to avoid extra calculations
                                kernel_size=(28, 28),
                                stride=1,
                                trunc_quant=RoundTo8bit,
                                return_quant_tensor=True)),
                    ("flatten5", nn.Flatten(start_dim=1)),
                    ("dropout5", nn.Dropout(p=0.2)),
                    ("linear51", qnn.QuantLinear(
                                in_features=self.last_channels, out_features=16, 
                                bias=True,
                                weight_bit_width=self.weight_bw,
                                weight_quant=MyInt8Weight,
                                bias_quant=MyIntBias)),
                    ("relu5", qnn.QuantReLU(
                                bit_width=self.act_bw,
                                act_quant=Uint8ActPerTensorFixedPoint, 
                                return_quant_tensor=True)),
                    ("linear52", qnn.QuantLinear(
                                in_features=16, out_features=self.num_classes,
                                bias=True,
                                weight_bit_width=self.weight_bw,
                                weight_quant=MyInt8Weight,
                                bias_quant=MyIntBias)),
                ]
            )                    
        )     
        return BED_model


    def forward(self, x):
        x = self.model(x)
        return x


# ______________________________________________________________ #
# ____________________    NO BATCH NORM     ____________________ #
# ______________________________________________________________ #
class FUSED_BED_CLASSIFIER(nn.Module):
    def __init__(self, num_classes, in_channels=3):
        super(FUSED_BED_CLASSIFIER, self).__init__()
        self.in_channels = in_channels
        self.last_channels = 64
        self.num_classes = num_classes
        
        self.model = self.__create_BED__()
        
    def __create_BED__(self):
        BED_model = nn.Sequential(

            collections.OrderedDict(
                [
            # Conv2d [in_channels, out_channels, kernel_size, stride, padding, bias]

                    # CNNBlock 224x224
                    ("conv1", nn.Conv2d(self.in_channels, 32, kernel_size=3, stride=1, padding=1)),
                    ("relu1", nn.ReLU()),
                    ("dropout1", nn.Dropout2d(p=0.3)),
        
                    # CNNBlock 112x112
                    ("maxpool2", nn.MaxPool2d(kernel_size=2, stride=2)),
                    ("conv2", nn.Conv2d(32, 16, kernel_size=3, stride=1, padding=1)),
                    ("relu2", nn.ReLU()),
                    ("dropout2",nn.Dropout2d(p=0.3)),
        
                    # CNNBlock 56x56
                    ("maxpool3", nn.MaxPool2d(kernel_size=2, stride=2)),
                    ("conv31", nn.Conv2d(16, 16, kernel_size=1, stride=1, padding=0)),
                    ("relu31", nn.ReLU()),
                    
                    ("conv32", nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)),
                    ("relu32", nn.ReLU()),
                    
                    ("conv33", nn.Conv2d(32, 32, kernel_size=1, stride=1, padding=0)),
                    ("relu33", nn.ReLU()),
                    
                    ("conv34", nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)),
                    ("relu34", nn.ReLU()),
        
                    # CNNBlock 28x28
                    ("maxpool4", nn.MaxPool2d(kernel_size=2, stride=2)),
                    ("conv41", nn.Conv2d(64, 32, kernel_size=1, stride=1, padding=0)),
                    ("relu41", nn.ReLU()),
                    
                    ("conv42", nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)),
                    ("relu42", nn.ReLU()),
                    
                    ("conv43", nn.Conv2d(64, 32, kernel_size=1, stride=1, padding=0)),
                    ("relu43", nn.ReLU()),
                    
                    ("conv44", nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)),
                    ("relu44", nn.ReLU()),
                    
                    ("conv45", nn.Conv2d(64, 32, kernel_size=1, stride=1, padding=0)),
                    ("relu45", nn.ReLU()),
                    
                    ("conv46", nn.Conv2d(32, self.last_channels, kernel_size=3, stride=1, padding=1)),
                    ("relu46", nn.ReLU()),
        
                    # Output One Head, 2 Neurons
                    ("avgpool5", nn.AdaptiveAvgPool2d((1, 1))),
                    ("flatten5", nn.Flatten(start_dim=1)),
                    ("dropout5", nn.Dropout(p=0.2)),
                    ("linear51", nn.Linear(in_features=self.last_channels, out_features=16)),
                    ("relu5", nn.ReLU()),
                    ("linear52", nn.Linear(in_features=16, out_features=2)),
                ]
            )
        )
        return BED_model
    
    def forward(self, x):
        x = self.model(x)
        return x


# ______________________________________________________________ #
# ____________________    Original Model    ____________________ #
# ______________________________________________________________ #
class BED_CLASSIFIER(nn.Module):
    def __init__(self, num_classes, in_channels=3):
        super(BED_CLASSIFIER, self).__init__()
        self.in_channels = in_channels
        self.last_channels = 64
        self.num_classes = num_classes
        
        self.model = self.__create_BED__()
        
    def __create_BED__(self):
        BED_model = nn.Sequential(

            collections.OrderedDict(
                [
            # Conv2d [in_channels, out_channels, kernel_size, stride, padding, bias]

                    # CNNBlock 224x224
                    ("conv1", nn.Conv2d(self.in_channels, 32, kernel_size=3, stride=1, padding=1,  bias=False)),
                    ("bn1", nn.BatchNorm2d(32)),
                    ("relu1", nn.ReLU()),
                    ("dropout1", nn.Dropout2d(p=0.3)),
        
                    # CNNBlock 112x112
                    ("maxpool2", nn.MaxPool2d(kernel_size=2, stride=2)),
                    ("conv2", nn.Conv2d(32, 16, kernel_size=3, stride=1, padding=1,  bias=False)),
                    ("bn2", nn.BatchNorm2d(16)),
                    ("relu2", nn.ReLU()),
                    ("dropout2",nn.Dropout2d(p=0.3)),
        
                    # CNNBlock 56x56
                    ("maxpool3", nn.MaxPool2d(kernel_size=2, stride=2)),
                    ("conv31", nn.Conv2d(16, 16, kernel_size=1, stride=1, padding=0,  bias=False)),
                    ("bn31", nn.BatchNorm2d(16)),
                    ("relu31", nn.ReLU()),
                    
                    ("conv32", nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1,  bias=False)),
                    ("bn32", nn.BatchNorm2d(32)),
                    ("relu32", nn.ReLU()),
                    
                    ("conv33", nn.Conv2d(32, 32, kernel_size=1, stride=1, padding=0,  bias=False)),
                    ("bn33", nn.BatchNorm2d(32)),
                    ("relu33", nn.ReLU()),
                    
                    ("conv34", nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1,  bias=False)),
                    ("bn34", nn.BatchNorm2d(64)),
                    ("relu34", nn.ReLU()),
        
                    # CNNBlock 28x28
                    ("maxpool4", nn.MaxPool2d(kernel_size=2, stride=2)),
                    ("conv41", nn.Conv2d(64, 32, kernel_size=1, stride=1, padding=0,  bias=False)),
                    ("bn41", nn.BatchNorm2d(32)),
                    ("relu41", nn.ReLU()),
                    
                    ("conv42", nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1,  bias=False)),
                    ("bn42", nn.BatchNorm2d(64)),
                    ("relu42", nn.ReLU()),
                    
                    ("conv43", nn.Conv2d(64, 32, kernel_size=1, stride=1, padding=0,  bias=False)),
                    ("bn43", nn.BatchNorm2d(32)),
                    ("relu43", nn.ReLU()),
                    
                    ("conv44", nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1,  bias=False)),
                    ("bn44", nn.BatchNorm2d(64)),
                    ("relu44", nn.ReLU()),
                    
                    ("conv45", nn.Conv2d(64, 32, kernel_size=1, stride=1, padding=0,  bias=False)),
                    ("bn45", nn.BatchNorm2d(32)),
                    ("relu45", nn.ReLU()),
                    
                    ("conv46", nn.Conv2d(32, self.last_channels, kernel_size=3, stride=1, padding=1,  bias=False)),
                    ("bn46", nn.BatchNorm2d(self.last_channels)),
                    ("relu46", nn.ReLU()),
        
                    # Output One Head, 2 Neurons
                    ("avgpool5", nn.AdaptiveAvgPool2d((1, 1))),
                    ("flatten5", nn.Flatten(start_dim=1)),
                    ("dropout5", nn.Dropout(p=0.2)),
                    ("linear51", nn.Linear(in_features=self.last_channels, out_features=16)),
                    ("relu5", nn.ReLU()),
                    ("linear52", nn.Linear(in_features=16, out_features=2)),
                ]
            )
        )
        return BED_model
    

    def __initialize_weights__(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in',
                    nonlinearity='relu'
                )
                if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


    def forward(self, x):
        x = self.model(x)
        return x