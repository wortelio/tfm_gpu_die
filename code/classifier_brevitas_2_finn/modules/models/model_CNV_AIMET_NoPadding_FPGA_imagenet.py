import torch
from torch.nn import BatchNorm1d
from torch.nn import BatchNorm2d
from torch.nn import MaxPool2d
from torch.nn import Module
from torch.nn import ModuleList

from brevitas.core.restrict_val import RestrictValueType
from brevitas.nn import QuantIdentity
from brevitas.nn import QuantConv2d
from brevitas.nn import QuantReLU
from brevitas.nn import TruncAvgPool2d
from brevitas.nn import QuantLinear

from brevitas.quant import TruncTo8bit

from modules.models.common_imagenet import CommonIntWeightPerTensorQuant
from modules.models.common_imagenet import CommonIntWeightPerChannelQuant
from modules.models.common_imagenet import CommonUintActQuant
from modules.models.common_imagenet import CommonIntActQuant # For initial Q1.7 Identity Layer
from .tensor_norm import TensorNorm

import config

# ________________________________________________________________ #
#
# Balanced Model, with all channels adjusted to fit in BRAM efficiently
#
# ________________________________________________________________ #

# ________________________________________________________________ #
#          ADD Custom Quants for PerTensor and PerChannel          #
#                         and Fixed Point                          #
# ________________________________________________________________ #
class MyWeightsQuant_PerTensor(CommonIntWeightPerTensorQuant):
    restrict_scaling_type = RestrictValueType.POWER_OF_TWO

class MyWeightsQuant_PerChannel(CommonIntWeightPerChannelQuant):
    restrict_scaling_type = RestrictValueType.POWER_OF_TWO

class MyReLUQuant(CommonUintActQuant):
    restrict_scaling_type = RestrictValueType.POWER_OF_TWO

    
class CNV_AIMET(Module):

    def __init__(self, 
                 num_classes = config.NUM_CLASSES, 
                 small_layer_weight_bit_width = config.WEIGHTS_BIT_WIDTH,
                 big_layer_weight_bit_width = config.BIG_LAYERS_WEIGHTS_BIT_WIDTH,
                 act_bit_width = config.ACTIVATIONS_BIT_WIDTH, 
                 in_bit_width = 8, 
                 in_channels = config.NUM_CHANNELS):
        super(CNV_AIMET, self).__init__()

        self.num_classes = num_classes
        self.in_channels = in_channels

        self.small_layer_weight_bit_width = small_layer_weight_bit_width
        self.big_layer_weight_bit_width = big_layer_weight_bit_width
        
        self.return_quant_tensor = False
        
        self.conv_features = ModuleList()
        self.linear_features = ModuleList()

        # Input 230x230x3
        self.conv_features.append(QuantIdentity( # for Q1.7 input format -> sign.7bits
            act_quant = CommonIntActQuant,#CommonActQuant,
            bit_width = in_bit_width,
            min_val = -1.0,
            max_val = 1.0 - 2.0 ** (-7),
            narrow_range = False, 
            restrict_scaling_type = RestrictValueType.POWER_OF_TWO))

        # CNNBlock 230x230
            # conv1
        self.conv_features.append(
            QuantConv2d(
                kernel_size=3, stride=1, padding=0,
                in_channels=self.in_channels,
                out_channels=12,
                bias=False,
                weight_quant=MyWeightsQuant_PerChannel,
                weight_bit_width=self.small_layer_weight_bit_width))
        self.conv_features.append(BatchNorm2d(12))
        self.conv_features.append(
            QuantReLU(
                act_quant=MyReLUQuant,
                bit_width=act_bit_width, 
                return_quant_tensor=self.return_quant_tensor))
        
        self.conv_features.append(MaxPool2d(kernel_size=2, stride=2))

        # CNNBlock 114x114
            # conv2
        self.conv_features.append(
            QuantConv2d(
                kernel_size=(3,1), stride=1, padding=(0,0),
                in_channels=12,
                out_channels=24,
                bias=False,
                weight_quant=MyWeightsQuant_PerChannel,
                weight_bit_width=self.small_layer_weight_bit_width))

        # New insert
        # self.conv_features.append(BatchNorm2d(8))
        # self.conv_features.append(
        #     QuantReLU(
        #         act_quant=MyReLUQuant,
        #         bit_width=act_bit_width, 
        #         return_quant_tensor=self.return_quant_tensor))
        self.conv_features.append(
            QuantIdentity(
                act_quant = CommonIntActQuant,
                bit_width = act_bit_width,
                narrow_range = False,
                restrict_scaling_type = RestrictValueType.POWER_OF_TWO))
        # New insert

        self.conv_features.append(
            QuantConv2d(
                kernel_size=(1,3), stride=1, padding=(0,0),
                in_channels=24,
                out_channels=16,
                bias=False,
                weight_quant=MyWeightsQuant_PerChannel,
                weight_bit_width=self.small_layer_weight_bit_width))
        self.conv_features.append(BatchNorm2d(16))
        self.conv_features.append(
            QuantReLU(
                act_quant=MyReLUQuant,
                bit_width=act_bit_width, 
                return_quant_tensor=self.return_quant_tensor))
        
        self.conv_features.append(MaxPool2d(kernel_size=2, stride=2))

        # CNNBlock 56x56 
            # conv31
        self.conv_features.append(
            QuantConv2d(
                kernel_size=1, stride=1, padding=0,
                in_channels=16,
                out_channels=16,
                bias=False,
                weight_quant=MyWeightsQuant_PerChannel,
                weight_bit_width=self.small_layer_weight_bit_width))
        self.conv_features.append(BatchNorm2d(16))
        self.conv_features.append(
            QuantReLU(
                act_quant=MyReLUQuant,
                bit_width=act_bit_width, 
                return_quant_tensor=self.return_quant_tensor))            
            # conv32
        self.conv_features.append(
            QuantConv2d(
                kernel_size=(3,1), stride=1, padding=(0,0),
                in_channels=16,
                out_channels=16,
                bias=False,
                weight_quant=MyWeightsQuant_PerChannel,
                weight_bit_width=self.small_layer_weight_bit_width))

        # New insert
        # self.conv_features.append(BatchNorm2d(16))
        # self.conv_features.append(
        #     QuantReLU(
        #         act_quant=MyReLUQuant,
        #         bit_width=act_bit_width, 
        #         return_quant_tensor=self.return_quant_tensor))
        self.conv_features.append(
            QuantIdentity(
                act_quant = CommonIntActQuant,
                bit_width = act_bit_width,
                narrow_range = False,
                restrict_scaling_type = RestrictValueType.POWER_OF_TWO))
        # New insert

        self.conv_features.append(
            QuantConv2d(
                kernel_size=(1,3), stride=1, padding=(0,0),
                in_channels=16,
                out_channels=32,
                bias=False,
                weight_quant=MyWeightsQuant_PerChannel,
                weight_bit_width=self.small_layer_weight_bit_width))
        self.conv_features.append(BatchNorm2d(32))
        self.conv_features.append(
            QuantReLU(
                act_quant=MyReLUQuant,
                bit_width=act_bit_width, 
                return_quant_tensor=self.return_quant_tensor))  
            # conv33
        self.conv_features.append(
            QuantConv2d(
                kernel_size=1, stride=1, padding=0,
                in_channels=32,
                out_channels=32,
                bias=False,
                weight_quant=MyWeightsQuant_PerChannel,
                weight_bit_width=self.small_layer_weight_bit_width))
        self.conv_features.append(BatchNorm2d(32))
        self.conv_features.append(
            QuantReLU(
                act_quant=MyReLUQuant,
                bit_width=act_bit_width, 
                return_quant_tensor=self.return_quant_tensor))            
            # conv34
        self.conv_features.append(
            QuantConv2d(
                kernel_size=(3,1), stride=1, padding=(0,0),
                in_channels=32,
                out_channels=42,
                bias=False,
                weight_quant=MyWeightsQuant_PerChannel,
                weight_bit_width=self.small_layer_weight_bit_width))
        
        # New insert
        # self.conv_features.append(BatchNorm2d(32))
        # self.conv_features.append(
        #     QuantReLU(
        #         act_quant=MyReLUQuant,
        #         bit_width=act_bit_width, 
        #         return_quant_tensor=self.return_quant_tensor))
        self.conv_features.append(
            QuantIdentity(
                act_quant = CommonIntActQuant,
                bit_width = act_bit_width,
                narrow_range = False,
                restrict_scaling_type = RestrictValueType.POWER_OF_TWO))
        # New insert

        self.conv_features.append(
            QuantConv2d(
                kernel_size=(1,3), stride=1, padding=(0,0),
                in_channels=42,
                out_channels=64,
                bias=False,
                weight_quant=MyWeightsQuant_PerChannel,
                weight_bit_width=self.big_layer_weight_bit_width))
        self.conv_features.append(BatchNorm2d(64))
        self.conv_features.append(
            QuantReLU(
                act_quant=MyReLUQuant,
                bit_width=act_bit_width, 
                return_quant_tensor=self.return_quant_tensor))
        
        self.conv_features.append(MaxPool2d(kernel_size=2, stride=2))

        # CNNBlock 28x28 
            # conv41
        self.conv_features.append(
            QuantConv2d(
                kernel_size=1, stride=1, padding=0,
                in_channels=64,
                out_channels=30,
                bias=False,
                weight_quant=MyWeightsQuant_PerChannel,
                weight_bit_width=self.small_layer_weight_bit_width))
        self.conv_features.append(BatchNorm2d(30))
        self.conv_features.append(
            QuantReLU(
                act_quant=MyReLUQuant,
                bit_width=act_bit_width, 
                return_quant_tensor=self.return_quant_tensor))
            # conv42
        self.conv_features.append(
            QuantConv2d(
                kernel_size=3, stride=1, padding=0,
                in_channels=30,
                out_channels=60,
                bias=False,
                weight_quant=MyWeightsQuant_PerChannel,
                weight_bit_width=self.big_layer_weight_bit_width))
        self.conv_features.append(BatchNorm2d(60))
        self.conv_features.append(
            QuantReLU(
                act_quant=MyReLUQuant,
                bit_width=act_bit_width, 
                return_quant_tensor=self.return_quant_tensor))
            # conv43
        self.conv_features.append(
            QuantConv2d(
                kernel_size=1, stride=1, padding=0,
                in_channels=60,
                out_channels=32,
                bias=False,
                weight_quant=MyWeightsQuant_PerChannel,
                weight_bit_width=self.small_layer_weight_bit_width))
        self.conv_features.append(BatchNorm2d(32))
        self.conv_features.append(
            QuantReLU(
                act_quant=MyReLUQuant,
                bit_width=act_bit_width, 
                return_quant_tensor=self.return_quant_tensor))
            # conv44
        self.conv_features.append(
            QuantConv2d(
                kernel_size=(3,1), stride=1, padding=(0,0),
                in_channels=32,
                out_channels=42,
                bias=False,
                weight_quant=MyWeightsQuant_PerChannel,
                weight_bit_width=self.small_layer_weight_bit_width))
        
        # New insert
        # self.conv_features.append(BatchNorm2d(48))
        # self.conv_features.append(
        #     QuantReLU(
        #         act_quant=MyReLUQuant,
        #         bit_width=act_bit_width, 
        #         return_quant_tensor=self.return_quant_tensor))
        self.conv_features.append(
            QuantIdentity(
                act_quant = CommonIntActQuant,
                bit_width = act_bit_width,
                narrow_range = False,
                restrict_scaling_type = RestrictValueType.POWER_OF_TWO))
        # New insert
        
        self.conv_features.append(
            QuantConv2d(
                kernel_size=(1,3), stride=1, padding=(0,0),
                in_channels=42,
                out_channels=58,
                bias=False,
                weight_quant=MyWeightsQuant_PerChannel,
                weight_bit_width=self.big_layer_weight_bit_width))
        self.conv_features.append(BatchNorm2d(58))
        self.conv_features.append(
            QuantReLU(
                act_quant=MyReLUQuant,
                bit_width=act_bit_width, 
                return_quant_tensor=self.return_quant_tensor))
            # conv45
        self.conv_features.append(
            QuantConv2d(
                kernel_size=1, stride=1, padding=0,
                in_channels=58,
                out_channels=32,
                bias=False,
                weight_quant=MyWeightsQuant_PerChannel,
                weight_bit_width=self.small_layer_weight_bit_width))
        self.conv_features.append(BatchNorm2d(32))
        self.conv_features.append(
            QuantReLU(
                act_quant=MyReLUQuant,
                bit_width=act_bit_width, 
                return_quant_tensor=self.return_quant_tensor))
            # conv46
        self.conv_features.append(
            QuantConv2d(
                kernel_size=(3,1), stride=1, padding=(0,0),
                in_channels=32,
                out_channels=20,
                bias=False,
                weight_quant=MyWeightsQuant_PerChannel,
                weight_bit_width=self.small_layer_weight_bit_width))
        
        # New insert
        # self.conv_features.append(BatchNorm2d(16))
        # self.conv_features.append(
        #     QuantReLU(
        #         act_quant=MyReLUQuant,
        #         bit_width=act_bit_width, 
        #         return_quant_tensor=self.return_quant_tensor))
        self.conv_features.append(
            QuantIdentity(
                act_quant = CommonIntActQuant,
                bit_width = act_bit_width,
                narrow_range = False,
                restrict_scaling_type = RestrictValueType.POWER_OF_TWO))
        # New insert

        self.conv_features.append(
            QuantConv2d(
                kernel_size=(1,3), stride=1, padding=(0,0),
                in_channels=20,
                out_channels=64,
                bias=False,
                weight_quant=MyWeightsQuant_PerChannel,
                weight_bit_width=self.small_layer_weight_bit_width))
        self.conv_features.append(BatchNorm2d(64))
        self.conv_features.append(
            QuantReLU(
                act_quant=MyReLUQuant,
                bit_width=act_bit_width, 
                return_quant_tensor=True)) # Important to keep this TRUE for GlobalAvgPool
        
       # Average Pooling
        self.conv_features.append(TruncAvgPool2d(
                kernel_size=20, # 28, -> 20 if IMG_W, IMG_H = 230 
                trunc_quant=TruncTo8bit,
                float_to_int_impl_type='FLOOR'))

        # Linear 1
        self.linear_features.append(
            QuantLinear(
                in_features=64,
                out_features=32,
                bias=False,
                weight_quant=MyWeightsQuant_PerTensor,
                weight_bit_width=self.small_layer_weight_bit_width))
        self.linear_features.append(BatchNorm1d(32))
        self.linear_features.append(
            QuantReLU(
                act_quant=MyReLUQuant,
                bit_width=act_bit_width, 
                return_quant_tensor=self.return_quant_tensor))

        # Linear 2
        self.linear_features.append(
            QuantLinear(
                in_features=32,
                out_features=self.num_classes,
                bias=False,
                weight_quant=MyWeightsQuant_PerTensor,
                weight_bit_width=8))
        self.linear_features.append(TensorNorm())


    def forward(self, x):
        x = 2.0 * x - torch.tensor([1.0], device=x.device)
        for mod in self.conv_features:
            x = mod(x)
        x = x.view(x.shape[0], -1)
        for mod in self.linear_features:
            x = mod(x)
        return x

