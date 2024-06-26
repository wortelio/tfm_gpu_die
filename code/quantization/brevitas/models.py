import collections
import torch
import torch.nn as nn 
import brevitas.nn as qnn
#from brevitas.inject.defaults import Int8WeightPerTensorFloat, Uint8ActPerTensorFloat, Int8ActPerTensorFloat
from brevitas.quant import Int8WeightPerTensorFloat, Uint8ActPerTensorFloat, Int8ActPerTensorFloat
from brevitas.quant import Int32Bias
from brevitas.inject.defaults import RoundTo8bit
from brevitas.inject.enum import ScalingImplType, RestrictValueType, BitWidthImplType

class LearnedIntWeightPerChannelFloat(Int8WeightPerTensorFloat):
    scaling_per_output_channel = True
    scaling_impl_type = ScalingImplType.PARAMETER_FROM_STATS
    restrict_scaling_type = RestrictValueType.LOG_FP
    bit_width_impl_type = BitWidthImplType.PARAMETER

# ______________________________________________________________ #
# ____________________    QUANT WEIGHTS     ____________________ #
# ______________________________________________________________ #

class QUANTWeights_BED_CLASSIFIER(nn.Module):
    def __init__(self, num_classes, weight_bw, act_bw, bias_bw, in_channels=3):
        super(QUANTWeights_BED_CLASSIFIER, self).__init__()
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
                [   # Conv2d [in_channels, out_channels, kernel_size, stride, padding, bias]
          
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
    

    # def __initialize_weights__(self):
    #     for m in self.modules():
    #         if isinstance(m, nn.Conv2d):
    #             nn.init.kaiming_normal_(m.weight, mode='fan_in',
    #                 nonlinearity='relu'
    #             )
    #             if m.bias is not None:
    #                     nn.init.constant_(m.bias, 0)
    #         elif isinstance(m, nn.Linear):
    #             nn.init.normal_(m.weight, 0, 0.01)
    #             nn.init.constant_(m.bias, 0)


    def forward(self, x):
        x = self.model(x)
        return x



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