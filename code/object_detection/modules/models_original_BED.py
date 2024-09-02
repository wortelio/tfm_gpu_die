import collections
import torch
import torch.nn as nn 
import config

# ______________________________________________________________ #
# ____________________    ORIGINAL MODEL    ____________________ #
# ______________________________________________________________ #   
class ORIGINAL_BED_DETECTOR(nn.Module):
    def __init__(self, in_channels=3):
        super(ORIGINAL_BED_DETECTOR, self).__init__()
        self.in_channels = in_channels
        self.B = config.B
        self.C = config.C
        
        self.model = self.__create_BED__()
        
    def __create_BED__(self):
        BED_model = nn.Sequential(

            collections.OrderedDict(
                [
            # Conv2d [in_channels, out_channels, kernel_size, stride, padding, bias]

                    # To test BED manual optim
                    # Change the first convs 64 24, to 32 16
                    
                    # CNNBlock 224x224
                    #("conv1", nn.Conv2d(self.in_channels, 64, kernel_size=3, stride=1, padding=1,  bias=False)),
                    ("conv1", nn.Conv2d(self.in_channels, 32, kernel_size=3, stride=1, padding=1,  bias=False)), # To test BED manual optim
                    #("bn1", nn.BatchNorm2d(64)),
                    ("bn1", nn.BatchNorm2d(32)), # To test BED manual optim
                    ("relu1", nn.ReLU()),
                    #("dropout1", nn.Dropout2d(p=0.3)),
        
                    # CNNBlock 112x112
                    ("maxpool2", nn.MaxPool2d(kernel_size=2, stride=2)),
                    #("conv2", nn.Conv2d(64, 24, kernel_size=3, stride=1, padding=1,  bias=False)),
                    ("conv2", nn.Conv2d(32, 16, kernel_size=3, stride=1, padding=1,  bias=False)), # To test BED manual optim
                    #("bn2", nn.BatchNorm2d(24)),
                    ("bn2", nn.BatchNorm2d(16)),
                    ("relu2", nn.ReLU()),
                    #("dropout2",nn.Dropout2d(p=0.3)),
        
                    # CNNBlock 56x56
                    ("maxpool3", nn.MaxPool2d(kernel_size=2, stride=2)),
                    #("conv31", nn.Conv2d(24, 16, kernel_size=1, stride=1, padding=0,  bias=False)),
                    ("conv31", nn.Conv2d(16, 16, kernel_size=1, stride=1, padding=0,  bias=False)), # To test BED manual optim
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
                    
                    ("conv46", nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1,  bias=False)),
                    ("bn46", nn.BatchNorm2d(64)),
                    ("relu46", nn.ReLU()),
                    
                    # CNNBlock 14x14
                    ("maxpool5", nn.MaxPool2d(kernel_size=2, stride=2)), 
                    ("conv51", nn.Conv2d(64, 32, kernel_size=1, stride=1, padding=0,  bias=False)),
                    ("bn51", nn.BatchNorm2d(32)),
                    ("relu51", nn.ReLU()),

                    ("conv52", nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1,  bias=False)),
                    ("bn52", nn.BatchNorm2d(64)),
                    ("relu52", nn.ReLU()),

                    ("conv53", nn.Conv2d(64, 32, kernel_size=1, stride=1, padding=0,  bias=False)),
                    ("bn53", nn.BatchNorm2d(32)),
                    ("relu53", nn.ReLU()),

                    ("conv54", nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1,  bias=False)),
                    ("bn54", nn.BatchNorm2d(64)),
                    ("relu54", nn.ReLU()),

                    ("conv55", nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1,  bias=False)),
                    ("bn55", nn.BatchNorm2d(64)),
                    ("relu55", nn.ReLU()),

                    ("conv56", nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1,  bias=False)),
                    ("bn56", nn.BatchNorm2d(64)),
                    ("relu56", nn.ReLU()),
            
                    # CNNBlock 7x7
                    ("maxpool6", nn.MaxPool2d(kernel_size=2, stride=2)),
                    ("conv61", nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1,  bias=False)),
                    ("bn61", nn.BatchNorm2d(64)),
                    ("relu61", nn.ReLU()),
                    
                    ("conv62", nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1,  bias=False)),
                    ("bn62", nn.BatchNorm2d(64)),
                    ("relu62", nn.ReLU()),

                    # CNNBlock Out
                    ("conv71", nn.Conv2d(64, 64, kernel_size=1, stride=1, padding=0,  bias=False)),
                    ("bn71", nn.BatchNorm2d(64)),
                    ("relu71", nn.ReLU()),

                    ("conv72", nn.Conv2d(64, 16, kernel_size=1, stride=1, padding=0,  bias=False)),
                    ("bn72", nn.BatchNorm2d(16)),
                    ("relu72", nn.ReLU()),

                    ("conv73", nn.Conv2d(16, 16, kernel_size=1, stride=1, padding=0,  bias=False)),
                    ("bn73", nn.BatchNorm2d(16)),
                    ("relu73", nn.ReLU()),

                    ("conv74", nn.Conv2d(16, self.B*5 + self.C, kernel_size=1, stride=1, padding=0)),
                    
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
        return self.model(x) 
    
    
