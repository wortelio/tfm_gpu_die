import collections
import torch
import torch.nn as nn 
import config

class Reshape(nn.Module):
    def __init__(self, shape):
        super(Reshape, self).__init__()
        self.shape = shape

    def forward(self, x):
        return x.view(self.shape)
    
class Tinyissimo(nn.Module):
    def __init__(self, in_channels=3):
        
        super(Tinyissimo, self).__init__()
        self.in_channels = in_channels
        self.C = config.C
        self.S = config.S
        self.B = config.B
        
        self.tiny_model = self.__create_tinyssimo__()
    
    def __create_tinyssimo__(self):
        tinyssimo_model = nn.Sequential(
            # Conv2d [in_channels, out_channels, kernel_size, stride, padding, bias]

            # CNNBlock 1
            nn.Conv2d(self.in_channels, 16, kernel_size=3, stride=1, padding='same', bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 16, kernel_size=3, stride=1, padding='same', bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # CNNBlock 2
            nn.Conv2d(16, 16, kernel_size=3, stride=1, padding='same', bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding='same', bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # CNNBlock 3
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding='same', bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding='same', bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # CNNBlock 4
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding='same', bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding='same', bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2), 

            # CNNBlock 5
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding='same', bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding='same', bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # HEAD
            nn.Flatten(),
            nn.Linear(in_features=128*2*2, out_features=192), # 2 and 2 are dims b4 FC layer
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(in_features=192, out_features= self.S * self.S * ((self.B * 5) + self.C)),
            # Adding Reshape
            Reshape((-1, 12, 4, 4)), # To fit with permute out of the model           
        )
        return tinyssimo_model 

    def _initialize_weights(self):
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
        return self.tiny_model(x)