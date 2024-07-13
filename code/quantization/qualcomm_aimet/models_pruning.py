import collections
import torch
import torch.nn as nn 

# ______________________________________________________________ #
# ____________________    NO BATCH NORM     ____________________ #
# ______________________________________________________________ #
class PRUNED_BED_CLASSIFIER(nn.Module):
    def __init__(self, num_classes, in_channels=3):
        super(PRUNED_BED_CLASSIFIER, self).__init__()
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
                    ("conv1", nn.Conv2d(self.in_channels, 28, kernel_size=3, stride=1, padding=1)),
                    ("relu1", nn.ReLU()),
                    ("dropout1", nn.Dropout2d(p=0.3)),
        
                    # CNNBlock 112x112
                    ("maxpool2", nn.MaxPool2d(kernel_size=2, stride=2)),
                    ("conv2", nn.Conv2d(28, 12, kernel_size=3, stride=1, padding=1)),
                    ("relu2", nn.ReLU()),
                    ("dropout2",nn.Dropout2d(p=0.3)),
        
                    # CNNBlock 56x56
                    ("maxpool3", nn.MaxPool2d(kernel_size=2, stride=2)),
                    ("conv31", nn.Conv2d(12, 8, kernel_size=1, stride=1, padding=0)),
                    ("relu31", nn.ReLU()),
                    
                    ("conv32", nn.Conv2d(8, 22, kernel_size=3, stride=1, padding=1)),
                    ("relu32", nn.ReLU()),
                    
                    ("conv33", nn.Conv2d(22, 32, kernel_size=1, stride=1, padding=0)),
                    ("relu33", nn.ReLU()),
                    
                    ("conv34", nn.Conv2d(32, 57, kernel_size=3, stride=1, padding=1)),
                    ("relu34", nn.ReLU()),
        
                    # CNNBlock 28x28
                    ("maxpool4", nn.MaxPool2d(kernel_size=2, stride=2)),
                    ("conv41", nn.Conv2d(57, 32, kernel_size=1, stride=1, padding=0)),
                    ("relu41", nn.ReLU()),
                    
                    ("conv42", nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)),
                    ("relu42", nn.ReLU()),
                    
                    ("conv43", nn.Conv2d(64, 32, kernel_size=1, stride=1, padding=0)),
                    ("relu43", nn.ReLU()),
                    
                    ("conv44", nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)),
                    ("relu44", nn.ReLU()),
                    
                    ("conv45", nn.Conv2d(64, 25, kernel_size=1, stride=1, padding=0)),
                    ("relu45", nn.ReLU()),
                    
                    ("conv46", nn.Conv2d(25, self.last_channels, kernel_size=3, stride=1, padding=1)),
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