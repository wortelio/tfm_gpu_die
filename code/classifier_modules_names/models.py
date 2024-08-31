import collections
import torch
import torch.nn as nn 

# ______________________________________________________________ #
# ____________________   Manually Reduced   ____________________ #
# ____________________         MODEL        ____________________ #
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


# ______________________________________________________________ #
# ____________________    ORIGINAL MODEL    ____________________ #
# ______________________________________________________________ #
class ORIGINAL_BED_CLASSIFIER(nn.Module):
    def __init__(self, num_classes=2, in_channels=3):
        super(ORIGINAL_BED_CLASSIFIER, self).__init__()
        self.in_channels = in_channels
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
        
                    # Output One Head, 2 Neurons
                    ("avgpool5", nn.AdaptiveAvgPool2d((1, 1))),
                    ("flatten5", nn.Flatten(start_dim=1)),
                    ("dropout5", nn.Dropout(p=0.2)),
                    ("linear51", nn.Linear(in_features=64, out_features=16)),
                    ("relu5", nn.ReLU()),
                    ("linear52", nn.Linear(in_features=16, out_features=self.num_classes)),
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
    
# ______________________________________________________________ #
# ____________________    NO BATCH NORM     ____________________ #
# ______________________________________________________________ #
class NoBN_BED_CLASSIFIER(nn.Module):
    def __init__(self, num_classes, in_channels=3):
        super(NoBN_BED_CLASSIFIER, self).__init__()
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
    
# ______________________________________________________________ #
# ____________________    NO BATCH NORM     ____________________ #
# ______________________________________________________________ #
# class NoBN_BED_CLASSIFIER(nn.Module):
#     def __init__(self, num_classes, in_channels=3):
#         super(NoBN_BED_CLASSIFIER, self).__init__()
#         self.in_channels = in_channels
#         self.last_channels = 64
#         self.num_classes = num_classes
                
#         # CNNBlock 224x224
#         self.conv1 = nn.Conv2d(self.in_channels, 32, kernel_size=3, stride=1, padding=1)
#         self.relu1 = nn.ReLU()
#         self.dropout1 = nn.Dropout2d(p=0.3)
#         # CNNBlock 112x112
#         self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
#         self.conv2 = nn.Conv2d(32, 16, kernel_size=3, stride=1, padding=1)
#         self.relu2 = nn.ReLU() 
#         self.dropout2 =nn.Dropout2d(p=0.3) 

#         # CNNBlock 56x56
#         self.maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2) 
#         self.conv31 = nn.Conv2d(16, 16, kernel_size=1, stride=1, padding=0) 
#         self.relu31 = nn.ReLU() 

#         self.conv32 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1) 
#         self.relu32 = nn.ReLU() 

#         self.conv33 = nn.Conv2d(32, 32, kernel_size=1, stride=1, padding=0) 
#         self.relu33 = nn.ReLU() 

#         self.conv34 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1) 
#         self.relu34 = nn.ReLU() 

#         # CNNBlock 28x28
#         self.maxpool4 = nn.MaxPool2d(kernel_size=2, stride=2) 
#         self.conv41 = nn.Conv2d(64, 32, kernel_size=1, stride=1, padding=0) 
#         self.relu41 = nn.ReLU() 

#         self.conv42 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1) 
#         self.relu42 = nn.ReLU() 

#         self.conv43 = nn.Conv2d(64, 32, kernel_size=1, stride=1, padding=0) 
#         self.relu43 = nn.ReLU() 

#         self.conv44 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1) 
#         self.relu44 = nn.ReLU() 

#         self.conv45 = nn.Conv2d(64, 32, kernel_size=1, stride=1, padding=0) 
#         self.relu45 = nn.ReLU() 

#         self.conv46 = nn.Conv2d(32, self.last_channels, kernel_size=3, stride=1, padding=1) 
#         self.relu46 = nn.ReLU() 

#         # Output One Head, 2 Neurons
#         self.avgpool5 = nn.AdaptiveAvgPool2d((1, 1)) 
#         self.flatten5 = nn.Flatten(start_dim=1) 
#         self.dropout5 = nn.Dropout(p=0.2) 
#         self.linear51 = nn.Linear(in_features=self.last_channels, out_features=16) 
#         self.relu5 = nn.ReLU() 
#         self.linear52 = nn.Linear(in_features=16, out_features=2) 
    

#     def __initialize_weights__(self):
#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 nn.init.kaiming_normal_(m.weight, mode='fan_in',
#                     nonlinearity='relu'
#                 )
#                 if m.bias is not None:
#                         nn.init.constant_(m.bias, 0)
#             elif isinstance(m, nn.Linear):
#                 nn.init.normal_(m.weight, 0, 0.01)
#                 nn.init.constant_(m.bias, 0)


#     def forward(self, x):
#         # CNNBlock 224x224
#         x = self.conv1(x)
#         x = self.relu1(x)
#         x = self.dropout1(x)
#         # CNNBlock 112x112
#         x = self.maxpool2(x)
#         x = self.conv2(x)
#         x = self.relu2(x)
#         x = self.dropout2(x)
#         # CNNBlock 56x56
#         x = self.maxpool3(x)
#         x = self.conv31(x)
#         x = self.relu31(x)
#         x = self.conv32(x)
#         x = self.relu32(x)
#         x = self.conv33(x)
#         x = self.relu33(x)
#         x = self.conv34(x)
#         x = self.relu34(x)
#         # CNNBlock 28x28
#         x = self.maxpool4(x)
#         x = self.conv41(x)
#         x = self.relu41(x)
#         x = self.conv42(x)
#         x = self.relu42(x)
#         x = self.conv43(x)
#         x = self.relu43(x)
#         x = self.conv44(x)
#         x = self.relu44(x)
#         x = self.conv45(x)
#         x = self.relu45(x)
#         x = self.conv46(x)
#         x = self.relu46(x)
#         # Output One Head, 2 Neurons
#         x = self.avgpool5(x)
#         x = self.flatten5(x)
#         x = self.dropout5(x)
#         x = self.linear51(x)
#         x = self.relu5(x)
#         x = self.linear52(x)
#         return x