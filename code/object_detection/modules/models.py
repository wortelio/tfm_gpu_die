import collections
import torch
import torch.nn as nn 
import config

# ______________________________________________________________ #
# ____________________    ORIGINAL MODEL    ____________________ #
# ______________________________________________________________ #
class BED_DETECTOR(nn.Module):
    def __init__(self, in_channels=3):
        super(BED_DETECTOR, self).__init__()
        self.in_channels = in_channels
        self.B = config.B
        self.C = config.C
        
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
        x_out = self.model(x)
        x = x_out.permute(0, 2, 3, 1)
        class_softmax = torch.softmax(x[..., 10:12], dim=-1)
        x = torch.cat((torch.sigmoid(x[..., 0:10]), class_softmax), dim=-1)
        return x 
    
    
# ______________________________________________________________ #
# ____________________    ORIGINAL MODEL     ___________________ #
# ____________________  NO SIGMOID, SOFTMAX  ___________________ #
# ______________________________________________________________ #
class Permute(nn.Module):
    def __init__(self):
        super(Permute, self).__init__()

    def forward(self, x):
        return x.permute(0, 2, 3, 1)
    
class SIMPLE_BED_DETECTOR(nn.Module):
    def __init__(self, in_channels=3):
        super(SIMPLE_BED_DETECTOR, self).__init__()
        self.in_channels = in_channels
        self.B = config.B
        self.C = config.C
        
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
                    
                    #("permute", Permute()),
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
    
    
# ______________________________________________________________ #
# ____________________       SVD MODEL      ____________________ #
# ____________________  NO SIGMOID, SOFTMAX  ___________________ #
# ______________________________________________________________ #
class SVD_BED_DETECTOR(nn.Module):
    def __init__(self, in_channels=3):
        super(SVD_BED_DETECTOR, self).__init__()
        self.in_channels = in_channels
        self.B = config.B
        self.C = config.C
        
        self.model = self.__create_BED__()
        
    def __create_BED__(self):
        BED_model = nn.Sequential(

            collections.OrderedDict(
                [
            # Conv2d [in_channels, out_channels, kernel_size, stride, padding, bias]

                    # CNNBlock 224x224
                    ("conv1", nn.Sequential(
                         nn.Conv2d(self.in_channels, 5, kernel_size=(3,1), stride=(1,1), padding=(1,0),  bias=False),
                         nn.Conv2d(5, 32, kernel_size=(1,3), stride=(1,1), padding=(0,1),  bias=False)),),
                    ("bn1", nn.BatchNorm2d(32)),
                    ("relu1", nn.ReLU()),
                    ("dropout1", nn.Dropout2d(p=0.3)),
        
                    # CNNBlock 112x112
                    ("maxpool2", nn.MaxPool2d(kernel_size=2, stride=2)),
                    ("conv2", nn.Sequential(
                         nn.Conv2d(32, 6, kernel_size=(3,1), stride=(1,1), padding=(1,0),  bias=False),
                         nn.Conv2d(6, 16, kernel_size=(1,3), stride=(1,1), padding=(0,1),  bias=False)),),
                    ("bn2", nn.BatchNorm2d(16)),
                    ("relu2", nn.ReLU()),
                    ("dropout2",nn.Dropout2d(p=0.3)),
        
                    # CNNBlock 56x56
                    ("maxpool3", nn.MaxPool2d(kernel_size=2, stride=2)),
                    ("conv31", nn.Sequential(
                         nn.Conv2d(16, 3, kernel_size=(1,1), stride=(1,1),  bias=False),
                         nn.Conv2d(3, 16, kernel_size=(1,1), stride=(1,1),  bias=False)),),
                    ("bn31", nn.BatchNorm2d(16)),
                    ("relu31", nn.ReLU()),
                    
                    ("conv32", nn.Sequential(
                         nn.Conv2d(16, 16, kernel_size=(3,1), stride=(1,1), padding=(1,0),  bias=False),
                         nn.Conv2d(16, 32, kernel_size=(1,3), stride=(1,1), padding=(0,1),  bias=False)),),
                    ("bn32", nn.BatchNorm2d(32)),
                    ("relu32", nn.ReLU()),
                    
                    ("conv33", nn.Conv2d(32, 32, kernel_size=1, stride=1, padding=0,  bias=False)),
                    ("bn33", nn.BatchNorm2d(32)),
                    ("relu33", nn.ReLU()),
                    
                    ("conv34", nn.Sequential(
                         nn.Conv2d(32, 51, kernel_size=(3,1), stride=(1,1), padding=(1,0),  bias=False),
                         nn.Conv2d(51, 64, kernel_size=(1,3), stride=(1,1), padding=(0,1),  bias=False)),),
                    ("bn34", nn.BatchNorm2d(64)),
                    ("relu34", nn.ReLU()),
        
                    # CNNBlock 28x28
                    ("maxpool4", nn.MaxPool2d(kernel_size=2, stride=2)),
                    ("conv41", nn.Conv2d(64, 32, kernel_size=1, stride=1, padding=0,  bias=False)),
                    ("bn41", nn.BatchNorm2d(32)),
                    ("relu41", nn.ReLU()),
                    
                    ("conv42", nn.Sequential(
                         nn.Conv2d(32, 38, kernel_size=(3,1), stride=(1,1), padding=(1,0),  bias=False),
                         nn.Conv2d(38, 64, kernel_size=(1,3), stride=(1,1), padding=(0,1),  bias=False)),),
                    ("bn42", nn.BatchNorm2d(64)),
                    ("relu42", nn.ReLU()),
                    
                    ("conv43", nn.Conv2d(64, 32, kernel_size=1, stride=1, padding=0,  bias=False)),
                    ("bn43", nn.BatchNorm2d(32)),
                    ("relu43", nn.ReLU()),
                    
                    ("conv44", nn.Sequential(
                         nn.Conv2d(32, 32, kernel_size=(3,1), stride=(1,1), padding=(1,0),  bias=False),
                         nn.Conv2d(32, 64, kernel_size=(1,3), stride=(1,1), padding=(0,1),  bias=False)),),
                    ("bn44", nn.BatchNorm2d(64)),
                    ("relu44", nn.ReLU()),
                    
                    ("conv45", nn.Sequential(
                         nn.Conv2d(64, 17, kernel_size=(1,1), stride=(1,1),  bias=False),
                         nn.Conv2d(17, 32, kernel_size=(1,1), stride=(1,1),  bias=False)),),
                    ("bn45", nn.BatchNorm2d(32)),
                    ("relu45", nn.ReLU()),
                    
                    ("conv46", nn.Sequential(
                         nn.Conv2d(32, 32, kernel_size=(3,1), stride=(1,1), padding=(1,0),  bias=False),
                         nn.Conv2d(32, 64, kernel_size=(1,3), stride=(1,1), padding=(0,1),  bias=False)),),
                    ("bn46", nn.BatchNorm2d(64)),
                    ("relu46", nn.ReLU()),
                    
                    # CNNBlock 14x14
                    ("maxpool5", nn.MaxPool2d(kernel_size=2, stride=2)), 
                    ("conv51", nn.Sequential(
                         nn.Conv2d(64, 14, kernel_size=(1,1), stride=(1,1),  bias=False),
                         nn.Conv2d(14, 32, kernel_size=(1,1), stride=(1,1),  bias=False)),),
                    ("bn51", nn.BatchNorm2d(32)),
                    ("relu51", nn.ReLU()),

                    ("conv52", nn.Sequential(
                         nn.Conv2d(32, 51, kernel_size=(3,1), stride=(1,1), padding=(1,0),  bias=False),
                         nn.Conv2d(51, 64, kernel_size=(1,3), stride=(1,1), padding=(0,1),  bias=False)),),
                    ("bn52", nn.BatchNorm2d(64)),
                    ("relu52", nn.ReLU()),

                    ("conv53", nn.Conv2d(64, 32, kernel_size=1, stride=1, padding=0,  bias=False)),
                    ("bn53", nn.BatchNorm2d(32)),
                    ("relu53", nn.ReLU()),

                    ("conv54", nn.Sequential(
                         nn.Conv2d(32, 57, kernel_size=(3,1), stride=(1,1), padding=(1,0),  bias=False),
                         nn.Conv2d(57, 64, kernel_size=(1,3), stride=(1,1), padding=(0,1),  bias=False)),),
                    ("bn54", nn.BatchNorm2d(64)),
                    ("relu54", nn.ReLU()),

                    ("conv55", nn.Sequential(
                         nn.Conv2d(64, 67, kernel_size=(3,1), stride=(1,1), padding=(1,0),  bias=False),
                         nn.Conv2d(67, 64, kernel_size=(1,3), stride=(1,1), padding=(0,1),  bias=False)),),
                    ("bn55", nn.BatchNorm2d(64)),
                    ("relu55", nn.ReLU()),

                    ("conv56", nn.Sequential(
                         nn.Conv2d(64, 86, kernel_size=(3,1), stride=(1,1), padding=(1,0),  bias=False),
                         nn.Conv2d(86, 64, kernel_size=(1,3), stride=(1,1), padding=(0,1),  bias=False)),),
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


# ______________________________________________________________ #
# ___________ PRUNED_AFTER_SVD_BED_DETECTOR MODEL ______________ #
# __________________  NO SIGMOID, SOFTMAX  _____________________ #
# ______________________________________________________________ #
class PRUNED_AFTER_SVD_BED_DETECTOR(nn.Module):
    def __init__(self, in_channels=3):
        super(PRUNED_AFTER_SVD_BED_DETECTOR, self).__init__()
        self.in_channels = in_channels
        self.B = config.B
        self.C = config.C
        
        self.model = self.__create_BED__()
        
    def __create_BED__(self):
        BED_model = nn.Sequential(

            collections.OrderedDict(
                [
            # Conv2d [in_channels, out_channels, kernel_size, stride, padding, bias]

                    # CNNBlock 224x224
                    ("conv1", nn.Sequential(
                         nn.Conv2d(self.in_channels, 5, kernel_size=(3,1), stride=(1,1), padding=(1,0),  bias=False),
                         nn.Conv2d(5, 32, kernel_size=(1,3), stride=(1,1), padding=(0,1),  bias=False)),),
                    ("bn1", nn.BatchNorm2d(32)),
                    ("relu1", nn.ReLU()),
                    ("dropout1", nn.Dropout2d(p=0.3)),
        
                    # CNNBlock 112x112
                    ("maxpool2", nn.MaxPool2d(kernel_size=2, stride=2)),
                    ("conv2", nn.Sequential(
                         nn.Conv2d(32, 4, kernel_size=(3,1), stride=(1,1), padding=(1,0),  bias=False),
                         nn.Conv2d(4, 11, kernel_size=(1,3), stride=(1,1), padding=(0,1),  bias=False)),),
                    ("bn2", nn.BatchNorm2d(11)),
                    ("relu2", nn.ReLU()),
                    ("dropout2",nn.Dropout2d(p=0.3)),
        
                    # CNNBlock 56x56
                    ("maxpool3", nn.MaxPool2d(kernel_size=2, stride=2)),
                    ("conv31", nn.Sequential(
                         nn.Conv2d(11, 3, kernel_size=(1,1), stride=(1,1),  bias=False),
                         nn.Conv2d(3, 14, kernel_size=(1,1), stride=(1,1),  bias=False)),),
                    ("bn31", nn.BatchNorm2d(14)),
                    ("relu31", nn.ReLU()),
                    
                    ("conv32", nn.Sequential(
                         nn.Conv2d(14, 14, kernel_size=(3,1), stride=(1,1), padding=(1,0),  bias=False),
                         nn.Conv2d(14, 25, kernel_size=(1,3), stride=(1,1), padding=(0,1),  bias=False)),),
                    ("bn32", nn.BatchNorm2d(25)),
                    ("relu32", nn.ReLU()),
                    
                    ("conv33", nn.Conv2d(25, 22, kernel_size=1, stride=1, padding=0,  bias=False)),
                    ("bn33", nn.BatchNorm2d(22)),
                    ("relu33", nn.ReLU()),
                    
                    ("conv34", nn.Sequential(
                         nn.Conv2d(22, 30, kernel_size=(3,1), stride=(1,1), padding=(1,0),  bias=False),
                         nn.Conv2d(30, 64, kernel_size=(1,3), stride=(1,1), padding=(0,1),  bias=False)),),
                    ("bn34", nn.BatchNorm2d(64)),
                    ("relu34", nn.ReLU()),
        
                    # CNNBlock 28x28
                    ("maxpool4", nn.MaxPool2d(kernel_size=2, stride=2)),
                    ("conv41", nn.Conv2d(64, 28, kernel_size=1, stride=1, padding=0,  bias=False)),
                    ("bn41", nn.BatchNorm2d(28)),
                    ("relu41", nn.ReLU()),
                    
                    ("conv42", nn.Sequential(
                         nn.Conv2d(28, 38, kernel_size=(3,1), stride=(1,1), padding=(1,0),  bias=False),
                         nn.Conv2d(38, 64, kernel_size=(1,3), stride=(1,1), padding=(0,1),  bias=False)),),
                    ("bn42", nn.BatchNorm2d(64)),
                    ("relu42", nn.ReLU()),
                    
                    ("conv43", nn.Conv2d(64, 28, kernel_size=1, stride=1, padding=0,  bias=False)),
                    ("bn43", nn.BatchNorm2d(28)),
                    ("relu43", nn.ReLU()),
                    
                    ("conv44", nn.Sequential(
                         nn.Conv2d(28, 25, kernel_size=(3,1), stride=(1,1), padding=(1,0),  bias=False),
                         nn.Conv2d(25, 57, kernel_size=(1,3), stride=(1,1), padding=(0,1),  bias=False)),),
                    ("bn44", nn.BatchNorm2d(57)),
                    ("relu44", nn.ReLU()),
                    
                    ("conv45", nn.Sequential(
                         nn.Conv2d(57, 15, kernel_size=(1,1), stride=(1,1),  bias=False),
                         nn.Conv2d(15, 28, kernel_size=(1,1), stride=(1,1),  bias=False)),),
                    ("bn45", nn.BatchNorm2d(28)),
                    ("relu45", nn.ReLU()),
                    
                    ("conv46", nn.Sequential(
                         nn.Conv2d(28, 28, kernel_size=(3,1), stride=(1,1), padding=(1,0),  bias=False),
                         nn.Conv2d(28, 57, kernel_size=(1,3), stride=(1,1), padding=(0,1),  bias=False)),),
                    ("bn46", nn.BatchNorm2d(57)),
                    ("relu46", nn.ReLU()),
                    
                    # CNNBlock 14x14
                    ("maxpool5", nn.MaxPool2d(kernel_size=2, stride=2)), 
                    ("conv51", nn.Sequential(
                         nn.Conv2d(57, 14, kernel_size=(1,1), stride=(1,1),  bias=False),
                         nn.Conv2d(14, 32, kernel_size=(1,1), stride=(1,1),  bias=False)),),
                    ("bn51", nn.BatchNorm2d(32)),
                    ("relu51", nn.ReLU()),

                    ("conv52", nn.Sequential(
                         nn.Conv2d(32, 40, kernel_size=(3,1), stride=(1,1), padding=(1,0),  bias=False),
                         nn.Conv2d(40, 64, kernel_size=(1,3), stride=(1,1), padding=(0,1),  bias=False)),),
                    ("bn52", nn.BatchNorm2d(64)),
                    ("relu52", nn.ReLU()),

                    ("conv53", nn.Conv2d(64, 32, kernel_size=1, stride=1, padding=0,  bias=False)),
                    ("bn53", nn.BatchNorm2d(32)),
                    ("relu53", nn.ReLU()),

                    ("conv54", nn.Sequential(
                         nn.Conv2d(32, 45, kernel_size=(3,1), stride=(1,1), padding=(1,0),  bias=False),
                         nn.Conv2d(45, 64, kernel_size=(1,3), stride=(1,1), padding=(0,1),  bias=False)),),
                    ("bn54", nn.BatchNorm2d(64)),
                    ("relu54", nn.ReLU()),

                    ("conv55", nn.Sequential(
                         nn.Conv2d(64, 67, kernel_size=(3,1), stride=(1,1), padding=(1,0),  bias=False),
                         nn.Conv2d(67, 64, kernel_size=(1,3), stride=(1,1), padding=(0,1),  bias=False)),),
                    ("bn55", nn.BatchNorm2d(64)),
                    ("relu55", nn.ReLU()),

                    ("conv56", nn.Sequential(
                         nn.Conv2d(64, 77, kernel_size=(3,1), stride=(1,1), padding=(1,0),  bias=False),
                         nn.Conv2d(77, 64, kernel_size=(1,3), stride=(1,1), padding=(0,1),  bias=False)),),
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



# ______________________________________________________________ #
# _________________          SVD MODEL         _________________ #
# _________________  WITH SIGMOID and SOFTMAX  _________________ #
# ______________________________________________________________ #
# class SVD_BED_DETECTOR(nn.Module):
#     def __init__(self, in_channels=3):
#         super(SVD_BED_DETECTOR, self).__init__()
#         self.in_channels = in_channels
#         self.B = config.B
#         self.C = config.C
        
#         self.model = self.__create_BED__()
        
#     def __create_BED__(self):
#         BED_model = nn.Sequential(

#             collections.OrderedDict(
#                 [
#             # Conv2d [in_channels, out_channels, kernel_size, stride, padding, bias]

#                     # CNNBlock 224x224
#                     ("conv1", nn.Sequential(
#                          nn.Conv2d(self.in_channels, 5, kernel_size=(3,1), stride=(1,1), padding=(1,0),  bias=False),
#                          nn.Conv2d(5, 32, kernel_size=(1,3), stride=(1,1), padding=(0,1),  bias=False)),),
#                     ("bn1", nn.BatchNorm2d(32)),
#                     ("relu1", nn.ReLU()),
#                     ("dropout1", nn.Dropout2d(p=0.3)),
        
#                     # CNNBlock 112x112
#                     ("maxpool2", nn.MaxPool2d(kernel_size=2, stride=2)),
#                     ("conv2", nn.Sequential(
#                          nn.Conv2d(32, 9, kernel_size=(3,1), stride=(1,1), padding=(1,0),  bias=False),
#                          nn.Conv2d(9, 16, kernel_size=(1,3), stride=(1,1), padding=(0,1),  bias=False)),),
#                     ("bn2", nn.BatchNorm2d(16)),
#                     ("relu2", nn.ReLU()),
#                     ("dropout2",nn.Dropout2d(p=0.3)),
        
#                     # CNNBlock 56x56
#                     ("maxpool3", nn.MaxPool2d(kernel_size=2, stride=2)),
#                     ("conv31", nn.Sequential(
#                          nn.Conv2d(16, 3, kernel_size=(1,1), stride=(1,1),  bias=False),
#                          nn.Conv2d(3, 16, kernel_size=(1,1), stride=(1,1),  bias=False)),),
#                     ("bn31", nn.BatchNorm2d(16)),
#                     ("relu31", nn.ReLU()),
                    
#                     ("conv32", nn.Sequential(
#                          nn.Conv2d(16, 28, kernel_size=(3,1), stride=(1,1), padding=(1,0),  bias=False),
#                          nn.Conv2d(28, 32, kernel_size=(1,3), stride=(1,1), padding=(0,1),  bias=False)),),
#                     ("bn32", nn.BatchNorm2d(32)),
#                     ("relu32", nn.ReLU()),
                    
#                     ("conv33", nn.Sequential(
#                          nn.Conv2d(32, 14, kernel_size=(1,1), stride=(1,1),  bias=False),
#                          nn.Conv2d(14, 32, kernel_size=(1,1), stride=(1,1),  bias=False)),),
#                     ("bn33", nn.BatchNorm2d(32)),
#                     ("relu33", nn.ReLU()),
                    
#                     ("conv34", nn.Sequential(
#                          nn.Conv2d(32, 38, kernel_size=(3,1), stride=(1,1), padding=(1,0),  bias=False),
#                          nn.Conv2d(38, 64, kernel_size=(1,3), stride=(1,1), padding=(0,1),  bias=False)),),
#                     ("bn34", nn.BatchNorm2d(64)),
#                     ("relu34", nn.ReLU()),
        
#                     # CNNBlock 28x28
#                     ("maxpool4", nn.MaxPool2d(kernel_size=2, stride=2)),
#                     ("conv41", nn.Conv2d(64, 32, kernel_size=1, stride=1, padding=0,  bias=False)),
#                     ("bn41", nn.BatchNorm2d(32)),
#                     ("relu41", nn.ReLU()),
                    
#                     ("conv42", nn.Sequential(
#                          nn.Conv2d(32, 38, kernel_size=(3,1), stride=(1,1), padding=(1,0),  bias=False),
#                          nn.Conv2d(38, 64, kernel_size=(1,3), stride=(1,1), padding=(0,1),  bias=False)),),
#                     ("bn42", nn.BatchNorm2d(64)),
#                     ("relu42", nn.ReLU()),
                    
#                     ("conv43", nn.Conv2d(64, 32, kernel_size=1, stride=1, padding=0,  bias=False)),
#                     ("bn43", nn.BatchNorm2d(32)),
#                     ("relu43", nn.ReLU()),
                    
#                     ("conv44", nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1,  bias=False)),
#                     ("bn44", nn.BatchNorm2d(64)),
#                     ("relu44", nn.ReLU()),
                    
#                     ("conv45", nn.Conv2d(64, 32, kernel_size=1, stride=1, padding=0,  bias=False)),
#                     ("bn45", nn.BatchNorm2d(32)),
#                     ("relu45", nn.ReLU()),
                    
#                     ("conv46", nn.Sequential(
#                          nn.Conv2d(32, 44, kernel_size=(3,1), stride=(1,1), padding=(1,0),  bias=False),
#                          nn.Conv2d(44, 64, kernel_size=(1,3), stride=(1,1), padding=(0,1),  bias=False)),),
#                     ("bn46", nn.BatchNorm2d(64)),
#                     ("relu46", nn.ReLU()),
                    
#                     # CNNBlock 14x14
#                     ("maxpool5", nn.MaxPool2d(kernel_size=2, stride=2)), 
#                     ("conv51", nn.Conv2d(64, 32, kernel_size=1, stride=1, padding=0,  bias=False)),
#                     ("bn51", nn.BatchNorm2d(32)),
#                     ("relu51", nn.ReLU()),

#                     ("conv52", nn.Sequential(
#                          nn.Conv2d(32, 44, kernel_size=(3,1), stride=(1,1), padding=(1,0),  bias=False),
#                          nn.Conv2d(44, 64, kernel_size=(1,3), stride=(1,1), padding=(0,1),  bias=False)),),
#                     ("bn52", nn.BatchNorm2d(64)),
#                     ("relu52", nn.ReLU()),

#                     ("conv53", nn.Conv2d(64, 32, kernel_size=1, stride=1, padding=0,  bias=False)),
#                     ("bn53", nn.BatchNorm2d(32)),
#                     ("relu53", nn.ReLU()),

#                     ("conv54", nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1,  bias=False)),
#                     ("bn54", nn.BatchNorm2d(64)),
#                     ("relu54", nn.ReLU()),

#                     ("conv55", nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1,  bias=False)),
#                     ("bn55", nn.BatchNorm2d(64)),
#                     ("relu55", nn.ReLU()),

#                     ("conv56", nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1,  bias=False)),
#                     ("bn56", nn.BatchNorm2d(64)),
#                     ("relu56", nn.ReLU()),
            
#                     # CNNBlock 7x7
#                     ("maxpool6", nn.MaxPool2d(kernel_size=2, stride=2)),
#                     ("conv61", nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1,  bias=False)),
#                     ("bn61", nn.BatchNorm2d(64)),
#                     ("relu61", nn.ReLU()),
                    
#                     ("conv62", nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1,  bias=False)),
#                     ("bn62", nn.BatchNorm2d(64)),
#                     ("relu62", nn.ReLU()),

#                     # CNNBlock Out
#                     ("conv71", nn.Conv2d(64, 64, kernel_size=1, stride=1, padding=0,  bias=False)),
#                     ("bn71", nn.BatchNorm2d(64)),
#                     ("relu71", nn.ReLU()),

#                     ("conv72", nn.Conv2d(64, 16, kernel_size=1, stride=1, padding=0,  bias=False)),
#                     ("bn72", nn.BatchNorm2d(16)),
#                     ("relu72", nn.ReLU()),

#                     ("conv73", nn.Conv2d(16, 16, kernel_size=1, stride=1, padding=0,  bias=False)),
#                     ("bn73", nn.BatchNorm2d(16)),
#                     ("relu73", nn.ReLU()),

#                     ("conv74", nn.Conv2d(16, self.B*5 + self.C, kernel_size=1, stride=1, padding=0)),
#                 ]
#             )
#         )
#         return BED_model
    

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
#         x_out = self.model(x)
#         x = x_out.permute(0, 2, 3, 1)
#         class_softmax = torch.softmax(x[..., 10:12], dim=-1)
#         x = torch.cat((torch.sigmoid(x[..., 0:10]), class_softmax), dim=-1)
#         return x 