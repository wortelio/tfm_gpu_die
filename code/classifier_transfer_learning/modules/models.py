import torch
import torch.nn as nn 
from torchvision.models import mobilenet_v3_small
from torchvision.models import shufflenet_v2_x0_5


class PRETRAINED_MODEL(nn.Module):
    def __init__(self, base_model, last_channels, num_classes, in_channels=3):
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.last_channels = last_channels
        
        self.base_model = nn.Sequential(*list(base_model.children())[:-1])

        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.linear = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(in_features=self.last_channels, out_features=32),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(in_features=32, out_features=self.num_classes)
        )
    
    def forward(self, x):
        x = self.base_model(x)
        x = self.pool(x)
        x = torch.flatten(x, start_dim=1)
        x = self.linear(x)
        return x

    
def setup_model(model_name, num_classes, device):
    if model_name == "MOBILENETV3":
        print("Using MOBILENETV3 Classifier")

        base_model = mobilenet_v3_small(weights='IMAGENET1K_V1')
        print(list(base_model.children())[-1])
        for name, layer in base_model.named_modules():
            if isinstance(layer, nn.Linear):
                last_channels = layer.in_features
                print(f'Model Pretrained has {last_channels} in features in last layer')
                break
        for param in base_model.parameters():
            param.requires_grad = False

        model = PRETRAINED_MODEL(base_model, last_channels=last_channels, num_classes=num_classes).to(device)  
        return model

    if model_name == "SHUFFLENET":
        print("Using SHUFFLE Classifier")

        base_model = shufflenet_v2_x0_5(weights='IMAGENET1K_V1')
        print(list(base_model.children())[-1])
        for name, layer in base_model.named_modules():
            if isinstance(layer, nn.Linear):
                last_channels = layer.in_features
                print(f'Model Pretrained has {last_channels} in features in last layer')
        for param in base_model.parameters():
            param.requires_grad = False

        model = PRETRAINED_MODEL(base_model, last_channels, num_classes=num_classes).to(device)   

        return model
    
    else:
        print("Wrong Model")
        raise SystemExit("Wrong Model")
    