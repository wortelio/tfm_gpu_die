import lightning as L
from torchmetrics.classification import BinaryAccuracy, BinaryPrecision, BinaryRecall, BinaryF1Score
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils import parameters_to_vector
from loss import BCE_LOSS
from torchvision.utils import make_grid
from config import (IMG_H, IMG_W)

class BED_CLASSIFIER(L.LightningModule):
    def __init__(self, num_classes, 
                 device, smoke_weight, learning_rate, weight_decay,
                 conv10_channels,
                 conv20_channels,
                 conv31_channels,
                 conv32_channels,
                 conv33_channels,
                 conv34_channels,
                 conv41_channels,
                 conv42_channels,
                 conv43_channels,
                 conv44_channels,
                 conv45_channels,
                 last_channels,
                 head_features,
                 dropout,
                 Pretrained=False,
                 in_channels=3):
        
        super().__init__()
        self.in_channels = in_channels
        
        self.conv10_channels = conv10_channels
        self.conv20_channels = conv20_channels
        self.conv31_channels = conv31_channels
        self.conv32_channels = conv32_channels
        self.conv33_channels = conv33_channels
        self.conv34_channels = conv34_channels
        self.conv41_channels = conv41_channels
        self.conv42_channels = conv42_channels
        self.conv43_channels = conv43_channels
        self.conv44_channels = conv44_channels
        self.conv45_channels = conv45_channels
        self.last_channels = last_channels
        self.head_features = head_features
        
        self.dropout = dropout
        
        self.num_classes = num_classes
        self.loss_fn = BCE_LOSS(device, smoke_weight)
        self.lr = learning_rate
        self.weight_decay = weight_decay
        

        # Metrics
        self.smoke_acc = BinaryAccuracy()
        self.smoke_pre = BinaryPrecision()
        self.smoke_rec = BinaryRecall()
        self.smoke_f1 = BinaryF1Score()
        self.fire_acc = BinaryAccuracy()
        self.fire_pre = BinaryPrecision()
        self.fire_rec = BinaryRecall()
        self.fire_f1 = BinaryF1Score()
        
        # Model Arquitecture and Initialization
        self.model = self.__create_BED__()
        
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.smoke = nn.Sequential(
            nn.Dropout(p=self.dropout),
            nn.Linear(in_features=self.last_channels, out_features=self.head_features),
            nn.ReLU(),
            nn.Linear(in_features=self.head_features, out_features=1)
        )
        self.fire = nn.Sequential(
            nn.Dropout(p=self.dropout),
            nn.Linear(in_features=self.last_channels, out_features=self.head_features),
            nn.ReLU(),
            nn.Linear(in_features=self.head_features, out_features=1)
        )
        
        self.num_params = parameters_to_vector(self.parameters()).numel()

        if Pretrained == False:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, mode='fan_in',
                        nonlinearity='relu'
                    )
                    #print("Initialize conv2d")
                    if m.bias is not None:
                            nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.Linear):
                    nn.init.normal_(m.weight, 0, 0.01)
                    nn.init.constant_(m.bias, 0)
                    #print("Initialize linear")
        
    def __create_BED__(self):
        BED_model = nn.Sequential(
            # Conv2d [in_channels, out_channels, kernel_size, stride, padding, bias]

            # CNNBlock 224x224
            nn.Conv2d(self.in_channels, self.conv10_channels, kernel_size=3, stride=1, padding=1,  bias=False),
            nn.BatchNorm2d(self.conv10_channels, affine=False),
            nn.ReLU(),
            nn.Dropout2d(p=0.3),

            # CNNBlock 112x112
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(self.conv10_channels, self.conv20_channels, kernel_size=3, stride=1, padding=1,  bias=False),
            nn.BatchNorm2d(self.conv20_channels, affine=False),
            nn.ReLU(),
            nn.Dropout2d(p=0.3),

            # CNNBlock 56x56
            nn.MaxPool2d(kernel_size=2, stride=2),
            # kernel = 1 in github
            nn.Conv2d(self.conv20_channels, self.conv31_channels, kernel_size=1, stride=1, padding=0,  bias=False),
            nn.BatchNorm2d(self.conv31_channels, affine=False),
            nn.ReLU(),
            
            nn.Conv2d(self.conv31_channels, self.conv32_channels, kernel_size=3, stride=1, padding=1,  bias=False),
            nn.BatchNorm2d(self.conv32_channels, affine=False),
            nn.ReLU(),
            
            # kernel = 1 in github
            nn.Conv2d(self.conv32_channels, self.conv33_channels, kernel_size=1, stride=1, padding=0,  bias=False),
            nn.BatchNorm2d(self.conv33_channels, affine=False),
            nn.ReLU(),
            
            nn.Conv2d(self.conv33_channels, self.conv34_channels, kernel_size=3, stride=1, padding=1,  bias=False),
            nn.BatchNorm2d(self.conv34_channels, affine=False),
            nn.ReLU(),

            # CNNBlock 28x28
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(self.conv34_channels, self.conv41_channels, kernel_size=1, stride=1, padding=0,  bias=False),
            nn.BatchNorm2d(self.conv41_channels, affine=False),
            nn.ReLU(),
            
            nn.Conv2d(self.conv41_channels, self.conv42_channels, kernel_size=3, stride=1, padding=1,  bias=False),
            nn.BatchNorm2d(self.conv42_channels, affine=False),
            nn.ReLU(),
            
            nn.Conv2d(self.conv42_channels, self.conv43_channels, kernel_size=1, stride=1, padding=0,  bias=False),
            nn.BatchNorm2d(self.conv43_channels, affine=False),
            nn.ReLU(),
            
            nn.Conv2d(self.conv43_channels, self.conv44_channels, kernel_size=3, stride=1, padding=1,  bias=False),
            nn.BatchNorm2d(self.conv44_channels, affine=False),
            nn.ReLU(),
            
            nn.Conv2d(self.conv44_channels, self.conv45_channels, kernel_size=1, stride=1, padding=0,  bias=False),
            nn.BatchNorm2d(self.conv45_channels, affine=False),
            nn.ReLU(),
            
            nn.Conv2d(self.conv45_channels, self.last_channels, kernel_size=3, stride=1, padding=1,  bias=False),
            nn.BatchNorm2d(self.last_channels, affine=False),
            nn.ReLU(),
        )
        return BED_model      

    def forward(self, x):
        x = self.model(x)
        x = self.pool(x)
        x = torch.flatten(x, start_dim=1)
        x = torch.cat((self.smoke(x), self.fire(x)), dim=-1)
        #x = torch.sigmoid(x) # BCE Logits
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        yhat = self.forward(x)
        loss = self.loss_fn(yhat, y)
        self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=False)
        # Grid of images to check data augmentation
        # if batch_idx % 200 == 0:
        #     pics = x[:16]
        #     grid = make_grid(pics.view(-1, 3, IMG_H, IMG_W))
        #     self.logger.experiment.add_image("bed_images", grid, self.global_step)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        yhat = self.forward(x)
        loss = self.loss_fn(yhat, y)
        dic_losses = self.loss_fn.get_last_losses()
        #yhat = torch.sigmoid(yhat)
        self.smoke_acc(yhat[..., 0], y[..., 0])
        self.smoke_pre(yhat[..., 0], y[..., 0])
        self.smoke_rec(yhat[..., 0], y[..., 0])
        self.smoke_f1(yhat[..., 0], y[..., 0])
        self.fire_acc(yhat[..., 1], y[..., 1])
        self.fire_pre(yhat[..., 1], y[..., 1])
        self.fire_rec(yhat[..., 1], y[..., 1])
        self.fire_f1(yhat[..., 1], y[..., 1])
        self.log_dict({
            'val_loss': loss,
            'val_smoke_loss': dic_losses['smoke_loss'],
            'val_fire_loss': dic_losses['fire_loss'], 
            'val_smoke_acc': self.smoke_acc,
            'val_smoke_pre': self.smoke_pre,
            'val_smoke_rec': self.smoke_rec,
            'val_smoke_f1': self.smoke_f1,
            'val_fire_acc': self.fire_acc,
            'val_fire_pre': self.fire_pre,
            'val_fire_rec': self.fire_rec,
            'val_fire_f1': self.fire_f1,
        }, on_step=False, on_epoch=True, prog_bar=False)
        return loss
    
    # Better to log in logger (tensorboard)
    def on_validation_epoch_end(self):
        # Log the learning rate.
        self.log('learning_rate', self.lr, on_step=False, on_epoch=True, prog_bar=False)

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)  
        # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 
        #                                                  mode='min',
        #                                                  factor=0.8, 
        #                                                  patience=2, 
        #                                                  threshold=0.001, 
        #                                                  threshold_mode='abs',
        #                                                  min_lr=1e-6)
        return optimizer
        # return {"optimizer": optimizer, 
        #         "lr_scheduler": {
        #             "scheduler": scheduler,
        #             "monitor": 'val_loss'
        #             }
        #        }