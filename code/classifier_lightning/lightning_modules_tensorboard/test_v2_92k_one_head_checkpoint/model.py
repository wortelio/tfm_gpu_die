import lightning as L
from torchmetrics.classification import BinaryAccuracy, BinaryPrecision, BinaryRecall, BinaryF1Score
import torch
import torch.nn as nn
import torch.optim as optim
from loss import BCE_LOSS
from torchvision.utils import make_grid
from config import (IMG_H, IMG_W)

class BED_CLASSIFIER(L.LightningModule):
    def __init__(self, num_classes, 
                 device, smoke_weight, learning_rate, weight_decay,
                 Pretrained=False,
                 in_channels=3):
        super().__init__()
        self.in_channels = in_channels
        self.last_channels = 64
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
            nn.Conv2d(self.in_channels, 32, kernel_size=3, stride=1, padding=1,  bias=False),
            nn.BatchNorm2d(32, affine=False),
            nn.ReLU(),
            nn.Dropout2d(p=0.3),

            # CNNBlock 112x112
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 16, kernel_size=3, stride=1, padding=1,  bias=False),
            nn.BatchNorm2d(16, affine=False),
            nn.ReLU(),
            nn.Dropout2d(p=0.3),

            # CNNBlock 56x56
            nn.MaxPool2d(kernel_size=2, stride=2),
            # kernel = 1 in github
            nn.Conv2d(16, 16, kernel_size=1, stride=1, padding=0,  bias=False),
            nn.BatchNorm2d(16, affine=False),
            nn.ReLU(),
            
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1,  bias=False),
            nn.BatchNorm2d(32, affine=False),
            nn.ReLU(),
            
            # kernel = 1 in github
            nn.Conv2d(32, 32, kernel_size=1, stride=1, padding=0,  bias=False),
            nn.BatchNorm2d(32, affine=False),
            nn.ReLU(),
            
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1,  bias=False),
            nn.BatchNorm2d(64, affine=False),
            nn.ReLU(),

            # CNNBlock 28x28
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 32, kernel_size=1, stride=1, padding=0,  bias=False),
            nn.BatchNorm2d(32, affine=False),
            nn.ReLU(),
            
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1,  bias=False),
            nn.BatchNorm2d(64, affine=False),
            nn.ReLU(),
            
            nn.Conv2d(64, 32, kernel_size=1, stride=1, padding=0,  bias=False),
            nn.BatchNorm2d(32, affine=False),
            nn.ReLU(),
            
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1,  bias=False),
            nn.BatchNorm2d(64, affine=False),
            nn.ReLU(),
            
            nn.Conv2d(64, 32, kernel_size=1, stride=1, padding=0,  bias=False),
            nn.BatchNorm2d(32, affine=False),
            nn.ReLU(),
            
            nn.Conv2d(32, self.last_channels, kernel_size=3, stride=1, padding=1,  bias=False),
            nn.BatchNorm2d(self.last_channels, affine=False),
            nn.ReLU(),

            # Output One Head, 2 Neurons
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(start_dim=1),
            nn.Dropout(p=0.2),
            nn.Linear(in_features=self.last_channels, out_features=16),
            nn.ReLU(),
            nn.Linear(in_features=16, out_features=2),
            
        )
        return BED_model

    def forward(self, x):
        x = self.model(x)
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        yhat = self.forward(x)
        loss = self.loss_fn(yhat, y)
        self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
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
        }, on_step=False, on_epoch=True, prog_bar=True)
        return loss
    
    # Better to log in logger (tensorboard)
    def on_validation_epoch_end(self):
        # Log the learning rate.
        self.log('learning_rate', self.lr, on_step=False, on_epoch=True, prog_bar=True)

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)  
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 
                                                         mode='min',
                                                         factor=0.8, 
                                                         patience=2, 
                                                         threshold=0.001, 
                                                         threshold_mode='abs',
                                                         min_lr=1e-6)
        # return optimizer
        return {"optimizer": optimizer, 
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": 'val_loss'
                    }
               }