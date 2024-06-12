import sys
import logging

import config
from dataset import DFireDataset
from model import BED_CLASSIFIER
from loss import BCE_LOSS
from callbacks import lr_monitor

import lightning as L
import numpy as np
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader
from torch.nn.utils import parameters_to_vector
from lightning.pytorch.loggers import TensorBoardLogger 

torch.set_float32_matmul_precision('medium')


#_________________________________________________#
#                   Logger                        #
#_________________________________________________#
log_path = 'results/'

logger = logging.getLogger("GonLogger")
logger.propagate = False
logger.setLevel(logging.INFO)
file_handler = logging.FileHandler(log_path + 'logfile.log')
formatter = logging.Formatter('%(message)s')
file_handler.setFormatter(formatter)

# add file handler to logger
logger.addHandler(file_handler)

logger.info('BED Classifier with Pytorch Lightning. Code Reestructured in modules.')
logger.info(f'Pytorch Lightning version: {L.__version__}')

#_________________________________________________#
#                   Model                         #
#_________________________________________________#
if config.MODEL == "BED":
    print("Using BED Classifier")
    logger.info("Using BED Classifier")
    model = BED_CLASSIFIER(device=config.DEVICE, 
                           smoke_weight=config.SMOKE_PRECISION_WEIGHT,
                           learning_rate=config.LEARNING_RATE,
                           weight_decay=config.WEIGHT_DECAY,
                           num_classes=config.N_CLASSES).to(config.DEVICE)
else:
    print("Wrong Model")
    logger.info("Wrong Model")
    raise SystemExit("Wrong Model")

# Check model shape
in_rand_np = np.random.rand(4, 3, config.IMG_H, config.IMG_W)
in_rand = torch.tensor(in_rand_np, dtype=torch.float32, device=config.DEVICE)
out_test = model(in_rand)
print(f'Model shape is {out_test}')
print(f'BED Model Arquitecture\n{model}')
logger.info(f'Model shape is {out_test}')
logger.info(f'BED Model Arquitecture\n{model}')

# MODEL PARAMETERS
n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f'\nTrainable parameters = {n_trainable}')
logger.info(f'\nTrainable parameters = {n_trainable}')

n_params = parameters_to_vector(model.parameters()).numel()
print(f'Total parameters = {n_params}\n')
logger.info(f'Total parameters = {n_params}\n')

#_________________________________________________#
#                   Dataset                       #
#_________________________________________________#
train_transform = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.OneOf([
            A.RandomBrightnessContrast(p=0.4),
            A.HueSaturationValue(hue_shift_limit=10, p=0.2),
            A.Blur(blur_limit=(3,3), p=0.3),
            A.CLAHE(clip_limit=2.0, p=0.3),
            A.RGBShift(r_shift_limit=15, g_shift_limit=15, b_shift_limit=15, p=0.5),
        ], p=0.9),
        A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=15, p=0.5),
        A.Resize(config.IMG_H, config.IMG_W, p=1),
        ToTensorV2(p=1),
    ]
)

# TRAIN DATASET
print("\nTRAIN DFIRE dataset")
logger.info("\nTRAIN DFIRE dataset")
train_dataset = DFireDataset(img_h = config.IMG_H,
                             img_w = config.IMG_W,
                             img_dir = config.TRAIN_IMG_DIR,
                             label_dir = config.TRAIN_LABEL_DIR,
                             num_classes = config.N_CLASSES,
                             ds_len = config.DS_LEN,
                             transform=train_transform)

print(f'Train dataset len: {len(train_dataset)}')
logger.info(f'Train dataset len: {len(train_dataset)}')


# VALIDATION DATASET
val_transform = A.Compose([
    A.Resize(config.IMG_H, config.IMG_W, p=1),
    ToTensorV2(p=1),
    ]
)

print("\nTEST DFire dataset")
logger.info("\nTEST DFire dataset")
val_dataset = DFireDataset(img_h = config.IMG_H,
                           img_w = config.IMG_W,
                           img_dir = config.VAL_IMG_DIR,
                           label_dir = config.VAL_LABEL_DIR,
                           num_classes = config.N_CLASSES,
                           ds_len = config.DS_LEN,
                           transform=val_transform)

print(f'Test dataset len: {len(val_dataset)}')
logger.info(f'Test dataset len: {len(val_dataset)}')


# LOADERS
train_loader = DataLoader(dataset=train_dataset,
                          batch_size=config.BATCH_SIZE,
                          num_workers=config.NUM_WORKERS,
                          pin_memory=config.PIN_MEMORY,
                          shuffle=True,
                          drop_last=True)

val_loader = DataLoader(dataset=val_dataset,
                        batch_size=config.BATCH_SIZE,
                        num_workers=config.NUM_WORKERS,
                        pin_memory=config.PIN_MEMORY,
                        shuffle=False,
                        drop_last=True)

tb_logger = TensorBoardLogger("tb_logs", name="bed_model_v0")
trainer = L.Trainer(logger=tb_logger, accelerator='gpu', devices=[0], max_epochs=config.EPOCHS, callbacks=[lr_monitor])

def main():
    trainer.fit(model, train_loader, val_loader)
    trainer.validate(model, val_loader)

if __name__ == "__main__":
    print("Starting script\n")
    logger.info("Starting script\n")
    main()
    