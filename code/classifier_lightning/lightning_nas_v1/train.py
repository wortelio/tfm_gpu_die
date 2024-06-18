import argparse
import logging
import os
import sys
import time
import warnings

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

from IPython.utils import io

warnings.filterwarnings("ignore")  # Disable data logger warnings
logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)  # Disable GPU/TPU prints

torch.set_float32_matmul_precision('medium')

#_________________________________________________#
#                   Arguments                     #
#_________________________________________________#
def parse_args():
    parser = argparse.ArgumentParser(description="train bed")
    parser.add_argument(
        "--log_path", type=str, required=True, help="dir to place tensorboard logs from all trials"
    )
    parser.add_argument(
        "--conv10_channels", type=int, required=True, help="channels in conv 1"
    )
    parser.add_argument(
        "--conv20_channels", type=int, required=True, help="channels in conv 2"
    )
    parser.add_argument(
        "--conv31_channels", type=int, required=True, help="channels in conv 3, 1 stage"
    )
    parser.add_argument(
        "--conv32_channels", type=int, required=True, help="channels in conv 3, 2 stage"
    )
    parser.add_argument(
        "--conv33_channels", type=int, required=True, help="channels in conv 3, 3 stage"
    )
    parser.add_argument(
        "--conv34_channels", type=int, required=True, help="channels in conv 3, 4 stage"
    )
    parser.add_argument(
        "--conv41_channels", type=int, required=True, help="channels in conv 4, 1 stage"
    )
    parser.add_argument(
        "--conv42_channels", type=int, required=True, help="channels in conv 4, 2 stage"
    )
    parser.add_argument(
        "--conv43_channels", type=int, required=True, help="channels in conv 4, 3 stage"
    )
    parser.add_argument(
        "--conv44_channels", type=int, required=True, help="channels in conv 4, 4 stage"
    )
    parser.add_argument(
        "--conv45_channels", type=int, required=True, help="channels in conv 4, 5 stage"
    )
    parser.add_argument(
        "--last_channels", type=int, required=True, help="last channels before head"
    )
    parser.add_argument(
        "--head_features", type=int, required=True, help="head number of features"
    )
    parser.add_argument("--learning_rate", type=float, required=True, help="learning rate")
    parser.add_argument("--dropout", type=float, required=True, help="dropout probability")
    return parser.parse_args()

args = parse_args()


def run_training_job():

    #_________________________________________________#
    #                   Model                         #
    #_________________________________________________#
    if config.MODEL == "BED":
        print("Using BED Classifier")
        model = BED_CLASSIFIER(
            num_classes=config.N_CLASSES,    
            device=config.DEVICE, 
            smoke_weight=config.SMOKE_PRECISION_WEIGHT,
            learning_rate=args.learning_rate,
            weight_decay=config.WEIGHT_DECAY,
            conv10_channels=args.conv10_channels,
            conv20_channels=args.conv20_channels,
            conv31_channels=args.conv31_channels,
            conv32_channels=args.conv32_channels,
            conv33_channels=args.conv33_channels,
            conv34_channels=args.conv34_channels,
            conv41_channels=args.conv41_channels,
            conv42_channels=args.conv42_channels,
            conv43_channels=args.conv43_channels,
            conv44_channels=args.conv44_channels,
            conv45_channels=args.conv45_channels,
            last_channels=args.last_channels,
            head_features=args.head_features,
            dropout=args.dropout).to(config.DEVICE)
    else:
        print("Wrong Model")
        raise SystemExit("Wrong Model")
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
            #A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=15, p=0.5),
            A.Resize(config.IMG_H, config.IMG_W, p=1),
            ToTensorV2(p=1),
        ]
    )

    # TRAIN DATASET
    print("\nTRAIN DFIRE dataset")
    train_dataset = DFireDataset(
        img_h = config.IMG_H,
        img_w = config.IMG_W,
        img_dir = config.TRAIN_IMG_DIR,
        label_dir = config.TRAIN_LABEL_DIR,
        num_classes = config.N_CLASSES,
        ds_len = config.TRAIN_DS_LEN,
        transform=train_transform)

    print(f'Train dataset len: {len(train_dataset)}')

    # VALIDATION DATASET
    val_transform = A.Compose([
        A.Resize(config.IMG_H, config.IMG_W, p=1),
        ToTensorV2(p=1),
        ]
    )

    print("\nTEST DFire dataset")
    val_dataset = DFireDataset(
        img_h = config.IMG_H,
        img_w = config.IMG_W,
        img_dir = config.VAL_IMG_DIR,
        label_dir = config.VAL_LABEL_DIR,
        num_classes = config.N_CLASSES,
        ds_len = config.VAL_DS_LEN,
        transform=val_transform)

    print(f'Test dataset len: {len(val_dataset)}')

    # LOADERS
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=config.BATCH_SIZE,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY,
        shuffle=True,
        drop_last=True)

    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=config.BATCH_SIZE,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY,
        shuffle=False,
        drop_last=True)

    #_________________________________________________#
    #                   Trainer                       #
    #_________________________________________________#
    # Initialize a trainer (don't log anything since things get so slow...)
    trainer = L.Trainer(
        logger=False, 
        accelerator='gpu', 
        devices=[0], 
        max_epochs=config.EPOCHS, 
        default_root_dir=args.log_path,
    )
        #callbacks=[lr_monitor])

    logger = TensorBoardLogger(args.log_path)
    print(f"Logging to path: {args.log_path}.")

    start = time.time()
    trainer.fit(model, train_loader, val_loader)
    end = time.time()
    train_time = end - start
    logger.log_metrics({"train_time": end - start})
    
    with io.capture_output() as captured:
        val_out = trainer.validate(model, val_loader)[0]
        # val_out_trainer = trainer.validate(model, val_loader)
        # val_out = val_out_trainer[0]
        val_smoke_f1 = val_out['val_smoke_f1']
        val_fire_f1 = val_out['val_fire_f1']
        val_mean_f1 = (val_smoke_f1 + val_fire_f1) / 2
    logger.log_metrics({'val_mean_f1': val_mean_f1})
        
    num_params = trainer.model.num_params
    logger.log_metrics({"num_params": num_params})

    logger.save()

    # Print outputs
    print(f'train time: {train_time}, val mean f1: {val_mean_f1}, num params: {num_params}')

    
if __name__ == "__main__":
    print("Starting script\n")
    run_training_job()
    