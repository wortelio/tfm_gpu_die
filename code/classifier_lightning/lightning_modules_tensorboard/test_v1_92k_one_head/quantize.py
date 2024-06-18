import sys
import logging
import os 

import config
from dataset import DFireDataset
from model import BED_CLASSIFIER

import lightning as L
from pytorch_lightning.utilities.model_summary import ModelSummary
import numpy as np
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader

from neural_compressor.quantization import fit as fit
from neural_compressor.config import PostTrainingQuantConfig, TuningCriterion, AccuracyCriterion

torch.set_float32_matmul_precision('medium')

#_________________________________________________#
#          Model: Load Checkpoint                 #
#_________________________________________________#
def load_model(checkpoint_path):
    model = BED_CLASSIFIER.load_from_checkpoint(
        checkpoint_path=checkpoint_path,
        device=config.DEVICE, 
        smoke_weight=config.SMOKE_PRECISION_WEIGHT,
        learning_rate=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY,
        num_classes=config.N_CLASSES).to(config.DEVICE)
    model.eval()
    return model

#_________________________________________________#
#                   Dataset                       #
#_________________________________________________#
# VALIDATION DATASET
val_transform = A.Compose([
    A.Resize(config.IMG_H, config.IMG_W, p=1),
    ToTensorV2(p=1),
    ]
)

print("\nTEST DFire dataset")
val_dataset = DFireDataset(img_h = config.IMG_H,
                           img_w = config.IMG_W,
                           img_dir = config.VAL_IMG_DIR,
                           label_dir = config.VAL_LABEL_DIR,
                           num_classes = config.N_CLASSES,
                           ds_len = config.QUANT_DS_LEN,
                           transform=val_transform)

print(f'Test dataset len: {len(val_dataset)}')

# LOADER
val_loader = DataLoader(dataset=val_dataset,
                        batch_size=config.BATCH_SIZE,
                        num_workers=config.NUM_WORKERS,
                        pin_memory=config.PIN_MEMORY,
                        shuffle=False,
                        drop_last=True)

#_________________________________________________#
#        Quantization Functions                   #
#_________________________________________________#
def eval_func_for_nc(model_n, trainer_n):
    setattr(model, "model", model_n)
    result = trainer_n.validate(model=model, dataloaders=val_loader)
    return result[0]["val_smoke_f1"]


def eval_func(model):
    return eval_func_for_nc(model, trainer)

if __name__ == "__main__":
    checkpoint_path = "tb_logs/bed_model_v0_one_head/version_0/checkpoints/epoch=124-step=33625.ckpt"
    model = load_model(checkpoint_path)
    print(f"Model Loaded from checkpoint: {checkpoint_path}\n")
    print(ModelSummary(model, max_depth=-1))
    
    trainer = L.Trainer(logger=False, accelerator='gpu', devices=[0])
    
    accuracy_criterion = AccuracyCriterion(tolerable_loss=0.1)
    tuning_criterion = TuningCriterion(max_trials=600)
    conf = PostTrainingQuantConfig(
        approach="static", 
        backend="default", 
        tuning_criterion=tuning_criterion, 
        accuracy_criterion=accuracy_criterion
    )
    q_model = fit(model=model.model, conf=conf, calib_dataloader=val_loader, eval_func=eval_func)

    #q_model.save("./saved_model/")
    


    
 