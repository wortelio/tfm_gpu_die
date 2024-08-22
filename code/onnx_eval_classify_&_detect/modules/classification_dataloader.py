import config
import modules.classification_dataset_dfire as dataset_dfire
import torch
from torch.utils.data import DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2

def get_val_loader(shuffle):
    val_transform = A.Compose([
        A.Resize(config.IMG_H, config.IMG_W, p=1),
        ToTensorV2(p=1),
        ]
    )
    
    print("\nTEST DFire dataset")
    val_dfire_dataset = dataset_dfire.DFireDataset(
        img_h = config.IMG_H,
        img_w = config.IMG_W,
        img_dir = config.DFIRE_VAL_IMGS_DIR,
        label_dir = config.DFIRE_VAL_LABELS_DIR,
        num_classes = config.N_CLASSES,
        ds_len = config.DS_LEN,
        transform=val_transform)
    print(f'\nTest dataset len: {len(val_dfire_dataset)}')
    
    
    val_loader = DataLoader(
        dataset=val_dfire_dataset,
        batch_size=config.BATCH_SIZE,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY,
        shuffle=shuffle,
        drop_last=True)
    
    return val_loader

