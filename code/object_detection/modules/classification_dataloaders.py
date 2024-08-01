import config
import modules.classification_dataset_dfire as dataset_dfire
import modules.classification_dataset_fasdd as dataset_fasdd
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
    
    print("\nTEST FASDD UAV dataset")
    val_fasdd_uav_ds = dataset_fasdd.FASDDDataset(
        img_h=config.IMG_H, 
        img_w=config.IMG_W, 
        imgs_dir=config.FASDD_UAV_IMGS_DIR, 
        labels_file=config.FASDD_UAV_TEST_LABELS_FILE, 
        num_classes=config.N_CLASSES,
        ds_len=config.DS_LEN,
        transform=val_transform)
    print(f'\nTest FASDD UAV dataset len: {len(val_fasdd_uav_ds)}')
    
    print("\nTEST FASDD CV dataset")
    val_fasdd_cv_ds = dataset_fasdd.FASDDDataset(
        img_h=config.IMG_H, 
        img_w=config.IMG_W, 
        imgs_dir=config.FASDD_CV_IMGS_DIR, 
        labels_file=config.FASDD_CV_TEST_LABELS_FILE, 
        num_classes=config.N_CLASSES,
        ds_len=config.DS_LEN,
        transform=val_transform)
    print(f'\nTest FASDD CV dataset len: {len(val_fasdd_cv_ds)}')
    
    print("\nConcatenate Test DFire and FASDD UAV datasets")
    val_ds_concat = torch.utils.data.ConcatDataset((val_dfire_dataset, val_fasdd_uav_ds))
    print(f'Test dataset len: {len(val_ds_concat)}')
    print("Concatenate with FASDD CV dataset")
    val_ds = torch.utils.data.ConcatDataset((val_ds_concat, val_fasdd_cv_ds))
    print(f'Test dataset len: {len(val_ds)}')
    
    val_loader = DataLoader(
        dataset=val_ds,
        batch_size=config.BATCH_SIZE,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY,
        shuffle=shuffle,
        drop_last=True)
    
    return val_loader

def get_dfire_val_loader(shuffle):
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

def get_fasdd_uav_val_loader(shuffle):
    val_transform = A.Compose([
        A.Resize(config.IMG_H, config.IMG_W, p=1),
        ToTensorV2(p=1),
        ]
    )
    
    print("\nTEST FASDD UAV dataset")
    val_fasdd_uav_ds = dataset_fasdd.FASDDDataset(
        img_h=config.IMG_H, 
        img_w=config.IMG_W, 
        imgs_dir=config.FASDD_UAV_IMGS_DIR, 
        labels_file=config.FASDD_UAV_TEST_LABELS_FILE, 
        num_classes=config.N_CLASSES,
        ds_len=config.DS_LEN,
        transform=val_transform)
    print(f'\nTest FASDD UAV dataset len: {len(val_fasdd_uav_ds)}')
      
    val_loader = DataLoader(
        dataset=val_fasdd_uav_ds,
        batch_size=config.BATCH_SIZE,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY,
        shuffle=shuffle,
        drop_last=True)
    
    return val_loader

def get_fasdd_cv_val_loader():
    val_transform = A.Compose([
        A.Resize(config.IMG_H, config.IMG_W, p=1),
        ToTensorV2(p=1),
        ]
    )
    
    print("\nTEST FASDD CV dataset")
    val_fasdd_cv_ds = dataset_fasdd.FASDDDataset(
        img_h=config.IMG_H, 
        img_w=config.IMG_W, 
        imgs_dir=config.FASDD_CV_IMGS_DIR, 
        labels_file=config.FASDD_CV_TEST_LABELS_FILE, 
        num_classes=config.N_CLASSES,
        ds_len=config.DS_LEN,
        transform=val_transform)
    print(f'\nTest FASDD CV dataset len: {len(val_fasdd_cv_ds)}')
        
    val_loader = DataLoader(
        dataset=val_fasdd_cv_ds,
        batch_size=config.BATCH_SIZE,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY,
        shuffle=shuffle,
        drop_last=True)
    
    return val_loader


