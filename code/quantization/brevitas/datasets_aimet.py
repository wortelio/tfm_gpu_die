import config_aimet
import dataset_dfire
import dataset_fasdd
import torch
from torch.utils.data import DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2

def get_train_loader():
    train_transform = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.OneOf([
                A.RandomBrightnessContrast(p=0.4),
                A.HueSaturationValue(hue_shift_limit=10, p=0.2),
                A.Blur(blur_limit=(3,3), p=0.3),
                A.CLAHE(clip_limit=2.0, p=0.3),
                A.RGBShift(r_shift_limit=15, g_shift_limit=15, b_shift_limit=15, p=0.5),
            ], p=0.9),
            A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=10, p=0.2),
            A.Resize(config_aimet.IMG_H, config_aimet.IMG_W, p=1),
            #A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), p=1),
            ToTensorV2(p=1),
        ]
    )
    
    # TRAIN DATASET
    print("\nTRAIN DFIRE dataset")
    train_dfire_dataset = dataset_dfire.DFireDataset(
        img_h = config_aimet.IMG_H,
        img_w = config_aimet.IMG_W,
        img_dir = config_aimet.TRAIN_IMG_DIR,
        label_dir = config_aimet.TRAIN_LABEL_DIR,
        num_classes=config_aimet.N_CLASSES,
        ds_len = config_aimet.DS_LEN,
        transform=train_transform)
    print(f'\nTrain DFire dataset len: {len(train_dfire_dataset)}')

    print("\nTRAIN FASDD UAV dataset")
    train_fasdd_uav_ds = dataset_fasdd.FASDDDataset(
        img_h=config_aimet.IMG_H, 
        img_w=config_aimet.IMG_W, 
        imgs_dir=config_aimet.FASDD_UAV_IMGS_DIR, 
        labels_file=config_aimet.FASDD_UAV_TRAIN_LABELS_FILE, 
        num_classes=config_aimet.N_CLASSES,
        ds_len=config_aimet.DS_LEN,
        transform=train_transform)
    print(f'\nTrain FASDD UAV dataset len: {len(train_fasdd_uav_ds)}')

    print("\nVAL FASDD UAV dataset")
    val_fasdd_uav_ds = dataset_fasdd.FASDDDataset(
        img_h=config_aimet.IMG_H, 
        img_w=config_aimet.IMG_W, 
        imgs_dir=config_aimet.FASDD_UAV_IMGS_DIR, 
        labels_file=config_aimet.FASDD_UAV_VAL_LABELS_FILE, 
        num_classes=config_aimet.N_CLASSES,
        ds_len=config_aimet.DS_LEN,
        transform=train_transform)
    print(f'\nVal FASDD UAV dataset len: {len(val_fasdd_uav_ds)}')
    
    print("\nTRAIN FASDD CV dataset")
    train_fasdd_cv_ds = dataset_fasdd.FASDDDataset(
        img_h=config_aimet.IMG_H, 
        img_w=config_aimet.IMG_W, 
        imgs_dir=config_aimet.FASDD_CV_IMGS_DIR, 
        labels_file=config_aimet.FASDD_CV_TRAIN_LABELS_FILE, 
        num_classes=config_aimet.N_CLASSES,
        ds_len=config_aimet.DS_LEN,
        transform=train_transform)
    print(f'\nTrain FASDD CV dataset len: {len(train_fasdd_cv_ds)}')

    print("\nVal FASDD CV dataset")
    val_fasdd_cv_ds = dataset_fasdd.FASDDDataset(
        img_h=config_aimet.IMG_H, 
        img_w=config_aimet.IMG_W, 
        imgs_dir=config_aimet.FASDD_CV_IMGS_DIR, 
        labels_file=config_aimet.FASDD_CV_VAL_LABELS_FILE, 
        num_classes=config_aimet.N_CLASSES,
        ds_len=config_aimet.DS_LEN,
        transform=train_transform)
    print(f'\nVal FASDD CV dataset len: {len(val_fasdd_cv_ds)}')
    
    print("\nConcatenate Train DFire and Train FASDD UAV datasets")
    #train_ds_concat = torch.utils.data.ConcatDataset((train_dfire_dataset, train_fasdd_uav_ds))
    #print(f'Train dataset len: {len(train_ds_concat)}')
    train_ds = torch.utils.data.ConcatDataset((train_dfire_dataset, train_fasdd_uav_ds))
    print(f'Train dataset len: {len(train_ds)}')
    print("Concatenate with Val FASDD UAV dataset")
    train_ds = torch.utils.data.ConcatDataset((train_ds, val_fasdd_uav_ds))
    print(f'Train dataset len: {len(train_ds)}')
    print("Concatenate with Train FASDD CV dataset")
    train_ds = torch.utils.data.ConcatDataset((train_ds, train_fasdd_cv_ds))
    print(f'Train dataset len: {len(train_ds)}')
    print("Concatenate with Val FASDD CV dataset")
    train_ds = torch.utils.data.ConcatDataset((train_ds, val_fasdd_cv_ds))
    print(f'Train dataset len: {len(train_ds)}')
    
    train_loader = DataLoader(
        dataset=train_ds,
        batch_size=config_aimet.BATCH_SIZE,
        num_workers=config_aimet.NUM_WORKERS,
        pin_memory=config_aimet.PIN_MEMORY,
        shuffle=True,
        drop_last=True)
    
    return train_loader

def get_val_loader():
    val_transform = A.Compose([
        A.Resize(config_aimet.IMG_H, config_aimet.IMG_W, p=1),
        ToTensorV2(p=1),
        ]
    )
    
    print("\nTEST DFire dataset")
    val_dfire_dataset = dataset_dfire.DFireDataset(
        img_h = config_aimet.IMG_H,
        img_w = config_aimet.IMG_W,
        img_dir = config_aimet.VAL_IMG_DIR,
        label_dir = config_aimet.VAL_LABEL_DIR,
        num_classes = config_aimet.N_CLASSES,
        ds_len = config_aimet.DS_LEN,
        transform=val_transform)
    print(f'\nTest dataset len: {len(val_dfire_dataset)}')
    
    print("\nTEST FASDD UAV dataset")
    val_fasdd_uav_ds = dataset_fasdd.FASDDDataset(
        img_h=config_aimet.IMG_H, 
        img_w=config_aimet.IMG_W, 
        imgs_dir=config_aimet.FASDD_UAV_IMGS_DIR, 
        labels_file=config_aimet.FASDD_UAV_TEST_LABELS_FILE, 
        num_classes=config_aimet.N_CLASSES,
        ds_len=config_aimet.DS_LEN,
        transform=val_transform)
    print(f'\nTest FASDD UAV dataset len: {len(val_fasdd_uav_ds)}')
    
    print("\nTEST FASDD CV dataset")
    val_fasdd_cv_ds = dataset_fasdd.FASDDDataset(
        img_h=config_aimet.IMG_H, 
        img_w=config_aimet.IMG_W, 
        imgs_dir=config_aimet.FASDD_CV_IMGS_DIR, 
        labels_file=config_aimet.FASDD_CV_TEST_LABELS_FILE, 
        num_classes=config_aimet.N_CLASSES,
        ds_len=config_aimet.DS_LEN,
        transform=val_transform)
    print(f'\nTest FASDD CV dataset len: {len(val_fasdd_cv_ds)}')
    
    print("Concatenate Test DFire and FASDD UAV datasets")
    val_ds_concat = torch.utils.data.ConcatDataset((val_dfire_dataset, val_fasdd_uav_ds))
    print(f'Test dataset len: {len(val_ds_concat)}')
    print("Concatenate with FASDD CV dataset")
    val_ds = torch.utils.data.ConcatDataset((val_ds_concat, val_fasdd_cv_ds))
    print(f'Test dataset len: {len(val_ds)}')
    
    val_loader = DataLoader(
        dataset=val_ds,
        batch_size=config_aimet.BATCH_SIZE,
        num_workers=config_aimet.NUM_WORKERS,
        pin_memory=config_aimet.PIN_MEMORY,
        shuffle=False,
        drop_last=True)
    
    return val_loader

