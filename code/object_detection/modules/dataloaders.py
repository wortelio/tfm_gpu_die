import config
import modules.dataset_dfire as dataset_dfire
import modules.dataset_fasdd as dataset_fasdd
import torch
from torch.utils.data import DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2


# ______________________________________________________________ #
# ____________________     TRAIN LOADER     ____________________ #
# ______________________________________________________________ #
def get_train_loader():
    train_transform = A.Compose([
        A.HorizontalFlip(p=0.5),
        # If boxes are to close, it can remove some because they fall inside same cell
        A.RandomSizedBBoxSafeCrop(height=int(1.4*config.IMG_H),
                                  width= int(1.4*config.IMG_W),
                                  erosion_rate=0.3,
                                  p=0.3),
        A.OneOf([
            A.RandomBrightnessContrast(p=0.4),
            A.HueSaturationValue(hue_shift_limit=10, p=0.1),
            A.Blur(blur_limit=(3,3), p=0.3),
            A.CLAHE(clip_limit=2.0, p=0.2),
        ], p=0.9),
        #Shifting, scaling and rotation could dive 2 bbox inside same grid...
        #A.ShiftScaleRotate(rotate_limit=10, p=0.1),
        A.Resize(config.IMG_H, config.IMG_W, p=1),
        ToTensorV2(p=1),
    ], bbox_params=A.BboxParams(format='yolo', 
                                min_area= 8*8, #16*16, Changed to 8*8 to see smaller objects 
                                min_visibility=0.1, 
                                label_fields=['class_labels']))
    
    # TRAIN DATASET
    print("\nTRAIN DFIRE dataset")
    train_dfire_dataset = dataset_dfire.DFireDataset(
        img_h = config.IMG_H, 
        img_w = config.IMG_H, 
        imgs_dir = config.DFIRE_TRAIN_IMGS_DIR, 
        labels_dir = config.DFIRE_TRAIN_LABELS_DIR,            
        S = config.S, 
        C = config.C, 
        max_obj = config.MAX_OBJ,
        ds_len = config.DS_LEN,
        transform=train_transform, 
        target_transform=None)
    print(f'\nTrain DFire dataset len: {len(train_dfire_dataset)}')

    print("\nTRAIN FASDD UAV dataset")
    train_fasdd_uav_ds = dataset_fasdd.FASDDDataset(
        img_h = config.IMG_H, 
        img_w = config.IMG_H, 
        imgs_dir = config.FASDD_UAV_IMGS_DIR, 
        labels_dir = config.FASDD_UAV_LABELS_DIR,  
        file_labels_list = config.FASDD_UAV_TRAIN_LABELS_FILE,
        S = config.S, 
        C = config.C, 
        max_obj = config.MAX_OBJ,
        ds_len = config.DS_LEN,
        transform=train_transform, 
        target_transform=None)
    print(f'\nTrain FASDD UAV dataset len: {len(train_fasdd_uav_ds)}')
    
    print("\nVAL FASDD UAV dataset")
    val_fasdd_uav_ds = dataset_fasdd.FASDDDataset(
        img_h = config.IMG_H, 
        img_w = config.IMG_H, 
        imgs_dir = config.FASDD_UAV_IMGS_DIR, 
        labels_dir = config.FASDD_UAV_LABELS_DIR,  
        file_labels_list = config.FASDD_UAV_VAL_LABELS_FILE,
        S = config.S, 
        C = config.C, 
        max_obj = config.MAX_OBJ,
        ds_len = config.DS_LEN,
        transform=train_transform, 
        target_transform=None)
    print(f'\nVal FASDD UAV dataset len: {len(val_fasdd_uav_ds)}')
    
    print("\nTRAIN FASDD CV dataset")
    train_fasdd_cv_ds = dataset_fasdd.FASDDDataset(
        img_h = config.IMG_H, 
        img_w = config.IMG_H, 
        imgs_dir = config.FASDD_CV_IMGS_DIR, 
        labels_dir = config.FASDD_CV_LABELS_DIR,  
        file_labels_list = config.FASDD_CV_TRAIN_LABELS_FILE,
        S = config.S, 
        C = config.C, 
        max_obj = config.MAX_OBJ,
        ds_len = config.DS_LEN,
        transform=train_transform, 
        target_transform=None)
    print(f'\nTrain FASDD CV dataset len: {len(train_fasdd_cv_ds)}')
    
    print("\nVAL FASDD CV dataset")
    val_fasdd_cv_ds = dataset_fasdd.FASDDDataset(
        img_h = config.IMG_H, 
        img_w = config.IMG_H, 
        imgs_dir = config.FASDD_CV_IMGS_DIR, 
        labels_dir = config.FASDD_CV_LABELS_DIR,  
        file_labels_list = config.FASDD_CV_VAL_LABELS_FILE,
        S = config.S, 
        C = config.C, 
        max_obj = config.MAX_OBJ,
        ds_len = config.DS_LEN,
        transform=train_transform, 
        target_transform=None)
    print(f'\nVal FASDD CV dataset len: {len(val_fasdd_cv_ds)}')
    
    print("\nConcatenate Train DFire and Train FASDD UAV datasets")
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
        batch_size=config.BATCH_SIZE,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY,
        shuffle=True,
        drop_last=True)
    
    return train_loader

# ______________________________________________________________ #
# ____________________      VAL LOADER      ____________________ #
# ______________________________________________________________ #
def get_val_loader(dfire_len=config.VAL_DS_LEN, 
                   fasdd_uav_len=config.VAL_DS_LEN, 
                   fasdd_cv_len=config.VAL_DS_LEN):
    '''
    Default lengths, but changeable to adjust for SVD, Pruning, etc.
    '''
    val_transform = A.Compose([
        A.Resize(config.IMG_H, config.IMG_W, p=1),
        ToTensorV2(p=1),
    ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))
    
    print("\nTEST DFire dataset")
    val_dfire_dataset = dataset_dfire.DFireDataset(
        img_h = config.IMG_H, 
        img_w = config.IMG_H, 
        imgs_dir = config.DFIRE_VAL_IMGS_DIR, 
        labels_dir = config.DFIRE_VAL_LABELS_DIR,            
        S = config.S, 
        C = config.C, 
        max_obj = config.MAX_OBJ,
        ds_len = dfire_len,
        transform=val_transform, 
        target_transform=None)
    print(f'\nTest DFire dataset len: {len(val_dfire_dataset)}')
    
    print("\nTEST FASDD UAV dataset")
    val_fasdd_uav_ds = dataset_fasdd.FASDDDataset(
        img_h = config.IMG_H, 
        img_w = config.IMG_H, 
        imgs_dir = config.FASDD_UAV_IMGS_DIR, 
        labels_dir = config.FASDD_UAV_LABELS_DIR,  
        file_labels_list = config.FASDD_UAV_TEST_LABELS_FILE,
        S = config.S, 
        C = config.C, 
        max_obj = config.MAX_OBJ,
        ds_len = fasdd_uav_len,
        transform=val_transform, 
        target_transform=None)
    print(f'\nTest FASDD UAV dataset len: {len(val_fasdd_uav_ds)}')
    
    print("\nTEST FASDD CV dataset")
    val_fasdd_cv_ds = dataset_fasdd.FASDDDataset(
        img_h = config.IMG_H, 
        img_w = config.IMG_H, 
        imgs_dir = config.FASDD_CV_IMGS_DIR, 
        labels_dir = config.FASDD_CV_LABELS_DIR,  
        file_labels_list = config.FASDD_CV_TEST_LABELS_FILE,
        S = config.S, 
        C = config.C, 
        max_obj = config.MAX_OBJ,
        ds_len = fasdd_cv_len,
        transform=val_transform, 
        target_transform=None)
    print(f'\nTest FASDD CV dataset len: {len(val_fasdd_cv_ds)}')
    
    print("\nConcatenate Test DFire and FASDD UAV datasets")
    val_ds = torch.utils.data.ConcatDataset((val_dfire_dataset, val_fasdd_uav_ds))
    print(f'Test dataset len: {len(val_ds)}')
    print("Concatenate with FASDD CV dataset")
    val_ds = torch.utils.data.ConcatDataset((val_ds, val_fasdd_cv_ds))
    print(f'Test dataset len: {len(val_ds)}')
    
    val_loader = DataLoader(
        dataset=val_ds,
        batch_size=config.BATCH_SIZE,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY,
        shuffle=False,
        drop_last=True)
    
    return val_loader

# ______________________________________________________________ #
# ____________________   SPECIFIC LOADERS   ____________________ #
# ______________________________________________________________ #
def get_dfire_train_loader():
    train_transform = A.Compose([
        A.HorizontalFlip(p=0.5),
        # If boxes are to close, it can remove some because they fall inside same cell
        A.RandomSizedBBoxSafeCrop(height=int(1.4*config.IMG_H),
                                  width= int(1.4*config.IMG_W),
                                  erosion_rate=0.3,
                                  p=0.3),
        A.OneOf([
            A.RandomBrightnessContrast(p=0.4),
            A.HueSaturationValue(hue_shift_limit=10, p=0.2),
            A.Blur(blur_limit=(3,3), p=0.3),
            A.CLAHE(clip_limit=2.0, p=0.3),
        ], p=0.9),
            # Shifting, scaling and rotation could dive 2 bbox inside same grid...
            #A.ShiftScaleRotate(rotate_limit=10, p=0.2),
        A.Resize(config.IMG_H, config.IMG_W, p=1),
        ToTensorV2(p=1),
    ], bbox_params=A.BboxParams(format='yolo', 
                                min_area=16*16, 
                                min_visibility=0.1, 
                                label_fields=['class_labels']))
    
    # TRAIN DATASET
    print("\nTRAIN DFIRE dataset")
    train_dfire_dataset = dataset_dfire.DFireDataset(
        img_h = config.IMG_H, 
        img_w = config.IMG_H, 
        imgs_dir = config.DFIRE_TRAIN_IMGS_DIR, 
        labels_dir = config.DFIRE_TRAIN_LABELS_DIR,            
        S = config.S, 
        C = config.C, 
        max_obj = config.MAX_OBJ,
        ds_len = config.DS_LEN,
        transform=train_transform, 
        target_transform=None)
    print(f'\nTrain DFire dataset len: {len(train_dfire_dataset)}')
   
    train_loader = DataLoader(
        dataset=train_dfire_dataset,
        batch_size=config.BATCH_SIZE,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY,
        shuffle=True,
        drop_last=True)
    
    return train_loader


def get_dfire_val_loader(dfire_len=config.VAL_DS_LEN):
    val_transform = A.Compose([
        A.Resize(config.IMG_H, config.IMG_W, p=1),
        ToTensorV2(p=1),
    ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))
       
    print("\nTEST DFire dataset")
    val_dfire_dataset = dataset_dfire.DFireDataset(
        img_h = config.IMG_H, 
        img_w = config.IMG_H, 
        imgs_dir = config.DFIRE_VAL_IMGS_DIR, 
        labels_dir = config.DFIRE_VAL_LABELS_DIR,            
        S = config.S, 
        C = config.C, 
        max_obj = config.MAX_OBJ,
        ds_len = dfire_len,
        transform=val_transform, 
        target_transform=None)
    print(f'\nTest dataset len: {len(val_dfire_dataset)}')
    
    val_loader = DataLoader(
        dataset=val_dfire_dataset,
        batch_size=config.BATCH_SIZE,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY,
        shuffle=False,
        drop_last=True)
    
    return val_loader

def get_fasdd_uav_train_loader():
    train_transform = A.Compose([
        A.HorizontalFlip(p=0.5),
        # If boxes are to close, it can remove some because they fall inside same cell
        A.RandomSizedBBoxSafeCrop(height=int(1.4*config.IMG_H),
                                  width= int(1.4*config.IMG_W),
                                  erosion_rate=0.3,
                                  p=0.3),
        A.OneOf([
            A.RandomBrightnessContrast(p=0.4),
            A.HueSaturationValue(hue_shift_limit=10, p=0.2),
            A.Blur(blur_limit=(3,3), p=0.3),
            A.CLAHE(clip_limit=2.0, p=0.3),
        ], p=0.9),
            # Shifting, scaling and rotation could dive 2 bbox inside same grid...
            #A.ShiftScaleRotate(rotate_limit=10, p=0.2),
        A.Resize(config.IMG_H, config.IMG_W, p=1),
        ToTensorV2(p=1),
    ], bbox_params=A.BboxParams(format='yolo', 
                                min_area=16*16, 
                                min_visibility=0.1, 
                                label_fields=['class_labels']))
       
    print("\nTRAIN FASDD UAV dataset")
    train_fasdd_uav_ds = dataset_fasdd.FASDDDataset(
        img_h = config.IMG_H, 
        img_w = config.IMG_H, 
        imgs_dir = config.FASDD_UAV_IMGS_DIR, 
        labels_dir = config.FASDD_UAV_LABELS_DIR,  
        file_labels_list = config.FASDD_UAV_TRAIN_LABELS_FILE,
        S = config.S, 
        C = config.C, 
        max_obj = config.MAX_OBJ,
        ds_len = config.DS_LEN,
        transform=train_transform, 
        target_transform=None)
    print(f'\nTrain FASDD UAV dataset len: {len(train_fasdd_uav_ds)}')
    
    print("\nVAL FASDD UAV dataset")
    val_fasdd_uav_ds = dataset_fasdd.FASDDDataset(
        img_h = config.IMG_H, 
        img_w = config.IMG_H, 
        imgs_dir = config.FASDD_UAV_IMGS_DIR, 
        labels_dir = config.FASDD_UAV_LABELS_DIR,  
        file_labels_list = config.FASDD_UAV_VAL_LABELS_FILE,
        S = config.S, 
        C = config.C, 
        max_obj = config.MAX_OBJ,
        ds_len = config.DS_LEN,
        transform=train_transform, 
        target_transform=None)
    print(f'\nVal FASDD UAV dataset len: {len(val_fasdd_uav_ds)}')
    
    print("\nConcatenate Train FASDD UAV and Val FASDD UAV datasets")
    train_ds_concat = torch.utils.data.ConcatDataset((train_fasdd_uav_ds, val_fasdd_uav_ds))
    print(f'Train dataset len: {len(train_ds_concat)}')   
    
    train_loader = DataLoader(
        dataset=train_ds_concat,
        batch_size=config.BATCH_SIZE,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY,
        shuffle=True,
        drop_last=True)
    
    return train_loader
    

def get_fasdd_uav_val_loader(fasdd_uav_len=config.VAL_DS_LEN):
    val_transform = A.Compose([
        A.Resize(config.IMG_H, config.IMG_W, p=1),
        ToTensorV2(p=1),
    ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))
       
    print("\nTEST FASDD UAV dataset")
    test_fasdd_uav_ds = dataset_fasdd.FASDDDataset(
        img_h = config.IMG_H, 
        img_w = config.IMG_H, 
        imgs_dir = config.FASDD_UAV_IMGS_DIR, 
        labels_dir = config.FASDD_UAV_LABELS_DIR,  
        file_labels_list = config.FASDD_UAV_TEST_LABELS_FILE,
        S = config.S, 
        C = config.C, 
        max_obj = config.MAX_OBJ,
        ds_len = fasdd_uav_len,
        transform=val_transform, 
        target_transform=None)
    print(f'\nTest FASDD UAV dataset len: {len(test_fasdd_uav_ds)}')
    
    
    val_loader = DataLoader(
        dataset=test_fasdd_uav_ds,
        batch_size=config.BATCH_SIZE,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY,
        shuffle=False,
        drop_last=True)
    
    return val_loader

def get_fasdd_cv_val_loader(fasdd_cv_len=config.VAL_DS_LEN):
    val_transform = A.Compose([
        A.Resize(config.IMG_H, config.IMG_W, p=1),
        ToTensorV2(p=1),
    ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))
       
    print("\nTEST FASDD CV dataset")
    test_fasdd_cv_ds = dataset_fasdd.FASDDDataset(
        img_h = config.IMG_H, 
        img_w = config.IMG_H, 
        imgs_dir = config.FASDD_CV_IMGS_DIR, 
        labels_dir = config.FASDD_CV_LABELS_DIR,  
        file_labels_list = config.FASDD_CV_TEST_LABELS_FILE,
        S = config.S, 
        C = config.C, 
        max_obj = config.MAX_OBJ,
        ds_len = fasdd_cv_len,
        transform=val_transform, 
        target_transform=None)
    print(f'\nTest FASDD CV dataset len: {len(test_fasdd_cv_ds)}')
    
    
    val_loader = DataLoader(
        dataset=test_fasdd_cv_ds,
        batch_size=config.BATCH_SIZE,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY,
        shuffle=False,
        drop_last=True)
    
    return val_loader


def get_fasdd_rs_val_loader(fasdd_rs_len=config.VAL_DS_LEN):
    val_transform = A.Compose([
        A.Resize(config.IMG_H, config.IMG_W, p=1),
        ToTensorV2(p=1),
    ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))
       
    print("\nTEST FASDD RS dataset")
    test_fasdd_rs_ds = dataset_fasdd.FASDDDataset(
        img_h = config.IMG_H, 
        img_w = config.IMG_H, 
        imgs_dir = config.FASDD_RS_IMGS_DIR, 
        labels_dir = config.FASDD_RS_LABELS_DIR,  
        file_labels_list = config.FASDD_RS_TEST_LABELS_FILE,
        S = config.S, 
        C = config.C, 
        max_obj = config.MAX_OBJ,
        ds_len = fasdd_rs_len,
        ds_rs = True,
        transform=val_transform, 
        target_transform=None)
    print(f'\nTest FASDD RS dataset len: {len(test_fasdd_rs_ds)}')
    
    
    val_loader = DataLoader(
        dataset=test_fasdd_rs_ds,
        batch_size=config.BATCH_SIZE,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY,
        shuffle=False,
        drop_last=True)
    
    return val_loader
