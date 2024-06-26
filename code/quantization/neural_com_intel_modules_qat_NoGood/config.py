import os
import torch

CLASSES = ["smoke", "fire"]
N_CLASSES = len(CLASSES)

IMG_DIM = {'W':224, 'H':224} # (W, H)
IMG_H = IMG_DIM['H']
IMG_W = IMG_DIM['W']

''' ============================
             Folders
============================ '''
ds_dir = '../../../datasets/ds2fire/dfire_yolo/'

train_dir = ds_dir + 'train/'
train_imgs = train_dir + 'images/'
train_labels = train_dir + 'labels/'

#val_dir = ds_dir + 'test/'

val_dir = ds_dir + 'test/'
val_imgs = val_dir + 'images/'
val_labels = val_dir + 'labels/'

# print(f'Train dir: {os.listdir(train_dir)}')
# print(f'val dir: {os.listdir(val_dir)}')

RUN_FOLDER = 'qat_v0/'
LOGS_FOLDER = RUN_FOLDER + 'logs/'
PLOTS_FOLDER = RUN_FOLDER + 'plots/'
''' ============================
    Hyperparameters and More
============================ '''

MODEL = "BED"

LEARNING_RATE = 0.001
#DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DEVICE = "cpu"
BATCH_SIZE = 64 
WEIGHT_DECAY = 1e-3
EPOCHS = 3 
NUM_WORKERS = 8
PIN_MEMORY = True
LOAD_MODEL = False
if MODEL == "BED":
    LOAD_MODEL_FILE = MODEL + "_classifier_best.pt"
elif MODEL == "SQUEEZE":
    LOAD_MODEL_FILE = MODEL + "_classifier_best.pt"

# Optimizer
FACTOR = 0.8
PATIENCE = 2
THRES = 0.001
MIN_LR = 1e-6

LOSS_FN = "BCE"
SMOKE_PRECISION_WEIGHT = 0.7

TRAIN_IMG_DIR = train_imgs
TRAIN_LABEL_DIR = train_labels
VAL_IMG_DIR = val_imgs
VAL_LABEL_DIR = val_labels

DS_LEN = None