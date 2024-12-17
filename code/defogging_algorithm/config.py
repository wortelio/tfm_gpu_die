import torch

# ______________________________________________________________________ #
#                        Classes and Dimensions                          #
# ______________________________________________________________________ #
CLASSES = ["smoke", "fire"]
NUM_CLASSES = len(CLASSES)
#___   Padding Model  ___#
IMG_DIM = {'W':224, 'H':224} # (W, H)
IMG_H = IMG_DIM['H']
IMG_W = IMG_DIM['W']
NUM_CHANNELS = 3
# ______________________________________________________________________ #
#                        Folders and Datasets                            #
# ______________________________________________________________________ #
dfire_dir = '../../datasets/ds2fire/dfire_yolo/'
DFIRE_TRAIN_IMG_DIR = dfire_dir + 'train/images/'
DFIRE_TRAIN_LABEL_DIR = dfire_dir + 'train/labels/'
DFIRE_TEST_IMG_DIR = dfire_dir + 'test/images/'
DFIRE_TEST_LABEL_DIR = dfire_dir + 'test/labels/'

FASDD_UAV_IMGS_DIR = '../../datasets/fasdd/fasdd_uav/images/'
FASDD_UAV_TRAIN_LABELS_FILE = '../../datasets/fasdd/fasdd_uav/annotations/YOLO_UAV/train.txt'
FASDD_UAV_VAL_LABELS_FILE = '../../datasets/fasdd/fasdd_uav/annotations/YOLO_UAV/val.txt'
FASDD_UAV_TEST_LABELS_FILE = '../../datasets/fasdd/fasdd_uav/annotations/YOLO_UAV/test.txt'

FASDD_CV_IMGS_DIR = '../../datasets/fasdd/fasdd_cv/images/'
FASDD_CV_TRAIN_LABELS_FILE = '../../datasets/fasdd/fasdd_cv/annotations/YOLO_CV/train.txt'
FASDD_CV_VAL_LABELS_FILE = '../../datasets/fasdd/fasdd_cv/annotations/YOLO_CV/val.txt'
FASDD_CV_TEST_LABELS_FILE = '../../datasets/fasdd/fasdd_cv/annotations/YOLO_CV/test.txt'

CLOUDS_IMG_DIR = '../../datasets/clouds/images/'

DS_LEN = None

### DFire Mini ###
dfire_mini_dir = '../../datasets/dfire_mini/'
DFIRE_MINI_TRAIN_IMG_DIR = dfire_mini_dir + 'train/images/'
DFIRE_MINI_TRAIN_LABEL_DIR = dfire_mini_dir + 'train/labels/'
DFIRE_MINI_TEST_IMG_DIR = dfire_mini_dir + 'test/images/'
DFIRE_MINI_TEST_LABEL_DIR = dfire_mini_dir + 'test/labels/'
# ______________________________________________________________________ #
#                   Hyperparameters and More                             #
# ______________________________________________________________________ #
BREVITAS_MODEL = True
MODEL = "MY_MBLNET_V2_Resnet"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 64 
NUM_WORKERS = 1
PIN_MEMORY = True