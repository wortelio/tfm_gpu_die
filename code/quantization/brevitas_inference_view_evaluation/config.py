import torch

CLASSES = ["smoke", "fire"]
N_CLASSES = len(CLASSES)

IMG_DIM = {'W':224, 'H':224} # (W, H)
IMG_H = IMG_DIM['H']
IMG_W = IMG_DIM['W']
# ______________________________________________________________________ #
#                        Folders and Datasets                            #
# ______________________________________________________________________ #
ds_dir = '../../../datasets/ds2fire/dfire_yolo/'

val_dir = ds_dir + 'test/'
val_imgs = val_dir + 'images/'
val_labels = val_dir + 'labels/'

VAL_IMG_DIR = val_imgs
VAL_LABEL_DIR = val_labels

FASDD_UAV_IMGS_DIR = '../../../datasets/fasdd/fasdd_uav/images/'
FASDD_UAV_TRAIN_LABELS_FILE = '../../../datasets/fasdd/fasdd_uav/annotations/YOLO_UAV/train.txt'
FASDD_UAV_TEST_LABELS_FILE = '../../../datasets/fasdd/fasdd_uav/annotations/YOLO_UAV/test.txt'

FASDD_CV_IMGS_DIR = '../../../datasets/fasdd/fasdd_cv/images/'
FASDD_CV_TRAIN_LABELS_FILE = '../../../datasets/fasdd/fasdd_cv/annotations/YOLO_CV/train.txt'
FASDD_CV_TEST_LABELS_FILE = '../../../datasets/fasdd/fasdd_cv/annotations/YOLO_CV/test.txt'


DS_LEN = None
# ______________________________________________________________________ #
#                   Hyperparameters and More                             #
# ______________________________________________________________________ #
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 64 
NUM_WORKERS = 8
PIN_MEMORY = True
# ______________________________________________________________________ #
#                             NO COMPRESSION                             #
# ______________________________________________________________________ #
NO_COMP_WEIGHTS_BIT_WIDTH = 4
NO_COMP_BIG_LAYERS_WEIGHTS_BIT_WIDTH = 2
NO_COMP_ACTIVATIONS_BIT_WIDTH = 8
NO_COMP_BIAS_BIT_WIDTH = 4
# ______________________________________________________________________ #
#                             LOW COMPRESSION                            #
# ______________________________________________________________________ #
LOW_COMP_WEIGHTS_BIT_WIDTH = 4
LOW_COMP_BIG_LAYERS_WEIGHTS_BIT_WIDTH = 3
LOW_COMP_ACTIVATIONS_BIT_WIDTH = 6
LOW_COMP_BIAS_BIT_WIDTH = 4
# ______________________________________________________________________ #
#                          MEDIUM COMPRESSION                            #
# ______________________________________________________________________ #
MED_COMP_WEIGHTS_BIT_WIDTH = 4
MED_COMP_BIG_LAYERS_WEIGHTS_BIT_WIDTH = 3
MED_COMP_ACTIVATIONS_BIT_WIDTH = 8
MED_COMP_BIAS_BIT_WIDTH = 4
# ______________________________________________________________________ #
#                             HIGH COMPRESSION                           #
# ______________________________________________________________________ #
HIGH_COMP_WEIGHTS_BIT_WIDTH = 4
HIGH_COMP_BIG_LAYERS_WEIGHTS_BIT_WIDTH = 4
HIGH_COMP_ACTIVATIONS_BIT_WIDTH = 8
HIGH_COMP_BIAS_BIT_WIDTH = 4
