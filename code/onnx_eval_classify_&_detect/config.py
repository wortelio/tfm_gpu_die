import os
import torch

# ______________________________________________________________________ #
#                                Logs                                    #
# ______________________________________________________________________ #
RUN_FOLDER = 'experiments/'
if not os.path.isdir(RUN_FOLDER):
    os.mkdir(RUN_FOLDER)
# ______________________________________________________________________ #
#               Classes, Dimensions, IOUs, NMS, etc                      #
# ______________________________________________________________________ #
CLASSES = ["smoke", "fire"]
N_CLASSES = len(CLASSES)
BBOX_COLORS = {"smoke":(0,255,255), "fire":(255,255,0)}
GRID_COLOR = (100, 100, 100)

IMG_DIM = {'W':224, 'H':224} # (W, H)
IMG_H = IMG_DIM['H']
IMG_W = IMG_DIM['W']

S = 7
B = 2
C = N_CLASSES

MAX_OBJ = 10
IOU_THRESHOLD = 0.5
NMS_IOU_THRESHOLD = 0.5 # Supress boxes of the same inference during NMS
SCORE_THRESHOLD = 0.2 # For confidence score, to consider there is a positive sample in a cell
# ______________________________________________________________________ #
#                        Folders and Datasets                            #
# ______________________________________________________________________ #
MINI = True
# ===== DFire ===== #
if MINI == True:
    DFIRE_DS_DIR = '../../datasets/dfire_mini/'
    DFIRE_VAL_DIR = DFIRE_DS_DIR + 'train/'
    DFIRE_VAL_IMGS_DIR = DFIRE_VAL_DIR + 'images/'
    DFIRE_VAL_LABELS_DIR = DFIRE_VAL_DIR + 'labels/'
else:
    DFIRE_DS_DIR = '../../datasets/ds2fire/dfire_yolo/'
    DFIRE_VAL_DIR = DFIRE_DS_DIR + 'test/'
    DFIRE_VAL_IMGS_DIR = DFIRE_VAL_DIR + 'images/'
    DFIRE_VAL_LABELS_DIR = DFIRE_VAL_DIR + 'labels/'  

DS_LEN = None
# ______________________________________________________________________ #
#                   Hyperparameters and More                             #
# ______________________________________________________________________ #
MODEL = "BED"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 1 
NUM_WORKERS = 1
PIN_MEMORY = True


