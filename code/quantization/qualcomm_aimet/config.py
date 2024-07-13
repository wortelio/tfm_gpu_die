import os
import torch
# ______________________________________________________________________ #
#                                Logs                                    #
# ______________________________________________________________________ #
RUN_FOLDER = 'experiments_soft/' + 'test_v11_Pruning_090_Memory_after_SVD_080_FASDD/'
if not os.path.isdir(RUN_FOLDER):
    os.mkdir(RUN_FOLDER)
LOGS_FOLDER = RUN_FOLDER + 'logs/'
if not os.path.isdir(LOGS_FOLDER):
    os.mkdir(LOGS_FOLDER)
PLOTS_FOLDER = RUN_FOLDER + 'plots/'
if not os.path.isdir(PLOTS_FOLDER):
    os.mkdir(PLOTS_FOLDER)
PLOTS_FOLDER_2 = RUN_FOLDER + 'plots_2/'
if not os.path.isdir(PLOTS_FOLDER_2):
    os.mkdir(PLOTS_FOLDER_2)
WEIGHTS_FOLDER = RUN_FOLDER + 'weights/'
if not os.path.isdir(WEIGHTS_FOLDER):
    os.mkdir(WEIGHTS_FOLDER)
WEIGHTS_FOLDER_2 = RUN_FOLDER + 'weights_2/'
if not os.path.isdir(WEIGHTS_FOLDER_2):
    os.mkdir(WEIGHTS_FOLDER_2)
# ______________________________________________________________________ #
#                        AIMET CONFIGURATION                             #
# ______________________________________________________________________ #
USE_PREVIOUS_DIC = True
SVD_DIC_FILE = './data/greedy_selection_eval_scores_dict.pkl' if USE_PREVIOUS_DIC else None
PRUNING_DIC_FILE = './data/greedy_selection_eval_scores_dict.pkl' if USE_PREVIOUS_DIC else None
SVD_COMPRESSION_RATIO = 0.80
PRUNING_COMPRESSION_RATIO = 0.90
# ______________________________________________________________________ #
#                        Classes and Dimensions                          #
# ______________________________________________________________________ #
CLASSES = ["smoke", "fire"]
N_CLASSES = len(CLASSES)

IMG_DIM = {'W':224, 'H':224} # (W, H)
IMG_H = IMG_DIM['H']
IMG_W = IMG_DIM['W']

# ______________________________________________________________________ #
#                        Folders and Datasets                            #
# ______________________________________________________________________ #
ds_dir = '../../../datasets/ds2fire/dfire_yolo/'

train_dir = ds_dir + 'train/'
train_imgs = train_dir + 'images/'
train_labels = train_dir + 'labels/'

val_dir = ds_dir + 'test/'
val_imgs = val_dir + 'images/'
val_labels = val_dir + 'labels/'

TRAIN_IMG_DIR = train_imgs
TRAIN_LABEL_DIR = train_labels
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
MODEL = "BED"

LEARNING_RATE = 1e-4
# Optimizer
WEIGHT_DECAY = 1e-4
FACTOR = 0.8
PATIENCE = 2
THRES = 0.001
MIN_LR = 1e-6

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 64 
NUM_WORKERS = 8
PIN_MEMORY = True

EPOCHS = 15

LOAD_MODEL = True
LOAD_MODEL_DIR = './models/'
if MODEL == "BED":
    LOAD_MODEL_FILE = LOAD_MODEL_DIR + "BED_classifier__fused__dfire_fasdd.pt"


LOSS_FN = "BCE"
SMOKE_PRECISION_WEIGHT = 0.8
