import os
import torch
# ______________________________________________________________________ #
#                                Logs                                    #
# ______________________________________________________________________ #
RUN_FOLDER = 'experiments_resnet/' + 'test_v38__w1a1__DEEPER_Resnet_V51__clip1__128ds/'
if not os.path.isdir(RUN_FOLDER):
    os.mkdir(RUN_FOLDER)
LOGS_FOLDER = RUN_FOLDER + 'logs/'
if not os.path.isdir(LOGS_FOLDER):
    os.mkdir(LOGS_FOLDER)
PLOTS_FOLDER = RUN_FOLDER + 'plots/'
if not os.path.isdir(PLOTS_FOLDER):
    os.mkdir(PLOTS_FOLDER)
WEIGHTS_FOLDER = RUN_FOLDER + 'weights/'
if not os.path.isdir(WEIGHTS_FOLDER):
    os.mkdir(WEIGHTS_FOLDER)
ONNX_FOLDER = RUN_FOLDER + 'onnx/'
if not os.path.isdir(ONNX_FOLDER):
    os.mkdir(ONNX_FOLDER)
# ______________________________________________________________________ #
#                        Classes and Dimensions                          #
# ______________________________________________________________________ #
CLASSES = ["smoke", "fire"]
NUM_CLASSES = len(CLASSES)

#___   Padding Model  ___#
IMG_DIM = {'W':224, 'H':224} # (W, H)
#___ No Padding Model ___#
# IMG_DIM = {'W':230, 'H':230} # (W, H)

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

DS_LEN = 128

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
MODEL = "BNN_BED_Resnet_V7"

LEARNING_RATE = 1e-3
#LEARNING_RATE = 1e-4
# Optimizer
#WEIGHT_DECAY = 1e-4
WEIGHT_DECAY = 1e-3
FACTOR = 0.8
PATIENCE = 2
THRES = 0.001
MIN_LR = 1e-6

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 64 
NUM_WORKERS = 8
PIN_MEMORY = True

EPOCHS = 5

LOAD_MODEL = False
LOAD_MODEL_DIR = './models/'
if MODEL == "BED":
    LOAD_MODEL_FILE = LOAD_MODEL_DIR + "BED_classifier__fused__dfire_fasdd.pt"


LOSS_FN = "BCE"
SMOKE_PRECISION_WEIGHT = 0.8

# ______________________________________________________________________ #
#                             Quantization                               #
# ______________________________________________________________________ #
FIXED_POINT = True

# WEIGHTS_BIT_WIDTH = 4
# BIG_LAYERS_WEIGHTS_BIT_WIDTH = 2
# ACTIVATIONS_BIT_WIDTH = 8
# BIAS_BIT_WIDTH = 4

##### FINN
WEIGHTS_BIT_WIDTH = 4
BIG_LAYERS_WEIGHTS_BIT_WIDTH = 2 #4
ACTIVATIONS_BIT_WIDTH = 4
BIAS_BIT_WIDTH = 4
