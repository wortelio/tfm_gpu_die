import os
import torch

# ______________________________________________________________________ #
#                                Logs                                    #
# ______________________________________________________________________ #
RUN_FOLDER = 'experiments/' + 'test_11_307K_full_ds/'
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
#               Classes, Dimensions, IOUs, NMS, etc                      #
# ______________________________________________________________________ #
CLASSES = ["smoke", "fire"]
N_CLASSES = len(CLASSES)
BBOX_COLORS = {"smoke":(0,255,255), "fire":(255,255,0)}
GRID_COLOR = (100, 100, 100)

IMG_DIM = {'W':224, 'H':224} 
IMG_H = IMG_DIM['H']
IMG_W = IMG_DIM['W']
NUM_CHANNELS = 3

S = 7
B = 2
C = N_CLASSES

MAX_OBJ = 10
IOU_THRESHOLD = 0.5
NMS_IOU_THRESHOLD = 0.5 # Supress boxes of the same inference during NMS
SCORE_THRESHOLD = 0.2 # For confidence score, to consider there is a positive sample in a cell
# If you config 0.001, there are so many predictions that training is really slow during the epochs calculating mAP
# It is better to use 0.001 only for the final model, to check how PR curve is

# ______________________________________________________________________ #
#                        Folders and Datasets                            #
# ______________________________________________________________________ #

# ===== DFire ===== #
DFIRE_DS_DIR = '../../datasets/ds2fire/dfire_yolo/'

DFIRE_TRAIN_DIR = DFIRE_DS_DIR + 'train/'
DFIRE_TRAIN_IMGS_DIR = DFIRE_TRAIN_DIR + 'images/'
DFIRE_TRAIN_LABELS_DIR = DFIRE_TRAIN_DIR + 'labels/'

DFIRE_VAL_DIR = DFIRE_DS_DIR + 'test/'
DFIRE_VAL_IMGS_DIR = DFIRE_VAL_DIR + 'images/'
DFIRE_VAL_LABELS_DIR = DFIRE_VAL_DIR + 'labels/'

# ===== FASDD ===== #
FASDD_UAV_IMGS_DIR = '../../datasets/fasdd/fasdd_uav/images/'
FASDD_UAV_LABELS_DIR = '../../datasets/fasdd/fasdd_uav/annotations/YOLO_UAV/labels/'
FASDD_UAV_TRAIN_LABELS_FILE = '../../datasets/fasdd/fasdd_uav/annotations/YOLO_UAV/train.txt'
FASDD_UAV_VAL_LABELS_FILE = '../../datasets/fasdd/fasdd_uav/annotations/YOLO_UAV/val.txt'
FASDD_UAV_TEST_LABELS_FILE = '../../datasets/fasdd/fasdd_uav/annotations/YOLO_UAV/test.txt'

FASDD_CV_IMGS_DIR = '../../datasets/fasdd/fasdd_cv/images/'
FASDD_CV_LABELS_DIR = '../../datasets/fasdd/fasdd_cv/annotations/YOLO_CV/labels/'
FASDD_CV_TRAIN_LABELS_FILE = '../../datasets/fasdd/fasdd_cv/annotations/YOLO_CV/train.txt'
FASDD_CV_VAL_LABELS_FILE = '../../datasets/fasdd/fasdd_cv/annotations/YOLO_CV/val.txt'
FASDD_CV_TEST_LABELS_FILE = '../../datasets/fasdd/fasdd_cv/annotations/YOLO_CV/test.txt'

FASDD_RS_IMGS_DIR = '../../datasets/fasdd/fasdd_rs/images/'
FASDD_RS_LABELS_DIR = '../../datasets/fasdd/fasdd_rs/annotations/YOLO_RS_RGB/labels/'
FASDD_RS_TRAIN_LABELS_FILE = '../../datasets/fasdd/fasdd_rs/annotations/YOLO_RS_RGB/train.txt'
FASDD_RS_VAL_LABELS_FILE = '../../datasets/fasdd/fasdd_rs/annotations/YOLO_RS_RGB/val.txt'
FASDD_RS_TEST_LABELS_FILE = '../../datasets/fasdd/fasdd_rs/annotations/YOLO_RS_RGB/test.txt'

DS_LEN = None
VAL_DS_LEN = None
# ______________________________________________________________________ #
#                   Hyperparameters and More                             #
# ______________________________________________________________________ #
MODEL = "MOBILENETV2_DET"

LEARNING_RATE = 1e-3
GRADIENTS_CLIP_NORM = 10
# Optimizer
WEIGHT_DECAY = 1e-3
FACTOR = 0.8
PATIENCE = 3
THRES = 0.01
MIN_LR = 1e-6

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 64 
NUM_WORKERS = 8
PIN_MEMORY = True

EPOCHS = 125

LOSS_FN = "YOLOV1_LOSS"
LAMBDA_L1_LOSS = 0

# ______________________________________________________________________ #
#                             Quantization                               #
# ______________________________________________________________________ #
FIXED_POINT = True

WEIGHTS_BIT_WIDTH = 4
BIG_LAYERS_WEIGHTS_BIT_WIDTH = 4
HEAD_WEIGHTS_BIT_WIDTH = 8
BIAS_BIT_WIDTH = 4
ACTIVATIONS_BIT_WIDTH = 4
