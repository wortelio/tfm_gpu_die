import torch

CLASSES = ["smoke", "fire"]
N_CLASSES = len(CLASSES)

IMG_DIM = {'W':224, 'H':224} # (W, H)
IMG_H = IMG_DIM['H']
IMG_W = IMG_DIM['W']

#---------------#
#    Dataset    #
#---------------#
DS_DIR = '../../../../datasets/ds2fire/dfire_yolo/'

TRAIN_DIR = DS_DIR + 'train/'
TRAIN_IMG_DIR = TRAIN_DIR + 'images/'
TRAIN_LABEL_DIR = TRAIN_DIR + 'labels/'

VAL_DIR = DS_DIR + 'test/'
VAL_IMG_DIR = VAL_DIR + 'images/'
VAL_LABEL_DIR = VAL_DIR + 'labels/'

DS_LEN = None
QUANT_DS_LEN = 512

#---------------#
#     Model     #
#---------------#
MODEL = "BED"

LEARNING_RATE = 0.001
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 64
WEIGHT_DECAY = 1e-3
EPOCHS = 125
NUM_WORKERS = 8
PIN_MEMORY = True
PRETRAINED = False
LOAD_MODEL = False
if MODEL == "BED":
    LOAD_MODEL_FILE = 'results/' + "bed_classifier_best.pt"

#---------------#
# Loss Function #
#---------------#
LOSS_FN = "BCE"
SMOKE_PRECISION_WEIGHT = 0.7


