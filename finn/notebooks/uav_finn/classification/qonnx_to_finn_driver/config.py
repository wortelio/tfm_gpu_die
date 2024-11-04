import os
# ______________________________________________________________________ #
#                                Logs                                    #
# ______________________________________________________________________ #
EXPERIMENTS_FOLDER = 'experiments_pynq-z1/'
if not os.path.isdir(EXPERIMENTS_FOLDER):
    os.mkdir(EXPERIMENTS_FOLDER)

RUN_FOLDER = '353_pynq-z1__700FPS__workspace__AIMET_Balanced__BIPOLAR__w4W2a4__full_build/'
RUN_FOLDER = EXPERIMENTS_FOLDER + RUN_FOLDER
if not os.path.isdir(RUN_FOLDER):
    os.mkdir(RUN_FOLDER)
    print(f'Run folder created in: {RUN_FOLDER}')

# BUILD_FOLDER = RUN_FOLDER + 'finn_build'
# if not os.path.isdir(BUILD_FOLDER):
#     os.mkdir(BUILD_FOLDER)
#     print(f'Build folder created in: {BUILD_FOLDER}')

# TMP_FOLDER = RUN_FOLDER + 'tmp'
# if not os.path.isdir(TMP_FOLDER):
#     os.mkdir(TMP_FOLDER)
#     print(f'Temp folder created in: {TMP_FOLDER}')

# LOGS_FOLDER = RUN_FOLDER + 'logs/'
# if not os.path.isdir(LOGS_FOLDER):
#     os.mkdir(LOGS_FOLDER)

# MODELS_FOLDER = RUN_FOLDER + 'models/'
# if not os.path.isdir(MODELS_FOLDER):
#     os.mkdir(MODELS_FOLDER)
#     print(f'Models folder created in: {MODELS_FOLDER}')

# ______________________________________________________________________ #
#                        Classes and Dimensions                          #
# ______________________________________________________________________ #
NUM_CLASSES = 2

IMG_H = 224
IMG_W = 224
NUM_CHANNELS = 3

# ______________________________________________________________________ #
#                             Quantization                               #
# ______________________________________________________________________ #
# WEIGHTS_BIT_WIDTH = 4
# ACTIVATIONS_BIT_WIDTH = 8

FIXED_POINT = True

WEIGHTS_BIT_WIDTH = 4
BIG_LAYERS_WEIGHTS_BIT_WIDTH = 2
ACTIVATIONS_BIT_WIDTH = 4
BIAS_BIT_WIDTH = 4
