import os
# ______________________________________________________________________ #
#                                Logs                                    #
# ______________________________________________________________________ #
EXPERIMENTS_FOLDER = 'experiments/'
EXPERIMENTS_FOLDER += '200_FPS/'
if not os.path.isdir(EXPERIMENTS_FOLDER):
    os.mkdir(EXPERIMENTS_FOLDER)

RUN_FOLDER = '01_estimates__V5_StandAlone_rtl_hls/'
# RUN_FOLDER = '01_estimates_vvau_lut_hls_mvau_hls/'
RUN_FOLDER = EXPERIMENTS_FOLDER + RUN_FOLDER
if not os.path.isdir(RUN_FOLDER):
    os.mkdir(RUN_FOLDER)
    print(f'Run folder created in: {RUN_FOLDER}')

# ______________________________________________________________________ #
#                        Classes and Dimensions                          #
# ______________________________________________________________________ #
NUM_CLASSES = 2

IMG_H = 224
IMG_W = 224
NUM_CHANNELS = 3

