import os
# ______________________________________________________________________ #
#                                Logs                                    #
# ______________________________________________________________________ #
EXPERIMENTS_FOLDER = 'experiments_CLK/'
EXPERIMENTS_FOLDER += '30_FPS_CLK_200ns/'
if not os.path.isdir(EXPERIMENTS_FOLDER):
    os.mkdir(EXPERIMENTS_FOLDER)

# RUN_FOLDER = '02_full_build_vvau_lut_hls_final_mvau_hls/'
RUN_FOLDER = '03_full_build_final_mvau_hls/'
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
# IMG_H = 160
# IMG_W = 160
NUM_CHANNELS = 3

