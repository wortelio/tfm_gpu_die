import os
# ______________________________________________________________________ #
#                                Logs                                    #
# ______________________________________________________________________ #
# EXPERIMENTS_FOLDER = 'experiments_CLK/'
# EXPERIMENTS_FOLDER += 'NEW_30_FPS_CLK_250ns/'

EXPERIMENTS_FOLDER = 'experiments/'
EXPERIMENTS_FOLDER += '750_FPS/'
if not os.path.isdir(EXPERIMENTS_FOLDER):
    os.mkdir(EXPERIMENTS_FOLDER)

# RUN_FOLDER = '40_estimates_mvau_rtl_mvau_wwidth_max_16_manual_folding/'
# RUN_FOLDER = '25_full_build_final_mvau_rtl_mvau_wwidth_max_16/'
RUN_FOLDER = '41_full_build_json_mvau_rtl_mvau_wwidth_max_16_manual_folding/'

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

