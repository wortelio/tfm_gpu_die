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

IOU_THRESHOLD = 0.5
SCORE_THRESHOLD = 0.2 