VERSION = '1.0.2'
VIDEO_WINDOW_NAME = 'BB-Tool ' + VERSION + ' : Video'
CTRL_WINDOW_NAME = 'BB-Tool ' + VERSION + ' : Control'
TRACKING_PARAMS_FILE = 'tracking_params.json'
LTS300_X = '45874644'
LTS300_Y = '45874637'

IMG_WIDTH = 1920
IMG_HEIGHT = 1080

VIDEO_PORT = 5556
VIDEO_IP = 'localhost'
TRACK_PORT = 5557
TRACK_IP = 'localhost'

FPS = 60
BBOX_AREA_THRESH = 100

TARGET_JUMP_THRESH = 300 # if the bounding box jumps by this much, assume it's an outlier
LP_IIR_DECAY = .6  # LP_IIR_DECAY * est_current + (1 - LP_IIR_DECAY) * obs_current

X_MOVE_SCALE = 20
Y_MOVE_SCALE = 20

STAGE_DEADBAND = 30