VERSION = '1.0.2'
VIDEO_WINDOW_NAME = 'BB-Tool ' + VERSION + ' : Video'
CTRL_WINDOW_NAME = 'BB-Tool ' + VERSION + ' : Control'

# Stage Params
##############
LTS300_X = '45874644'
LTS300_Y = '45874637'
LTS300_Z = '45968230'

STAGE_USE_Z = True

# Spotter Camera Params
#######################
CAMERA_ID_SPOTTER = 0

IMG_WIDTH_SPOTTER = 1920
IMG_HEIGHT_SPOTTER = 1080
IMG_DISP_WIDTH_SPOTTER = int(IMG_WIDTH_SPOTTER / 4)
IMG_DISP_HEIGHT_SPOTTER = int(IMG_HEIGHT_SPOTTER / 4)

DISPLAY_RESCALE_SPOTTER = IMG_DISP_WIDTH_SPOTTER / IMG_WIDTH_SPOTTER

FPS_SPOTTER = 30

VIDEO_PORT_SPOTTER = 5556
VIDEO_IP_SPOTTER = 'localhost'

ROTATE_SPOTTER = False


# Zoom Camera Params
####################
CAMERA_ID_ZOOM = 1

IMG_WIDTH_ZOOM = 1920
IMG_HEIGHT_ZOOM = 1080
IMG_DISP_WIDTH_ZOOM = 831
IMG_DISP_HEIGHT_ZOOM = 685

FPS_ZOOM = 30

VIDEO_PORT_ZOOM = 5555
VIDEO_IP_ZOOM = 'localhost'

# Autofocus Params
##################

AUTOFOCUS_JOG = 10 # mm
AUTOFOCUS_DISTANCE = 150
N_AUTOFOCUS_JOG = int(AUTOFOCUS_DISTANCE / AUTOFOCUS_JOG)

# Tracking Params
#################

TRACK_PORT = 5557
TRACK_IP = 'localhost'

BBOX_AREA_THRESH = 300

TARGET_JUMP_THRESH = 300  # if the bounding box jumps by this much, assume it's an outlier
LP_IIR_DECAY = .5  # LP_IIR_DECAY * est_current + (1 - LP_IIR_DECAY) * obs_current
LP_IIR_DECAY_2 = .7

X_MOVE_SCALE = 20  # Needs to be calibrated!!!!
Y_MOVE_SCALE = 20

STAGE_DEADBAND = 1  # mm space (currently not calibrated though

ARROW_MOVE_RATE = 5