VERSION = '1.0.2'
VIDEO_WINDOW_NAME = 'BB-Tool ' + VERSION + ' : Video'
CTRL_WINDOW_NAME = 'BB-Tool ' + VERSION + ' : Control'

# Stage Params
##############
LTS300_X = '45874644'
LTS300_Y = '45874637'
LTS300_Z = '45968230'

STAGE_USE_Z = True

STAGE_POSITION_IP = 'localhost'
STAGE_POSITION_PORT = 5561

STAGE_TANK_OFFSET = 10  # mm

# Spotter Camera Params
#######################
CAMERA_ID_SPOTTER = 1

IMG_WIDTH_SPOTTER = 640
IMG_HEIGHT_SPOTTER = 480  # configured for the e-con systems camera
COLOR_CHAN_SPOTTER = 3
IMG_DISP_WIDTH_SPOTTER = int(IMG_WIDTH_SPOTTER)
IMG_DISP_HEIGHT_SPOTTER = int(IMG_HEIGHT_SPOTTER)

DISPLAY_RESCALE_SPOTTER = IMG_DISP_WIDTH_SPOTTER / IMG_WIDTH_SPOTTER

FPS_SPOTTER = 30

VIDEO_PORT_SPOTTER = 5556
VIDEO_IP_SPOTTER = 'localhost'

ROTATE_SPOTTER = 0


# Zoom Camera Params
####################
CAMERA_ID_ZOOM = 0

IMG_WIDTH_ZOOM = 1920
IMG_HEIGHT_ZOOM = 1080
COLOR_CHAN_ZOOM = 1
IMG_DISP_WIDTH_ZOOM = 831
IMG_DISP_HEIGHT_ZOOM = 685

FPS_ZOOM = 30

VIDEO_PORT_ZOOM = 5555
VIDEO_IP_ZOOM = 'localhost'

ROTATE_ZOOM = 2

# distance from front of macro to focus plane is 140.38 mm
FOCUS_DISTANCE_ZOOM = 140.48 + 20  # mm, NOTE: This is the distance from the LL camera face, *not* from the macro lens

# Autofocus Params
##################

AUTOFOCUS_ENABLE = True
AUTOFOCUS_JOG = 10  # mm
AUTOFOCUS_DISTANCE = 150
N_AUTOFOCUS_JOG = int(AUTOFOCUS_DISTANCE / AUTOFOCUS_JOG)
AUTOFOCUS_IIR_DECAY = .85

AUTOFOCUS_PORT = 5558
AUTOFOCUS_IP = 'localhost'

CURRENT_FOCUS_PORT = 5559
CURRENT_FOCUS_IP = 'localhost'

FOCUS_ROI_PORT = 5560
FOCUS_ROI_IP = 'localhost'

EXPECTED_SHARPNESS_THRESHOLD = .6

# Tracking Params
#################

TRACK_PORT = 5557
TRACK_IP = 'localhost'

BBOX_AREA_THRESH = 1000

TARGET_JUMP_THRESH = 300  # if the bounding box jumps by this much, assume it's an outlier
LP_IIR_DECAY = .5  # LP_IIR_DECAY * est_current + (1 - LP_IIR_DECAY) * obs_current
LP_IIR_DECAY_2 = .7

X_MOVE_SCALE = 5  # Needs to be calibrated!!!!
Y_MOVE_SCALE = 5
Z_MOVE_SCALE = 20

STAGE_DEADBAND = .3  # mm space (currently not calibrated though)

ARROW_MOVE_RATE = 1

MACRO_LL_CENTER = (182, 284)  # center of macro FoV in LL frame (pixels)
MACRO_FOV_OFFSET = (MACRO_LL_CENTER[0] - IMG_WIDTH_SPOTTER / 2, MACRO_LL_CENTER[1] - IMG_HEIGHT_SPOTTER / 2)  # offset (in LL pixel space) of the Macro FoV relative to spotter FoV
