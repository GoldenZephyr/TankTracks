VERSION = '1.0.2'
VIDEO_WINDOW_NAME = 'BB-Tool ' + VERSION + ' : Video'
CTRL_WINDOW_NAME = 'BB-Tool ' + VERSION + ' : Control'

# ZMQ IP / Ports
################
STAGE_POSITION_IP = 'localhost'
VIDEO_SPOTTER_IP = 'localhost'
VIDEO_ZOOM_IP = 'localhost'
AUTOFOCUS_IP = 'localhost'
CURRENT_FOCUS_IP = 'localhost'
FOCUS_ROI_IP = 'localhost'
TRACK_IP = 'localhost'
MACRO_SHARPNESS_IP = 'localhost'
SPOTTER_FOCUS_STATE_IP = 'localhost'


VIDEO_ZOOM_PORT = 5555
VIDEO_SPOTTER_PORT = 5556
TRACK_PORT = 5557
AUTOFOCUS_PORT = 5558
CURRENT_FOCUS_PORT = 5559
FOCUS_ROI_PORT = 5560
STAGE_POSITION_PORT = 5561
MACRO_SHARPNESS_PORT = 5562
SPOTTER_FOCUS_STATE_PORT = 5563


# Stage Params
##############
LTS300_X = '45874644'
LTS300_Y = '45874637'
LTS300_Z = '45968230'

STAGE_USE_Z = True

STAGE_TANK_OFFSET = 55  # mm
TANK_DEPTH_MM = 120  # mm

# Spotter Camera Params
#######################
CAMERA_ID_SPOTTER = 1

IMG_WIDTH_SPOTTER = 640
IMG_HEIGHT_SPOTTER = 480  # configured for the e-con systems camera
COLOR_CHAN_SPOTTER = 3
IMG_DISP_WIDTH_SPOTTER = 960 #int(IMG_WIDTH_SPOTTER)
IMG_DISP_HEIGHT_SPOTTER = 540 #int(IMG_HEIGHT_SPOTTER)

IMG_TRACK_WIDTH_SPOTTER = 1920
IMG_TRACK_HEIGHT_SPOTTER = 1080

DISPLAY_RESCALE_TRACK_SPOTTER = IMG_DISP_WIDTH_SPOTTER / IMG_TRACK_WIDTH_SPOTTER
DISPLAY_RESCALE_SPOTTER = IMG_DISP_WIDTH_SPOTTER / IMG_WIDTH_SPOTTER

FPS_SPOTTER = 30


ROTATE_SPOTTER = 0


# Zoom Camera Params
####################
CAMERA_ID_ZOOM = 0

IMG_WIDTH_ZOOM = 1920
IMG_HEIGHT_ZOOM = 1080
COLOR_CHAN_ZOOM = 1
IMG_DISP_WIDTH_ZOOM = 1920
IMG_DISP_HEIGHT_ZOOM = 1080

FPS_ZOOM = 30


ROTATE_ZOOM = 2

# distance from front of macro to focus plane is 140.38 mm
FOCUS_DISTANCE_ZOOM = 140.48 + 40  # mm, NOTE: This is the distance from the LL camera face, *not* from the macro lens


# Autofocus Params
##################

AUTOFOCUS_ENABLE = True
AUTOFOCUS_JOG = 10  # mm
AUTOFOCUS_DISTANCE = 150
N_AUTOFOCUS_JOG = int(AUTOFOCUS_DISTANCE / AUTOFOCUS_JOG)
AUTOFOCUS_IIR_DECAY = .8




EXPECTED_SHARPNESS_THRESHOLD = .6

# Tracking Params
#################


BBOX_AREA_THRESH = 1000

TARGET_JUMP_THRESH = 300  # if the bounding box jumps by this much, assume it's an outlier
LP_IIR_DECAY = .5  # LP_IIR_DECAY * est_current + (1 - LP_IIR_DECAY) * obs_current
LP_IIR_DECAY_2 = .7

X_MOVE_SCALE = 20  # Needs to be calibrated!!!!
Y_MOVE_SCALE = 20
Z_MOVE_SCALE = 1

STAGE_DEADBAND = .3  # mm space (currently not calibrated though)

ARROW_MOVE_RATE = 1

MACRO_LL_CENTER = (190, 270)  # center of macro FoV in LL frame (pixels)
MACRO_FOV_OFFSET = (MACRO_LL_CENTER[0] - IMG_WIDTH_SPOTTER / 2, MACRO_LL_CENTER[1] - IMG_HEIGHT_SPOTTER / 2)  # offset (in LL pixel space) of the Macro FoV relative to spotter FoV

BYPASS_LL_ESTIMATE = False
