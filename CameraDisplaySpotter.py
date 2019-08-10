#!/usr/bin/python3
import time
import signal
import sys

import zmq
import cv2
import numpy as np
import keyboard

from EnhancedWindow import EnhancedWindow
import cvui
import Parameters as p
import Messages as m

from trackers.KCFTracker import KCFTracker
from trackers.KCFTracker import update_kcf_tracker

from trackers.CannyTracker import CannyTracker
from trackers.CannyTracker import update_canny_tracker

from trackers.ThresholdTracker import ThresholdTracker
from trackers.ThresholdTracker import update_threshold_tracker

keep_running = True


class SharpnessFocusState:
    def __init__(self):
        self.object_distance_ll = None
        self.stage_z = None
        self.macro_sharpness = None
        self.mode = None
        self.sweep_lowerbound_abs = None
        self.best_sharpness = None
        self.z_moving = None
        self.stage_z_dir = None
        self.stage_sweep_end = None


class ControlPanes:
    def __init__(self):
        self.stage_control_pane = None
        self.focus_control_pane = None
        self.tracker_select_pane = None
        self.canny_settings_pane = None
        self.threshold_setting_pane = None


def tank_corners_clip(frame, stage_pos, stage_zero, world_points, intrinsic):
    stage_delta = stage_pos - stage_zero
    corners_translated = world_points - stage_delta

    full_corners_img = np.zeros((2,8))
    for pt_ix in range(8):
        corner_img = intrinsic @ corners_translated[:, pt_ix]
        corner_img = corner_img / corner_img[2]
        full_corners_img[0,pt_ix] = corner_img[0]
        full_corners_img[1,pt_ix] = corner_img[1]
        #center = (int(corner_img[0]), int(corner_img[1]))
        #if pt_ix < 4:
        #    color = (0,0,255)
        #else:
        #    color = (255,0,0)

        #cv2.circle(frame, center, 5, color, -1)
    #cv2.imshow('MaskWindow1', frame)

    poly_frame_front = np.zeros(frame.shape, dtype=np.uint8)
    poly_frame_back = np.zeros(frame.shape, dtype=np.uint8)
    poly_frame_front = cv2.fillPoly(poly_frame_front, [np.array([full_corners_img[:,0], full_corners_img[:,1], full_corners_img[:,2], full_corners_img[:,3]], dtype='int32')], (255, 255, 255))
    poly_frame_back = cv2.fillPoly(poly_frame_back, [np.array([full_corners_img[:,4], full_corners_img[:,5], full_corners_img[:,6], full_corners_img[:,7]], dtype='int32')], (255, 255, 255))

    clipped_frame = cv2.bitwise_and(frame, poly_frame_front)
    clipped_frame = cv2.bitwise_and(clipped_frame, poly_frame_back)
    return clipped_frame


def sharpness_focus(sf_state, af_pub, focus_state_sub, video_socket, focus_sub):

    if sf_state.MODE == 'COARSE':  # -> Focus Control
        print('MODE is COARSE')
        dist_to_tank = (300 - sf_state.stage_z) + p.STAGE_TANK_OFFSET
        ll_max = 2953.5*dist_to_tank**-0.729
        ll_min = 2953.5*(dist_to_tank + p.TANK_DEPTH_MM)**-0.729
        print('llmin, llmax: (%f, %f)' % (ll_min, ll_max))
        af_pub.send_pyobj(m.AutofocusMessage(ll_min, ll_max, 1))
        state = ''  # The liquid lens will broadcast whether it is FOCUSING or FIXED

        # We told the liquid lens to autofocus. Now we just need to wait for it to finish.
        for ix in range(5):
            try:
                state = focus_state_sub.recv_string()
            except zmq.Again:
                continue
            if state == 'FOCUSING':
                break
            if ix == 4:
                # TODO: Should probably handle this more gracefully, but for now if it doesn't go into focusing mode something is broken and should be addressed
                print('Camera not entering focus mode!')
                sys.exit(1)
        while state == 'FOCUSING':
            frame = recv_img(video_socket)
            cv2.imshow(p.VIDEO_WINDOW_NAME, frame)
            cv2.waitKey(1)
            state = focus_state_sub.recv_string()

        try:
            current_ll_focus = float(focus_sub.recv_string())
            print('Received focus %d' % current_ll_focus)
        except zmq.Again:
            current_ll_focus = None

        if current_ll_focus is not None:
            object_distance_ll = (current_ll_focus/2953.5)**(1.0/-0.729)  # (0-255) -> mm
            dz = object_distance_ll - p.FOCUS_DISTANCE_ZOOM
            print('Object distance_ll: %f' % object_distance_ll)
        else:
            dz = 0
        sf_state.mode = 'FINE_UNINITIALIZED'
        sf_state.stage_z_dir = 1

    # UNINITIALIZED - We tell the stage to move to the pre-sweep position
    elif sf_state.mode == 'FINE_UNINITIALIZED':
        print('MODE is FINE_UNINITIALIZED')
        if p.BYPASS_LL_ESTIMATE:
            dist_to_tank = (300 - sf_state.stage_z) + p.STAGE_TANK_OFFSET
            dz = dist_to_tank - p.FOCUS_DISTANCE_ZOOM  # This should place macro focus at near edge of tank
        else:
            dz = sf_state.object_distance_ll - p.FOCUS_DISTANCE_ZOOM - 15
        sf_state.sweep_lowerbound_abs = sf_state.stage_z + dz
        sf_state.mode = 'MOVING_LOWER_BOUND'

    # MOVING_LOWER_BOUND - We wait until the stage has reached pre-sweep position
    elif sf_state.mode == 'MOVING_LOWER_BOUND':
        print('MODE is FINE -> MOVING_LOWER_BOUND')
        dz = 0
        if abs(sf_state.stage_z - sf_state.sweep_lowerbound_abs) < 0.1 and not sf_state.z_moving:
            if p.BYPASS_LL_ESTIMATE:
                dz = p.TANK_DEPTH_MM
            else:
                dz = 30.0
            sf_state.stage_sweep_end = sf_state.stage_z + dz
            sf_state.best_sharpness = 0
            sf_state.best_sharpness_location = 0
            sf_state.mode = 'SWEEPING_ROI'

        # SWEEPING_ROI - We wait while the stage sweeps focus through the ROI, tracking best sharpness
    elif sf_state.mode == 'SWEEPING_ROI':
        print('MODE is FINE -> SWEEPING_ROI')
        # don't order any z motion, but check if z stopped
        if abs(sf_state.stage_z - sf_state.sweep_lowerbound_abs) > 1 and sf_state.z_moving:
            dz = 0
        print('%.3f / %.3f' % (sf_state.macro_sharpness, sf_state.best_sharpness))
        if sf_state.macro_sharpness > sf_state.best_sharpness:
            sf_state.best_sharpness = sf_state.macro_sharpness
            sf_state.best_sharpness_location = sf_state.stage_z
        if abs(sf_state.stage_z - sf_state.stage_sweep_end) < 0.1 and not sf_state.z_moving:
            dz = sf_state.best_sharpness_location - sf_state.stage_z
            print(dz)
            sf_state.mode = 'MOVING_TO_PEAK'

    # MOVING_TO_PEAK - We wait until the stage is close to the best sharpness location
    elif sf_state.mode == 'MOVING_TO_PEAK':
        print('MODE is FINE -> MOVING_TO_PEAK')
        if abs(sf_state.stage_z - sf_state.best_sharpness_location) < 0.1:
            sf_state.mode = 'INITIALIZED'
            if p.BYPASS_LL_ESTIMATE:
                p.BYPASS_LL_ESTIMATE = False or p.BYPASS_LL_ESTIMATE
        dz = 0

    # INITIALIZED - Our stage sweep gave us an estimate of where the best focus for the object is. Once
    # we get to that point, hopefully we can keep it in focus by moving the stage to maintain focus
    # based on the sharpness gradient.
    elif sf_state.mode == 'INITIALIZED':
        print('MODE is FINE -> INITIALIZED')
        if sf_state.macro_sharpness < sf_state.macro_sharpness_last:
            sf_state.stage_z_dir = -1 * sf_state.stage_z_dir
        dz = 3 * sf_state.stage_z_dir

        # We have an invalid mode, which should never happen and it means there's an issue with the code
    else:
        print('fine_submode in illegal state %s!' % sf_state.mode)
        sys.exit(1)

    return dz, sf_state


def filter_target_position(target_pos, target_pos_slow, target_pos_obs):
    if target_pos_obs is not None and np.linalg.norm(target_pos_obs - target_pos) < p.TARGET_JUMP_THRESH:
        # (we threw out anything that's *really* off, now we are going to low pass filter
        # we'll just do IIR because the jump threshold will prevent crazy outliers
        target_pos = target_pos * p.LP_IIR_DECAY + target_pos_obs * (1 - p.LP_IIR_DECAY)
        target_pos_slow = target_pos_slow * p.LP_IIR_DECAY_2 + target_pos_obs * (1 - p.LP_IIR_DECAY_2)
        target_track_ok = True
    else:
        target_track_ok = False

    return target_track_ok, target_pos, target_pos_slow


def calculate_movement_offsets(frame, target_pos, target_pos_slow, feature_delta):

    cv2.circle(frame, (int(target_pos[0]), int(target_pos[1])), 75, (255, 0, 0), 3)
    cv2.circle(frame, (int(target_pos_slow[0] + feature_delta[0]), int(target_pos_slow[1] + feature_delta[1])), 5, (0, 255, 0), -1)
    dx = (target_pos_slow[0] + feature_delta[0]) - (p.IMG_DISP_WIDTH_SPOTTER / 2 + p.MACRO_FOV_OFFSET[0])
    dy = (target_pos_slow[1] + feature_delta[1]) - (p.IMG_DISP_HEIGHT_SPOTTER / 2 + p.MACRO_FOV_OFFSET[1])

    return dx, dy


def manual_focus_update():
    delta = 0
    if keyboard.is_pressed('d'):
        print('zoom out')
        delta = -0.1
    if keyboard.is_pressed('f'):
        print('zoom in')
        delta = 0.1

    if keyboard.is_pressed('alt'):
        print('Super Zoom!')
        modifier = 100
    else:
        modifier = 1
    return delta * modifier


def get_feature_2delta():
    feature_2delta = np.array([0, 0])
    if keyboard.is_pressed('up') or keyboard.is_pressed('k'):
        print('up')
        feature_2delta[1] -= p.ARROW_MOVE_RATE
    if keyboard.is_pressed('down') or keyboard.is_pressed('j'):
        print('down')
        feature_2delta[1] += p.ARROW_MOVE_RATE
    if keyboard.is_pressed('left') or keyboard.is_pressed('h'):
        print('left')
        feature_2delta[0] -= p.ARROW_MOVE_RATE
    if keyboard.is_pressed('right') or keyboard.is_pressed('l'):
        print('right')
        feature_2delta[0] += p.ARROW_MOVE_RATE

    return feature_2delta


def reset_target_selection():
    target_pos = np.array((cvui.mouse().x/p.DISPLAY_RESCALE_SPOTTER, cvui.mouse().y/p.DISPLAY_RESCALE_SPOTTER))
    feature_pos = np.array((cvui.mouse().x/p.DISPLAY_RESCALE_SPOTTER, cvui.mouse().y/p.DISPLAY_RESCALE_SPOTTER))
    feature_delta = feature_pos - target_pos
    return target_pos, feature_delta


def setup_zmq(context):

    # Receive video frames from camera
    video_socket = context.socket(zmq.SUB)
    video_socket.setsockopt(zmq.CONFLATE, 1)
    video_socket.setsockopt(zmq.RCVTIMEO, 1000)
    video_socket.connect('tcp://%s:%d' % (p.VIDEO_SPOTTER_IP, p.VIDEO_SPOTTER_PORT))
    topicfilter = ''
    video_socket.setsockopt_string(zmq.SUBSCRIBE, topicfilter)

    # Receive focus updates from LL camera
    focus_sub = context.socket(zmq.SUB)
    focus_sub.setsockopt(zmq.CONFLATE, 1)
    focus_sub.setsockopt(zmq.RCVTIMEO, 500)
    focus_sub.connect('tcp://%s:%d' % (p.CURRENT_FOCUS_IP, p.CURRENT_FOCUS_PORT))
    topicfilter = ''
    focus_sub.setsockopt_string(zmq.SUBSCRIBE, topicfilter)

    # Receive updates from stage
    stage_sub = context.socket(zmq.SUB)
    stage_sub.setsockopt(zmq.CONFLATE, 1)
    stage_sub.setsockopt(zmq.RCVTIMEO, 0)
    stage_sub.connect('tcp://%s:%d' % (p.STAGE_POSITION_IP, p.STAGE_POSITION_PORT))
    topicfilter = ''
    stage_sub.setsockopt_string(zmq.SUBSCRIBE, topicfilter)

    # Receives autofocus state updates
    focus_state_sub = context.socket(zmq.SUB)
    focus_state_sub.setsockopt(zmq.CONFLATE, 1)
    focus_state_sub.setsockopt(zmq.RCVTIMEO, 500)
    focus_state_sub.connect('tcp://%s:%d' % (p.SPOTTER_FOCUS_STATE_IP, p.SPOTTER_FOCUS_STATE_PORT))
    topicfilter = ''
    focus_state_sub.setsockopt_string(zmq.SUBSCRIBE, topicfilter)

    # Receives autofocus state updates
    macro_sharpness_sub = context.socket(zmq.SUB)
    macro_sharpness_sub.setsockopt(zmq.CONFLATE, 1)
    macro_sharpness_sub.setsockopt(zmq.RCVTIMEO, 100)
    macro_sharpness_sub.connect('tcp://%s:%d' % (p.MACRO_SHARPNESS_IP, p.MACRO_SHARPNESS_PORT))
    topicfilter = ''
    macro_sharpness_sub.setsockopt_string(zmq.SUBSCRIBE, topicfilter)

    # Publish tracking deltas
    track_socket = context.socket(zmq.PUB)
    track_socket.bind('tcp://*:%s' % p.TRACK_PORT)

    # Publish ROI bounding box for focusing
    roi_socket = context.socket(zmq.PUB)
    roi_socket.bind('tcp://*:%s' % p.FOCUS_ROI_PORT)

    # Publish autofocus requests
    af_pub = context.socket(zmq.PUB)
    af_pub.bind('tcp://*:%s' % p.AUTOFOCUS_PORT)

    return video_socket, focus_sub, stage_sub, focus_state_sub, macro_sharpness_sub, track_socket, roi_socket, af_pub


def draw_settings(ctrl_frame, control_panes, canny_state, threshold_state, stage_mode, focus_mode, tracker_mode):

    # define the mutually-exclusive mode toggles

    stage_manual = [stage_mode == 'MANUAL']
    stage_auto = [stage_mode == 'AUTO']
    stage_paused = [stage_mode == 'PAUSED']

    focus_manual = [focus_mode == 'MANUAL']
    focus_sharpness = [focus_mode == 'SHARPNESS']
    focus_depth = [focus_mode == 'DEPTH']

    tracker_box_kcf = [tracker_mode == 'KCF']
    tracker_box_canny = [tracker_mode == 'CANNY']
    tracker_box_threshold = [tracker_mode == 'THRESHOLD']
    macro_resweep = False
    ll_resweep = False

    control_panes.stage_control_pane.begin(ctrl_frame)
    if not control_panes.stage_control_pane.isMinimized():
        cvui.space(100)  # add 20px of empty space
        cvui.checkbox('Stage Paused', stage_paused)
        cvui.checkbox('Stage Manual', stage_manual)
        cvui.checkbox('Stage Auto', stage_auto)
    control_panes.stage_control_pane.end()

    control_panes.focus_control_pane.begin(ctrl_frame)
    if not control_panes.focus_control_pane.isMinimized():
        cvui.space(80)  # add 20px of empty space
        cvui.checkbox('Focus Manual', focus_manual)
        cvui.checkbox('Focus Sharpness', focus_sharpness)
        cvui.checkbox('Focus Depth', focus_depth)
    control_panes.focus_control_pane.end()

    control_panes.tracker_select_pane.begin(ctrl_frame)
    if not control_panes.tracker_select_pane.isMinimized():
        cvui.space(60)  # add 20px of empty space
        cvui.checkbox('KCF Tracker', tracker_box_kcf)
        cvui.checkbox('Canny Tracker', tracker_box_canny)
        cvui.checkbox('Threshold Tracker', tracker_box_threshold)
    control_panes.tracker_select_pane.end()

    control_panes.canny_settings_pane.begin(ctrl_frame)
    if not control_panes.canny_settings_pane.isMinimized():
        cvui.space(40)  # add 20px of empty space
        cvui.text('Canny Low Threshold')
        cvui.trackbar(control_panes.canny_settings_pane.width() - 20, canny_state.canny_low, 5, 150)
        cvui.text('Canny High Threshold')
        cvui.trackbar(control_panes.canny_settings_pane.width() - 20, canny_state.canny_high, 80, 300)
    control_panes.canny_settings_pane.end()

    control_panes.threshold_setting_pane.begin(ctrl_frame)
    if not control_panes.threshold_setting_pane.isMinimized():
        cvui.space(20)  # add 20px of empty space
        cvui.text('Binarization Threshold')
        cvui.trackbar(control_panes.threshold_setting_pane.width() - 20, threshold_state.threshold, 0, 255)
        cvui.checkbox('Show Binary Image', threshold_state.show_binary)
        cvui.text('MOVE THESE!!!!')
        if cvui.button('Force Macro Focus Sweep'):
            macro_resweep = True
        else:
            macro_resweep = False
        if cvui.button('Liquid Lens Focus Sweep'):
            ll_resweep = True
        else:
            ll_resweep = False
    control_panes.threshold_setting_pane.end()

    if stage_manual[0] and not stage_mode == 'MANUAL':
        stage_mode = 'MANUAL'
    elif stage_auto[0] and not stage_mode == 'AUTO':
        stage_mode = 'AUTO'
    elif stage_paused[0] and not stage_mode == 'PAUSED':
        stage_mode = 'PAUSED'

    if focus_manual[0] and not focus_mode == 'MANUAL':
        focus_mode = 'MANUAL'
    elif focus_sharpness[0] and not focus_mode == 'SHARPNESS':
        focus_mode = 'SHARPNESS'
    elif focus_depth[0] and not focus_mode == 'DEPTH':
        focus_mode = 'DEPTH'

    if tracker_box_kcf[0] and not tracker_mode == 'KCF':
        tracker_mode = 'KCF'
    elif tracker_box_canny[0] and not tracker_mode == 'CANNY':
        tracker_mode = 'CANNY'
    elif tracker_box_threshold and not tracker_mode == 'THRESHOLD':
        tracker_mode = 'THRESHOLD'

    return stage_mode, focus_mode, tracker_mode, macro_resweep, ll_resweep


def recv_img(socket, flags=0, copy=True, track=False):
    """recv a numpy array"""
    msg = socket.recv(flags=flags, copy=copy, track=track)
    buf = memoryview(msg)
    A = np.frombuffer(buf, np.uint8)
    A = A.copy()
    return A.reshape((p.IMG_HEIGHT_SPOTTER, p.IMG_WIDTH_SPOTTER, 3))


def sigint_handler(signo, stack_frame):
    global keep_running
    keep_running = False


def main():

    global keep_running

    # This is for saving video *with* detection boxes on it
    # To save raw video, use the CameraSaver.py script
    save_video = True
    if save_video:
        sz = (p.IMG_WIDTH_SPOTTER, p.IMG_HEIGHT_SPOTTER)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        vout = cv2.VideoWriter()
        vout.open('track_output.mp4', fourcc, p.FPS_SPOTTER, sz, False)

    signal.signal(signal.SIGINT, sigint_handler)

    control_panes = ControlPanes()
    control_panes.stage_control_pane = EnhancedWindow(0, 0, 300, 500, 'Stage Control')
    control_panes.focus_control_pane = EnhancedWindow(0, 20, 300, 500, 'Focus Control')
    control_panes.tracker_select_pane = EnhancedWindow(0, 40, 300, 500, 'Tracker Select')
    control_panes.canny_settings_pane = EnhancedWindow(0, 60, 300, 500, 'Canny Tuning')
    control_panes.threshold_setting_pane = EnhancedWindow(0, 80, 300, 500, 'Threshold Tuning')

    cvui.init(p.CTRL_WINDOW_NAME)
    cvui.init(p.VIDEO_WINDOW_NAME)

    context = zmq.Context()
    (video_socket, focus_sub, stage_sub, focus_state_sub, macro_sharpness_sub, track_socket, roi_socket, af_pub) = setup_zmq(context)


    stage_zero_offset = np.load('tank_corners_offset.npy')
    world_points = np.load('tank_corners.npy')
    intrinsic = np.load('intrinsic_calibration/ll_65/intrinsic.npy')

    stage_x = None
    stage_y = None
    stage_z = None
    z_moving = True
    current_ll_focus = None
    object_distance_ll = 0

    target_pos_obs = None
    target_pos = np.array([1, 1])
    target_pos_slow = target_pos.copy()
    feature_delta = np.array([0, 0])
    target_track_init = False
    STAGE_MODE = 'PAUSED'
    FOCUS_MODE = 'MANUAL'
    tracker_type = 'KCF'  # options are KCF or CANNY

    # These three structs store the state information necessary for the trackers
    canny_tracker_state = CannyTracker()
    canny_tracker_state.canny_low = [50]
    canny_tracker_state.canny_high = [150]

    kcf_tracker_state = KCFTracker()
    kcf_tracker_state.kcf_box_anchor = cvui.Point()
    kcf_tracker_state.kcf_roi = cvui.Rect(0, 0, 0, 0)
    kcf_tracker_state.kcf_tracker_init = False

    threshold_tracker_state = ThresholdTracker()
    threshold_tracker_state.threshold = [30]
    threshold_tracker_state.roi = cvui.Rect(0, 0, 0, 0)
    threshold_tracker_state.box_anchor = cvui.Point
    threshold_tracker_state.show_binary = [False]

    sharpness_focus_state = SharpnessFocusState()
    sharpness_focus_state.mode = 'COARSE'
    macro_sharpness = 0

    while keep_running:
        ctrl_frame = np.zeros((700, 300, 3), np.uint8)

        # Receive stage position updates
        try:
            stage_pos = stage_sub.recv_string()
            (stage_x, stage_y, stage_z_new) = [float(x) for x in stage_pos.split(' ')]
            if stage_z_new == stage_z:
                z_moving = False
            else:
                z_moving = True
            stage_z = stage_z_new
        except zmq.Again:
            # the stage publisher only publishes at ~10hz, so not having an update is common
            pass

        # Receive macro sharpness
        try:
            macro_sharpness_last = macro_sharpness
            macro_sharpness = float(macro_sharpness_sub.recv_string())
        except zmq.Again:
            # no sharpness value, which is unexpected
            print('No Macro Image Sharpness!')

        # receive next frame
        try:
            frame = recv_img(video_socket)
        except zmq.Again:
            print('Timed Out!')
            time.sleep(1)
            continue

        cvui.context(p.VIDEO_WINDOW_NAME)
        if cvui.mouse(cvui.IS_DOWN):
            (target_pos, feature_delta) = reset_target_selection()
            target_pos_slow = target_pos.copy()
            target_track_init = True
        feature_delta += get_feature_2delta()

        if stage_x is not None:
            stage_pos = np.array([stage_x, stage_y, -stage_z], ndmin=2).T
            frame = tank_corners_clip(frame, stage_pos, stage_zero_offset, world_points, intrinsic)

        # This is where the tracking happens. tracker_type is controlled by a button on the interface
        # Adding a new tracker is as simple as adding another case to this if/else and adding a button in
        # the UI to switch into the new tracking mode
        if tracker_type == 'CANNY':
            canny_tracker_state.target_pos = target_pos
            (target_pos_obs, roi, canny_tracker_state) = update_canny_tracker(frame, canny_tracker_state)

        elif tracker_type == 'KCF':
            cvui.context(p.VIDEO_WINDOW_NAME)
            (target_pos_obs, roi, kcf_tracker_state) = update_kcf_tracker(frame, kcf_tracker_state)

        elif tracker_type == 'THRESHOLD':
            cvui.context(p.VIDEO_WINDOW_NAME)
            threshold_tracker_state.target_pos = target_pos
            (target_pos_obs, roi, threshold_tracker_state) = update_threshold_tracker(frame, threshold_tracker_state)

        else:
            print('Invalid tracker mode: %s' % tracker_type)
            roi = None
            keep_running = False

        # This roi_msg takes an roi that may have been identified around the animal and sends it over zmq
        # This enables any cameras trying to autofocus to know which roi to keep in focus
        # if no autofocusing is happening, then these messages don't do anything
        if roi is not None:
            roi_msg = m.SetFocusROI(roi[0], roi[1])
        else:
            roi_msg = m.SetFocusROI(None, None)
        roi_socket.send_pyobj(roi_msg)  # tell the LL camera (or anything else I guess) which ROI to focus


        (target_track_ok, target_pos, target_pos_slow) = filter_target_position(target_pos, target_pos_slow, target_pos_obs)

        # This is probably where we want to use the other camera to estimate depth

        # Now we have a giant state machine. We need to structure the code this way, because we want 2D tracking and
        # user interaction to update even when we are waiting on some slower action to occur related to object depth
        # and focusing. The state machine provides a mechanism to handle these slower processes while not impeding the
        # rest of the tracking process.

        # STAGE_MODE = {MANUAL | AUTO | PAUSED}
        #   -- In MANUAL mode, dx,dy,dz all set by keyboard input.
        #   -- In AUTO mode, dx and dy are set by tracker. dz is set by autofocus if FOCUS_MODE is set to AUTO
        #   -- In PAUSED mode, dx = dy = dz = 0. The tracker will keep tracking, but the stage won't move
        #
        # FOCUS_MODE = {MANUAL | SHARPNESS | DEPTH}
        #   -- In MANUAL mode, dz is set by keyboard input
        #   -- In SHARPNESS mode, dz is set by trying to maximize sharpness, although the final position can be tweaked
        #      by user input. SHARPNESS mode does nothing if STAGE_MODE is MANUAL
        #   -- In DEPTH mode, dz is set by a target depth measurement that is estimated from a second camera
        #      (stereo or perpendicular)

        # Determine dx and dy
        if STAGE_MODE == 'PAUSED':  # -> Stage Control
            track_socket.send_string('0 0 0')
            dx = 0
            dy = 0
            dz = 0
        elif STAGE_MODE == 'MANUAL':  # TODO: Probably tune this better
            (dx, dy) = get_feature_2delta()
            dx = 10 * dx
            dy = 10 * dy
            print('FULL_MANUAL %f, %f' % (dx, dy))
            dz = manual_focus_update()
        elif STAGE_MODE == 'AUTO':
            # The tracker makes a determination in pixel space, then we may decide to filter it. We then determine the
            # dx and dy based on the distance between the feature of interest and the macro lens center
            # how much do we need to move in pixel-space?
            # Note dx and dy are 0 if there are no target tracks

            if stage_z is None:
                print('Waiting on stage node')
                dx = 0
                dy = 0
                dz = 0
            else:
                if target_pos_obs is not None:
                    if target_track_ok:
                        (dx, dy) = calculate_movement_offsets(frame, target_pos, target_pos_slow, feature_delta)
                    else:
                        dx = 0
                        dy = 0
                else:
                    dx = 0
                    dy = 0
                    target_track_ok = False

                # When STAGE_MODE == 'AUTO', we need to determine how to handle the focusing
                if FOCUS_MODE == 'MANUAL':
                    dz = manual_focus_update()
                elif FOCUS_MODE == 'SHARPNESS':

                    sharpness_focus_state.stage_z = stage_z
                    sharpness_focus_state.macro_sharpness = macro_sharpness
                    sharpness_focus_state.z_moving = z_moving
                    dz, sharpness_focus_state = sharpness_focus(sharpness_focus_state, af_pub, focus_state_sub, video_socket, focus_sub)
                elif FOCUS_MODE == 'DEPTH':
                    # this is the mode when we have a second camera to estimate depth
                    dz = 0
                else:
                    # invalid focus mode
                    print('Invalid focus mode %s' % FOCUS_MODE)
                    sys.exit(1)
        else:
            print('Unknown stage mode: %s' % STAGE_MODE)
            dx = 0
            dy = 0
            dz = 0

        print(dx, dy, dz)
        track_socket.send_string('%f %f %f' % (dx, dy, dz))  # 'wasteful', but easier debugging for now

        frame = cv2.resize(frame, (p.IMG_DISP_WIDTH_SPOTTER, p.IMG_DISP_HEIGHT_SPOTTER))

        # draw dots on frame centers
        cv2.circle(frame, (int(p.IMG_DISP_WIDTH_SPOTTER / 2), int(p.IMG_DISP_HEIGHT_SPOTTER / 2)), 5, (0, 0, 255), -1)  # center of frame
        cv2.circle(frame, (p.MACRO_LL_CENTER[0], p.MACRO_LL_CENTER[1]), 5, (255, 0, 255), -1)  # center of macro frame frame

        cvui.update(p.VIDEO_WINDOW_NAME)
        cv2.imshow(p.VIDEO_WINDOW_NAME, frame)
        if save_video:
            vout.write(frame)

        cvui.context(p.CTRL_WINDOW_NAME)
        STAGE_MODE, FOCUS_MODE, tracker_type, macro_resweep, ll_resweep = draw_settings(ctrl_frame, control_panes, canny_tracker_state, threshold_tracker_state, STAGE_MODE, FOCUS_MODE, tracker_type)

        if macro_resweep:
            p.BYPASS_LL_ESTIMATE = True
            sharpness_focus_state.mode = 'FINE_UNINITIALIZED'

        if ll_resweep:
            if stage_z is not None:
                print('Liquid Lens Refocus!')
                dist_to_tank = (300 - stage_z) + p.STAGE_TANK_OFFSET
                ll_max = 2953.5*dist_to_tank**-0.729
                ll_min = 2953.5*(dist_to_tank + p.TANK_DEPTH_MM)**-0.729
                print('llmin, llmax: (%f, %f)' % (ll_min, ll_max))
                af_pub.send_pyobj(m.AutofocusMessage(ll_min, ll_max, 1))
            else:
                print('Cannot refocus liquid lens until stage node is running')

        cvui.update(p.CTRL_WINDOW_NAME)
        cv2.imshow(p.CTRL_WINDOW_NAME, ctrl_frame)
        cv2.waitKey(1)

    if save_video:
        vout.release()


if __name__ == '__main__':
    main()
