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

keep_running = True
BYPASS_LL_ESTIMATE = False


def calculate_movement_offsets(frame, target_pos, target_pos_obs, target_pos_slow, feature_delta):

    if np.linalg.norm(target_pos_obs - target_pos) < p.TARGET_JUMP_THRESH:
        # (we threw out anything that's *really* off, now we are going to low pass filter
        # we'll just do IIR because the jump threshold will prevent crazy outliers
        target_pos = target_pos * p.LP_IIR_DECAY + target_pos_obs * (1 - p.LP_IIR_DECAY)
        target_pos_slow = target_pos_slow * p.LP_IIR_DECAY_2 + target_pos_obs * (1 - p.LP_IIR_DECAY_2)
        cv2.circle(frame, (int(target_pos[0]), int(target_pos[1])), 75, (255, 0, 0), 3)
        cv2.circle(frame, (int(target_pos_slow[0] + feature_delta[0]), int(target_pos_slow[1] + feature_delta[1])), 5, (0, 255, 0), -1)
        dx = (target_pos_slow[0] + feature_delta[0]) - (p.IMG_DISP_WIDTH_SPOTTER / 2 + p.MACRO_FOV_OFFSET[0])
        dy = (target_pos_slow[1] + feature_delta[1]) - (p.IMG_DISP_HEIGHT_SPOTTER / 2 + p.MACRO_FOV_OFFSET[1])
        target_track_ok = True  # only send commands to move the stage if we saw the target this frame
    else:
        target_track_ok = False  # don't command stage if no target track (prevents runaway stage behavior)
        dx = 0
        dy = 0

    return dx, dy, target_track_ok


def determine_roi(frame, contours, target_pos):
    bounding_rects = []
    br_centers = []
    for indx, c in enumerate(contours):
        if cv2.contourArea(c) > p.BBOX_AREA_THRESH:
            br = cv2.boundingRect(c)
            x, y, w, h = cv2.boundingRect(c)
            cv2.rectangle(frame, br, (0, 255, 0), 3)
            bounding_rects.append(br)
            br_centers.append((x + w/2, y + h/2))
    br_centers = np.array(br_centers)

    if len(br_centers) > 0:  # if we found any bounding boxes, determine their distances from target pt
        br_dists = np.linalg.norm(br_centers - target_pos, axis=1)
        best_bbox_ix = np.argmin(br_dists)
        target_pos_obs = br_centers[best_bbox_ix, :]
        x, y, w, h = bounding_rects[best_bbox_ix]
        bbox_ul = (x, y)
        bbox_lr = (x + w, y + h)
        roi_msg = m.SetFocusROI(bbox_ul, bbox_lr)
    else:
        roi_msg = m.SetFocusROI(None, None)  # Don't focus on any particular ROI
        target_pos_obs = None

    return br_centers, target_pos_obs, roi_msg


def process_contours(frame, canny_thresh_low, canny_thresh_high):
    frame_canny = apply_canny(cv2.GaussianBlur(frame, (0, 0), 3), canny_thresh_low, canny_thresh_high, 3)
    kernel = np.ones((5, 5), np.uint8)
    frame_canny = cv2.dilate(frame_canny, kernel, iterations=1)
    contours, _ = cv2.findContours(frame_canny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours


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


def draw_settings(ctrl_frame, settings, low_threshold, high_threshold, mode, tracker_mode):
    if mode == 'FULL_MANUAL':
        full_manual = [True]
        manual_focus = [False]
        coarse = [False]
        fine = [False]
        paused = [False]
    elif mode == 'COARSE':
        full_manual = [False]
        manual_focus = [False]
        coarse = [True]
        fine = [False]
        paused = [False]
    elif mode == 'FINE':
        full_manual = [False]
        manual_focus = [False]
        coarse = [False]
        fine = [True]
        paused = [False]
    elif mode == 'MANUAL_FOCUS':
        full_manual = [False]
        manual_focus = [True]
        coarse = [False]
        fine = [False]
        paused = [False]
    else:
        full_manual = [False]
        manual_focus = [False]
        coarse = [False]
        fine = [False]
        paused = [True]

    tracker_box = [tracker_mode == 'KCF']

    settings.begin(ctrl_frame)
    if not settings.isMinimized():
        cvui.trackbar(settings.width() - 20, low_threshold, 5, 150)
        cvui.trackbar(settings.width() - 20, high_threshold, 80, 300)
        cvui.space(20)  # add 20px of empty space
        cvui.checkbox('Full Manual', full_manual)
        cvui.checkbox('Manual Focus', manual_focus)
        cvui.checkbox('Coarse AutoFocus', coarse)
        cvui.checkbox('Fine AutoFocus', fine)
        cvui.checkbox('Paused', paused)
        cvui.checkbox('KCF Tracker', tracker_box)
        if cvui.button('Force Macro Focus Sweep'):
            macro_resweep = True
        else:
            macro_resweep = False

        if cvui.button('Liquid Lens Focus Sweep'):
            ll_resweep = True
        else:
            ll_resweep = False
    settings.end()

    # PAUSED -> {COARSE | FULL_MANUAL | MANUAL_FOCUS}
    if mode == 'PAUSED':
        if coarse[0]:
            mode = 'COARSE'
        elif full_manual[0]:
            mode = 'FULL_MANUAL'
        elif manual_focus[0]:
            mode = 'MANUAL_FOCUS'
    # COARSE -> {PAUSED | FULL_MANUAL | MANUAL_FOCUS}
    elif mode == 'COARSE':
        if paused[0]:
            mode = 'PAUSED'
        elif full_manual[0]:
            mode = 'FULL_MANUAL'
        elif manual_focus[0]:
            mode = 'MANUAL_FOCUS'
    # FINE -> {PAUSED | COARSE | FULL_MANUAL | MANUAL_FOCUS}
    elif mode == 'FINE':
        if paused[0]:
            mode = 'PAUSED'
        elif coarse[0]:
            mode = 'COARSE'
        elif full_manual[0]:
            mode = 'FULL_MANUAL'
        elif manual_focus[0]:
            mode = 'MANUAL_FOCUS'
    # MANUAL_FOCUS -> {PAUSED | COARSE | FULL_MANUAL}
    elif mode == 'MANUAL_FOCUS':
        if paused[0]:
            mode = 'PAUSED'
        elif coarse[0]:
            mode = 'COARSE'
        elif full_manual[0]:
            mode = 'FULL_MANUAL'
    # MANUAL_FOCUS -> {FULL_MANUAL | MANUAL_FOCUS | PAUSED}
    elif mode == 'FULL_MANUAL':
        if paused[0]:
            mode = 'PAUSED'
        elif coarse[0]:
            mode = 'COARSE'
        elif manual_focus[0]:
            mode = 'MANUAL_FOCUS'

    if tracker_box[0]:
        tracker_type = 'KCF'
    else:
        tracker_type = 'CANNY'

    return mode, tracker_type, macro_resweep, ll_resweep


def apply_canny(frame, high, low, kernel):
    frame_ret = cv2.Canny(frame, low, high, kernel)
    return frame_ret


def recv_img(socket, flags=0, copy=True, track=False):
    """recv a numpy array"""
    msg = socket.recv(flags=flags, copy=copy, track=track)
    buf = memoryview(msg)
    A = np.frombuffer(buf, np.uint8)
    return A.reshape((p.IMG_HEIGHT_SPOTTER, p.IMG_WIDTH_SPOTTER, 3))


def sigint_handler(signo, stack_frame):
    global keep_running
    keep_running = False


def roi_tool(frame, anchor, roi):
    # The function 'bool cvui.mouse(int query)' allows you to query the mouse for events.
    # E.g. cvui.mouse(cvui.DOWN)
    #
    # Available queries:
    #	- cvui.DOWN: any mouse button was pressed. cvui.mouse() returns true for single frame only.
    #	- cvui.UP: any mouse button was released. cvui.mouse() returns true for single frame only.
    #	- cvui.CLICK: any mouse button was clicked (went down then up, no matter the amount of frames in between). cvui.mouse() returns true for single frame only.
    #	- cvui.IS_DOWN: any mouse button is currently pressed. cvui.mouse() returns true for as long as the button is down/pressed.

    drawing = False
    new_bb = False

    # Did any mouse button go down?
    if cvui.mouse(cvui.DOWN):
        # Position the anchor at the mouse pointer.
        anchor.x = cvui.mouse().x / p.DISPLAY_RESCALE_TRACK_SPOTTER
        anchor.y = cvui.mouse().y / p.DISPLAY_RESCALE_TRACK_SPOTTER

        # Inform we are working, so the ROI window is not updated every frame
        drawing = True


    # Is any mouse button down (pressed)?
    if cvui.mouse(cvui.IS_DOWN):
        # Adjust roi dimensions according to mouse pointer
        width = cvui.mouse().x / p.DISPLAY_RESCALE_TRACK_SPOTTER - anchor.x
        height = cvui.mouse().y / p.DISPLAY_RESCALE_TRACK_SPOTTER - anchor.y

        roi.x = anchor.x + width if width < 0 else anchor.x
        roi.y = anchor.y + height if height < 0 else anchor.y
        roi.width = abs(width)
        roi.height = abs(height)

        # Show the roi coordinates and size
        cvui.printf(frame, roi.x + 5, roi.y + 5, 0.3, 0xff0000, '(%d,%d)', roi.x, roi.y)
        cvui.printf(frame, cvui.mouse().x / p.DISPLAY_RESCALE_TRACK_SPOTTER + 5, cvui.mouse().y / p.DISPLAY_RESCALE_TRACK_SPOTTER + 5, 0.3, 0xff0000, 'w:%d, h:%d', roi.width, roi.height)

    # Was the mouse clicked (any button went down then up)?
    if cvui.mouse(cvui.UP):
        # We are done working with the ROI.
        drawing = False
        if roi.width > 25 and roi.height > 25:
            new_bb = True

    # Ensure ROI is within bounds
    frameRows, frameCols, frameChannels = frame.shape
    roi.x = 0 if roi.x < 0 else roi.x
    roi.y = 0 if roi.y < 0 else roi.y
    roi.width = roi.width + frame.cols - (roi.x + roi.width) if roi.x + roi.width > frameCols else roi.width
    roi.height = roi.height + frame.rows - (roi.y + roi.height) if roi.y + roi.height > frameRows else roi.height

    # Render the roi
    cvui.rect(frame, roi.x, roi.y, roi.width, roi.height, 0xff0000)

    return anchor, roi, drawing, new_bb


def main():

    global keep_running
    global BYPASS_LL_ESTIMATE

    # This is for saving video *with* detection boxes on it
    # To save raw video, use the CameraSaver.py script
    save_video = False
    if save_video:
        sz = (p.IMG_WIDTH_SPOTTER, p.IMG_HEIGHT_SPOTTER)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        vout = cv2.VideoWriter()
        vout.open('track_output.mp4', fourcc, p.FPS_SPOTTER, sz, False)

    signal.signal(signal.SIGINT, sigint_handler)

    ctrl_frame = np.zeros((400, 300, 3), np.uint8)
    settings = EnhancedWindow(10, 50, 270, 320, 'Settings')
    control = EnhancedWindow(10, 300, 270, 100, 'Control')
    cvui.init(p.CTRL_WINDOW_NAME)
    cvui.init(p.VIDEO_WINDOW_NAME)

    context = zmq.Context()
    (video_socket, focus_sub, stage_sub, focus_state_sub, macro_sharpness_sub, track_socket, roi_socket, af_pub) = setup_zmq(context)

    stage_x = None
    stage_y = None
    stage_z = None
    z_moving = True
    current_ll_focus = None
    object_distance_ll = 0
    low_threshold = [50]
    high_threshold = [150]
    target_pos = np.array([1, 1])
    target_pos_slow = target_pos.copy()
    feature_delta = np.array([0, 0])
    target_track_init = False
    MODE = 'PAUSED'
    fine_submode = 'UNINITIALIZED'
    tracker_type = 'KCF'  # options are KCF or CANNY
    kcf_box_anchor = cvui.Point()
    kcf_roi = cvui.Rect(0, 0, 0, 0)
    kcf_tracker_init = False
    target_pos_obs = None

    macro_sharpness = 0

    while keep_running:

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
        frame = cv2.resize(frame, (1920, 1080), interpolation=cv2.INTER_LINEAR)

        if tracker_type == 'CANNY':
            # This is what is required to update the track based on bounding box detection
            ############################################################################################################
            contours = process_contours(frame, low_threshold[0], high_threshold[0])

            cvui.context(p.VIDEO_WINDOW_NAME)
            if cvui.mouse(cvui.IS_DOWN):
                (target_pos, feature_delta) = reset_target_selection()
                target_pos_slow = target_pos.copy()
                target_track_init = True

            feature_delta += get_feature_2delta()

            (br_centers, target_pos_obs, roi_msg) = determine_roi(frame, contours, target_pos)
            ############################################################################################################

        elif tracker_type == 'KCF':
            # kcf code
            cvui.context(p.VIDEO_WINDOW_NAME)
            anchor, kcf_roi, drawing, new_bb = roi_tool(frame, kcf_box_anchor, kcf_roi)
            if drawing:
                kcf_tracker_init = False
            print('drawing: %d' % drawing)
            print('new_bb: %d' % new_bb)
            roi_msg = m.SetFocusROI(None, None)
            if new_bb:
                kcf_tracker_init = True
                kcf_tracker = cv2.TrackerKCF_create()
                kcf_tracker.init(frame, (kcf_roi.x, kcf_roi.y, kcf_roi.width, kcf_roi.height))
            elif kcf_tracker_init:
                ok, new_roi = kcf_tracker.update(frame)
                x1 = kcf_roi.x
                y1 = kcf_roi.y
                w = kcf_roi.width
                h = kcf_roi.height
                target_pos_obs = np.array([x1 + w/2., y1 + h/2])
                kcf_roi = cvui.Rect(new_roi[0], new_roi[1], new_roi[2], new_roi[3])
                roi_msg = m.SetFocusROI((x1, y1), (x1 + w, y1 + h))
        else:
            print('Invalid tracker mode: %s' % tracker_type)
            keep_running = False

        roi_socket.send_pyobj(roi_msg)  # tell the LL camera which ROI to focus

        # The tracker makes a determination in pixel space, then we may decide to filter it. We then determine the
        # dx and dy based on the distance between the feature of interest and the macro lens center
        # how much do we need to move in pixel-space?
        # Note dx and dy are 0 if there are no target tracks
        if target_pos_obs is not None:
            (dx, dy, target_track_ok) = calculate_movement_offsets(frame, target_pos, target_pos_obs, target_pos_slow, feature_delta)
        else:
            dx = 0
            dy = 0
            target_track_ok = False

        # Now we have a giant state machine. We need to structure the code this way, because we want 2D tracking and
        # user interaction to update even when we are waiting on some slower action to occur related to object depth
        # and focusing. The state machine provides a mechanism to handle these slower processes while not impeding the
        # rest of the tracking process.
        #
        # There are 3 major states: PAUSED, COARSE, and FINE
        # In the PAUSED state, none of the stages will move, and no LL focusing happens
        # In the COARSE state, The liquid lens sweeps focus through the tank to determine a coarse depth estimate via depth-from-focus
        # In the FINE state, the stage makes adjustments based on the macro lens to keep the object in focus
        # The FINE state has a second state machine within it, which is described within the FINE branch below.
        if MODE == 'PAUSED':
            track_socket.send_string('0 0 0')
            dx = 0
            dy = 0
            dz = 0
        elif MODE == 'MANUAL_FOCUS':
            dz = manual_focus_update()
        elif MODE == 'FULL_MANUAL':
            (dx, dy) = get_feature_2delta()
            dx = 10 * dx
            dy = 10 * dy
            print('FULL_MANUAL %f, %f' % (dx, dy))
            dz = manual_focus_update()
        elif MODE == 'COARSE':
            print('MODE is COARSE')
            if stage_z is None:
                print('Cannot continue until stage node is up')
                dz = 0
            else:
                dist_to_tank = (300 - stage_z) + p.STAGE_TANK_OFFSET
                ll_max = 2953.5*dist_to_tank**-0.729
                ll_min = 2953.5*(dist_to_tank + p.TANK_DEPTH_MM)**-0.729
                print('llmin, llmax: (%f, %f)' % (ll_min, ll_max))
                af_pub.send_pyobj(m.AutofocusMessage(ll_min, ll_max, 1))
                state = '' # The liquid lens will broadcast whether it is FOCUSING or FIXED
                for ix in range(5):
                    try:
                        state = focus_state_sub.recv_string()
                    except zmq.Again:
                        continue
                    if state == 'FOCUSING':
                        break
                    if ix == 4:
                        print('Camera not entering focus mode!')
                        keep_running = False
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
                    #dz = object_distance_ll - p.FOCUS_DISTANCE_ZOOM
                    dz = 0
                    print('Object distance_ll: %f' % object_distance_ll)
                else:
                    dz = 0
                MODE = 'FINE'
                fine_submode = 'UNINITIALIZED'
                stage_z_dir = 1

        elif MODE == 'FINE':

            if target_track_ok and target_track_init:  # this means the track has been initialized
                # UNINITIALIZED - We tell the stage to move to the pre-sweep position
                if fine_submode == 'UNINITIALIZED':
                    print('MODE is FINE -> UNINITIALIZED')
                    if BYPASS_LL_ESTIMATE:
                        dist_to_tank = (300 - stage_z) + p.STAGE_TANK_OFFSET
                        dz = dist_to_tank - p.FOCUS_DISTANCE_ZOOM  # This should place macro focus at near edge of tank
                    else:
                        dz = object_distance_ll - p.FOCUS_DISTANCE_ZOOM - 15
                    sweep_lowerbound_abs = stage_z + dz
                    fine_submode = 'MOVING_LOWER_BOUND'

                # MOVING_LOWER_BOUND - We wait until the stage has reached pre-sweep position
                elif fine_submode == 'MOVING_LOWER_BOUND':
                    print('MODE is FINE -> MOVING_LOWER_BOUND')
                    dz = 0
                    if abs(stage_z - sweep_lowerbound_abs) < 0.1 and not z_moving:
                        if BYPASS_LL_ESTIMATE:
                            dz = p.TANK_DEPTH_MM
                        else:
                            dz = 30.0
                        stage_sweep_end = stage_z + dz
                        best_sharpness = 0
                        best_sharpness_location = 0
                        fine_submode = 'SWEEPING_ROI'

                # SWEEPING_ROI - We wait while the stage sweeps focus through the ROI, tracking best sharpness
                elif fine_submode == 'SWEEPING_ROI':
                    print('MODE is FINE -> SWEEPING_ROI')
                    # don't order any z motion, but check if z stopped
                    if abs(stage_z - sweep_lowerbound_abs) > 1 and z_moving:
                        dz = 0
                    print('%.3f / %.3f' % (macro_sharpness, best_sharpness))
                    if macro_sharpness > best_sharpness:
                        best_sharpness = macro_sharpness
                        best_sharpness_location = stage_z
                    if abs(stage_z - stage_sweep_end) < 0.1 and not z_moving:
                        dz = best_sharpness_location - stage_z
                        print(dz)
                        fine_submode = 'MOVING_TO_PEAK'

                # MOVING_TO_PEAK - We wait until the stage is close to the best sharpness location
                elif fine_submode == 'MOVING_TO_PEAK':
                    print('MODE is FINE -> MOVING_TO_PEAK')
                    if abs(stage_z - best_sharpness_location) < 0.1:
                        fine_submode = 'INITIALIZED'
                        if BYPASS_LL_ESTIMATE:
                            BYPASS_LL_ESTIMATE = False or p.BYPASS_LL_ESTIMATE

                # INITIALIZED - Our stage sweep gave us an estimate of where the best focus for the object is. Once
                # we get to that point, hopefully we can keep it in focus by moving the stage to maintain focus
                # based on the sharpness gradient.
                elif fine_submode == 'INITIALIZED':
                    print('MODE is FINE -> INITIALIZED')
                    if macro_sharpness < macro_sharpness_last:
                        stage_z_dir = -1 * stage_z_dir
                    dz = 3 * stage_z_dir

                # We have an invalid mode, which should never happen and it means there's an issue with the code
                else:
                    print('fine_submode in illegal state %s!' % fine_submode)
                    sys.exit(1)
            else:
                print('No target track!')

        # In an invalid mode, so we should quit and fix the problem with the code
        else:
            print('Unknown MODE %s' % MODE)
            sys.exit(1)

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
        MODE, tracker_type, macro_resweep, ll_resweep = draw_settings(ctrl_frame, settings, low_threshold, high_threshold, MODE, tracker_type)
        if macro_resweep:
            BYPASS_LL_ESTIMATE = True
            MODE = 'FINE'
            fine_submode = 'UNINITIALIZED'

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
