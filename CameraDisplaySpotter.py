#!/usr/bin/python3
import time
import signal

import zmq
import cv2
import numpy as np
import keyboard

from EnhancedWindow import EnhancedWindow
import cvui
import Parameters as p
import Messages as m

keep_running = True


def calculate_movement_offsets(frame, centers, target_pos, target_pos_obs, target_pos_slow, feature_delta):

    if len(centers) > 0 and np.linalg.norm(target_pos_obs - target_pos) < p.TARGET_JUMP_THRESH:
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
    frame_canny = apply_canny(cv2.GaussianBlur(frame, (0, 0), 5), canny_thresh_low, canny_thresh_high, 3)
    kernel = np.ones((5, 5), np.uint8)
    frame_canny = cv2.dilate(frame_canny, kernel, iterations=1)
    contours, _ = cv2.findContours(frame_canny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours


def get_feature_2delta():
    feature_2delta = np.array([0, 0])
    if keyboard.is_pressed('up'):
        print('up')
        feature_2delta[1] -= p.ARROW_MOVE_RATE
    if keyboard.is_pressed('down'):
        print('down')
        feature_2delta[1] += p.ARROW_MOVE_RATE
    if keyboard.is_pressed('left'):
        print('left')
        feature_2delta[0] -= p.ARROW_MOVE_RATE
    if keyboard.is_pressed('right'):
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


def draw_settings(ctrl_frame, settings, low_threshold, high_threshold, mode):
    if mode == 'COARSE':
        coarse = [True]
        fine = [False]
        paused = [False]
    elif mode == 'FINE':
        coarse = [False]
        fine = [True]
        paused = [False]
    else:
        coarse = [False]
        fine = [False]
        paused = [True]

    settings.begin(ctrl_frame)
    if not settings.isMinimized():
        cvui.trackbar(settings.width() - 20, low_threshold, 5, 150)
        cvui.trackbar(settings.width() - 20, high_threshold, 80, 300)
        cvui.space(20)  # add 20px of empty space
        cvui.checkbox('Coarse Mode', coarse)
        cvui.checkbox('Fine Mode', fine)
        cvui.checkbox('Paused', paused)
    settings.end()

    if mode == 'PAUSED':
        if coarse[0]:
            mode = 'COARSE'
    elif mode == 'COARSE':
        if paused[0]:
            mode = 'PAUSED'
    elif mode == 'FINE':
        if paused[0]:
            mode = 'PAUSED'
        elif coarse[0]:
            mode = 'COARSE'
    return mode


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


def main():

    global keep_running
    save_video = False
    if save_video:
        sz = (p.IMG_WIDTH_SPOTTER, p.IMG_HEIGHT_SPOTTER)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        vout = cv2.VideoWriter()
        vout.open('track_output.mp4', fourcc, p.FPS_SPOTTER, sz, False)

    signal.signal(signal.SIGINT, sigint_handler)

    ctrl_frame = np.zeros((400, 300, 3), np.uint8)
    settings = EnhancedWindow(10, 50, 270, 270, 'Settings')
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
    low_threshold = [50]
    high_threshold = [150]
    target_pos = np.array([1, 1])
    target_pos_slow = target_pos.copy()
    feature_delta = np.array([0, 0])
    target_track_init = False
    MODE = 'PAUSED'

    while keep_running:

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

        try:
            frame = recv_img(video_socket)
        except zmq.Again:
            print('Timed Out!')
            time.sleep(1)
            continue

        contours = process_contours(frame, low_threshold[0], high_threshold[0])

        cvui.context(p.VIDEO_WINDOW_NAME)

        if cvui.mouse(cvui.IS_DOWN):
            (target_pos, feature_delta) = reset_target_selection()
            target_pos_slow = target_pos.copy()
            target_track_init = True

        feature_delta += get_feature_2delta()
        print(feature_delta)

        (br_centers, target_pos_obs, roi_msg) = determine_roi(frame, contours, target_pos)
        roi_socket.send_pyobj(roi_msg)  # tell the LL camera which ROI to focus

        (dx, dy, target_track_ok) = calculate_movement_offsets(frame, br_centers, target_pos, target_pos_obs, target_pos_slow, feature_delta)

        # draw dots on frame centers
        cv2.circle(frame, (int(p.IMG_DISP_WIDTH_SPOTTER / 2), int(p.IMG_DISP_HEIGHT_SPOTTER / 2)), 5, (0, 0, 255), -1)  # center of frame
        cv2.circle(frame, (p.MACRO_LL_CENTER[0], p.MACRO_LL_CENTER[1]), 5, (255, 0, 255), -1)  # center of macro frame frame

        if MODE == 'PAUSED':
            track_socket.send_string('0 0 0')
        else:
            if target_track_ok and target_track_init:  # this means the track has been initialized
                if MODE == 'COARSE':
                    if stage_z is None:
                        print('Cannot continue until stage node is up')
                        dz = 0
                    else:
                        dist_to_tank = (300 - stage_z) + p.STAGE_TANK_OFFSET
                        ll_max = 2953.5*dist_to_tank**-.729
                        ll_min = 2953.5*(dist_to_tank + p.TANK_DEPTH_MM)**-0.729
                        print('llmin, llmax: (%f, %f)' % (ll_min, ll_max))
                        af_pub.send_pyobj(m.AutofocusMessage(ll_min, ll_max, 1))
                        state = ''
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
                            dz = object_distance_ll - p.FOCUS_DISTANCE_ZOOM
                            print('Object distance_ll: %f' % object_distance_ll)
                        else:
                            dz = 0
                        MODE = 'FINE'
                        stage_z_dir = 1

                elif MODE == 'FINE':
                    if z_moving:
                        dz = 0
                        print('z_moving mode, waiting for stage position set')
                    else:
                        try:
                            macro_sharpness_dir = float(macro_sharpness_sub.recv_string())
                            if macro_sharpness_dir < 0:
                                stage_z_dir = -1 * stage_z_dir
                            dz = 10 * stage_z_dir
                        except zmq.Again:
                            # no sharpness value, which is unexpected
                            dz = 0
                else:
                    print('Unknown MODE %s' % MODE)
                    dz = 0

                track_socket.send_string('%f %f %f' % (dx, dy, dz))  # 'wasteful', but easier debugging for now
            else:
                print('No target track')

        frame = cv2.resize(frame, (p.IMG_DISP_WIDTH_SPOTTER, p.IMG_DISP_HEIGHT_SPOTTER))

        cvui.update(p.VIDEO_WINDOW_NAME)
        cv2.imshow(p.VIDEO_WINDOW_NAME, frame)
        if save_video:
            vout.write(frame)

        cvui.context(p.CTRL_WINDOW_NAME)
        MODE = draw_settings(ctrl_frame, settings, low_threshold, high_threshold, MODE)
        cvui.update(p.CTRL_WINDOW_NAME)
        cv2.imshow(p.CTRL_WINDOW_NAME, ctrl_frame)
        cv2.waitKey(1)

    if save_video:
        vout.release()


if __name__ == '__main__':
    main()
