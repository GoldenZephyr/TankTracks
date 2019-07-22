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


def draw_settings(ctrl_frame, settings, low_threshold, high_threshold):
        settings.begin(ctrl_frame)
        if not settings.isMinimized():
            cvui.trackbar(settings.width() - 20, low_threshold, 5, 150)
            cvui.trackbar(settings.width() - 20, high_threshold, 80, 300)
            cvui.space(20) # add 20px of empty space
        settings.end()


def apply_canny(frame, high, low, kernel):
    #frame_ret = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #frame_ret = cv2.Canny(frame_ret, low, high, kernel)
    frame_ret = cv2.Canny(frame, low, high, kernel)
    #frame_ret = cv2.cvtColor(frame_ret, cv2.COLOR_GRAY2BGR)
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
    save_video = False
    if save_video:
        sz = (p.IMG_WIDTH_SPOTTER, p.IMG_HEIGHT_SPOTTER)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        vout = cv2.VideoWriter()
        vout.open('track_output.mp4', fourcc, p.FPS_SPOTTER, sz, False)

    signal.signal(signal.SIGINT, sigint_handler)
    #frame = np.zeros((p.IMG_HEIGHT, p.IMG_WIDTH,3), np.uint8)
    resize_scale = p.DISPLAY_RESCALE_SPOTTER
    #frame_rescaled = np.zeros((int(p.IMG_HEIGHT * resize_scale), int(p.IMG_WIDTH * resize_scale), 3), np.uint8)
    ctrl_frame= np.zeros((400, 300, 3), np.uint8)

    settings = EnhancedWindow(10, 50, 270, 270, 'Settings')
    control = EnhancedWindow(10, 300, 270, 100, 'Control')
    cvui.init(p.CTRL_WINDOW_NAME)
    cvui.init(p.VIDEO_WINDOW_NAME)

    context = zmq.Context()

    # Receive video frames from camera
    video_socket = context.socket(zmq.SUB)
    video_socket.setsockopt(zmq.CONFLATE, 1)
    video_socket.setsockopt(zmq.RCVTIMEO, 1000)
    video_socket.connect('tcp://%s:%d' % (p.VIDEO_IP_SPOTTER, p.VIDEO_PORT_SPOTTER))
    topicfilter = ''
    video_socket.setsockopt_string(zmq.SUBSCRIBE, topicfilter)

    # receive stage location updates from stage controller
    stage_loc_sub = context.socket(zmq.SUB)
    stage_loc_sub.setsockopt(zmq.CONFLATE, 1)
    stage_loc_sub.setsockopt(zmq.RCVTIMEO, 0)
    stage_loc_sub.connect('tcp://%s:%d' % (p.STAGE_POSITION_IP, p.STAGE_POSITION_PORT))
    topicfilter = ''
    stage_loc_sub.setsockopt_string(zmq.SUBSCRIBE, topicfilter)
    stage_x = None
    stage_y = None
    stage_z = None

    # Receive focus updates from camera
    focus_sub = context.socket(zmq.SUB)
    focus_sub.setsockopt(zmq.CONFLATE, 1)
    focus_sub.setsockopt(zmq.RCVTIMEO, 0)
    focus_sub.connect('tcp://%s:%d' % (p.CURRENT_FOCUS_IP, p.CURRENT_FOCUS_PORT))
    topicfilter = ''
    focus_sub.setsockopt_string(zmq.SUBSCRIBE, topicfilter)
    current_ll_focus = None

    # Publish tracking deltas
    track_socket = context.socket(zmq.PUB)
    track_socket.bind('tcp://*:%s' % p.TRACK_PORT)

    # Publish ROI bounding box for focusing
    roi_socket = context.socket(zmq.PUB)
    roi_socket.bind('tcp://*:%s' % p.FOCUS_ROI_PORT)

    low_threshold = [50]
    high_threshold = [150]
    target_pos = np.array([1, 1])
    target_pos_slow = target_pos.copy()
    feature_delta = np.array([0, 0])
    target_track_init = False

    while keep_running:

        try:
            stage_string = stage_loc_sub.recv_string()
            toks = stage_string.split(' ')
            stage_x = float(toks[0])
            stage_y = float(toks[1])
            stage_z = float(toks[2])
        except zmq.Again as e:
            pass

        try:
            current_ll_focus = int(focus_sub.recv_string())
            print('Received focus %d' % current_ll_focus)
        except zmq.Again:
            pass

        try:
            frame = recv_img(video_socket)
        except zmq.Again as e:
            print('Timed Out!')
            time.sleep(1)
            continue

        #frame_blur = cv2.GaussianBlur(frame, (0,0), 13)
        #frame = cv2.addWeighted(frame, 1.5, frame_blur, -0.5, 0)
        frame_canny = apply_canny(cv2.GaussianBlur(frame, (0, 0), 2), low_threshold[0], high_threshold[0], 3)
        #frame_canny = cv2.cvtColor(frame_canny, cv2.COLOR_BGR2GRAY)
        kernel = np.ones((5, 5), np.uint8)
        frame_canny = cv2.dilate(frame_canny, kernel, iterations=1)
        #frame_canny = cv2.erode(frame_canny, kernel, iterations=1)

        contours, hierarchy = cv2.findContours(frame_canny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        cvui.context(p.VIDEO_WINDOW_NAME)

        if keyboard.is_pressed('up'):
            print('up')
            feature_delta[1] -= p.ARROW_MOVE_RATE
        if keyboard.is_pressed('down'):
            print('down')
            feature_delta[1] += p.ARROW_MOVE_RATE
        if keyboard.is_pressed('left'):
            print('left')
            feature_delta[0] -= p.ARROW_MOVE_RATE
        if keyboard.is_pressed('right'):
            print('right')
            feature_delta[0] += p.ARROW_MOVE_RATE

        if cvui.mouse(cvui.IS_DOWN):
            if keyboard.is_pressed('a'):
                feature_pos = np.array((cvui.mouse().x/resize_scale, cvui.mouse().y/resize_scale))
                feature_delta = feature_pos - target_pos
            else:
                target_pos = np.array((cvui.mouse().x/resize_scale, cvui.mouse().y/resize_scale))
                target_pos_slow = target_pos.copy()
                target_track_init = True

        bounding_rects = []
        br_centers = []
        for indx, c in enumerate(contours):
            if cv2.contourArea(c) > p.BBOX_AREA_THRESH:
                br = cv2.boundingRect(c)
                x,y,w,h = cv2.boundingRect(c)
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

        roi_socket.send_pyobj(roi_msg) # tell the camera which ROI to focus

        if len(br_centers) > 0 and np.linalg.norm(target_pos_obs - target_pos) < p.TARGET_JUMP_THRESH:
            # (we threw out anything that's *really* off, now we are going to low pass filter
            # we'll just do IIR because the jump threshold will prevent crazy outliers
            target_pos = target_pos * p.LP_IIR_DECAY + target_pos_obs * (1 - p.LP_IIR_DECAY)
            target_pos_slow = target_pos_slow * p.LP_IIR_DECAY_2 + target_pos_obs * (1 - p.LP_IIR_DECAY_2)
            cv2.circle(frame, (int(target_pos[0]), int(target_pos[1])), 75, (255,0,0),3)
            cv2.circle(frame, (int(target_pos_slow[0] + feature_delta[0]), int(target_pos_slow[1] + feature_delta[1])), 5, (0, 255, 0), -1)
            target_track_ok = target_track_init  # only send commands to move the stage if we saw the target this frame
        else:
            target_track_ok = False  # don't command stage if no target track (prevents runaway stage behavior)

        cv2.circle(frame, (int(p.IMG_DISP_WIDTH_SPOTTER / 2), int(p.IMG_DISP_HEIGHT_SPOTTER / 2)), 5, (0, 0, 255), -1)  # center of frame
        cv2.circle(frame, (p.MACRO_LL_CENTER[0], p.MACRO_LL_CENTER[1]), 5, (255, 0, 255), -1)  # center of macro frame frame

        if target_track_ok:  # this means the track has been initialized
            if stage_x is not None and stage_y is not None and stage_z is not None and current_ll_focus is not None:
                dx = (target_pos_slow[0] + feature_delta[0]) - (p.IMG_DISP_WIDTH_SPOTTER / 2 + p.MACRO_FOV_OFFSET[0])
                dy = (target_pos_slow[1] + feature_delta[1]) - (p.IMG_DISP_HEIGHT_SPOTTER / 2 + p.MACRO_FOV_OFFSET[1])
                print(current_ll_focus)
                current_ll_focus = max(current_ll_focus, 47)
                object_distance_ll = (current_ll_focus/2953.5)**(1.0/-0.729)  # (0-255) -> mm
                #air_distance = (300 - stage_z) + p.STAGE_TANK_OFFSET
                #water_distance = object_distance_ll - air_distance
                #dz = air_distance - (p.FOCUS_DISTANCE_ZOOM - water_distance)
                dz = object_distance_ll - p.FOCUS_DISTANCE_ZOOM
                print('Object distance_ll: %f' % object_distance_ll)
                #dz = 0  # dz is determined by LL focus -> object distance -> (object distance - macro focus) = delta
                track_socket.send_string('%f %f %f' % (dx, dy, dz))  # 'wasteful', but easier debugging for now
            else:
                print('Cannot control stage until the current position is updated by the controller!')

        #frame_rescaled = cv2.resize(frame, (int(p.IMG_WIDTH * resize_scale), int(p.IMG_HEIGHT * resize_scale)))
        #frame_rescaled = cv2.resize(frame_canny, (int(p.IMG_WIDTH * resize_scale), int(p.IMG_HEIGHT * resize_scale)))
        frame = cv2.resize(frame, (p.IMG_DISP_WIDTH_SPOTTER, p.IMG_DISP_HEIGHT_SPOTTER))

        cvui.update(p.VIDEO_WINDOW_NAME)
        cv2.imshow(p.VIDEO_WINDOW_NAME, frame)
        if save_video:
            vout.write(frame)

        cvui.context(p.CTRL_WINDOW_NAME)
        draw_settings(ctrl_frame, settings, low_threshold, high_threshold)
        cvui.update(p.CTRL_WINDOW_NAME)
        cv2.imshow(p.CTRL_WINDOW_NAME, ctrl_frame)
        cv2.waitKey(1)

    if save_video:
        vout.release()


if __name__ == '__main__':
    main()
