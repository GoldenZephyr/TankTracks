#!/usr/bin/python3
import argparse
import time
import signal
import numpy as np
import cv2
import sys

import zmq
from video import PGCamera
import Parameters as p
import Messages as m

keep_running = True


def send_img(socket, A, flags=0, copy=True, track=False):
    """send a numpy array with metadata"""
    return socket.send(A, flags, copy=copy, track=track)


def sigint_handler(signo, stack_frame):
    # Called by the sigterm (i.e. ctrl-c) signal
    global keep_running
    print('Got ctrl-c!')
    keep_running = False


def autofocus_LL(cam, rotation, socket, af_min, af_max, af_step, roi_ul, roi_lr):
    af_min = max(af_min, 0)
    af_max = min(af_max, 255)
    sharpness_list = []
    for focus_ix in range(af_min, af_max, af_step):
        focus = focus_ix
        cam.set(cv2.CAP_PROP_FOCUS, focus)
        ret, frame = cam.read()
        #if roi_ul is not None:
        #    frame_send = frame.copy()
        #    frame_send[roi_ul[1], roi_ul[0],:] = [0, 0, 255]
        #    frame_send[roi_lr[1], roi_lr[0],:] = [0, 0, 255]
        #else:
        #    frame_send = frame
        send_img(socket, np.ascontiguousarray(np.rot90(frame, k=rotation)))
        if not (roi_ul is None or roi_lr is None):
            frame = frame[roi_ul[1]:roi_lr[1], roi_ul[0]:roi_lr[0]]
        frame_blur = cv2.GaussianBlur(frame, (5, 5), 0)
        sharpness = np.mean(cv2.convertScaleAbs(cv2.Laplacian(frame_blur, 3)))
        sharpness_list.append(sharpness)

    print('Best sharpness: %f' % max(sharpness_list))
    new_focus = np.argmax(sharpness_list) * af_step + af_min
    cam.set(cv2.CAP_PROP_FOCUS, new_focus)
    return (new_focus, max(sharpness_list))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('video_source', type=str, help='Use "spotter" for wide lens, or "zoom" for focused/micro lens.')
    parser.add_argument('camera_type', type=str, help='Currently-supported options are "pointgray" and "econsystems". ')
    parser.add_argument('--fps', '-r', type=float, help='Desired video fps. Defaults to setting in Parameters.py.')
    parser.add_argument('--video_port', '-p', type=float, help='Port to publish images. Defaults to setting in Parameters.py.')
    parser.add_argument('--camera_id', '-c', type=int, help='Device ID of camera. Defaults to setting in Parameters.py.')
    parser.add_argument('--rotation', type=int, help='Number of 90 degree rotations to perform. e.g. -1, 0, 1, or 2.')

    args = parser.parse_args()
    video_source = args.video_source.lower()
    camera_type = args.camera_type.lower()
    fps = args.fps
    video_port = args.video_port
    camera_id = args.camera_id
    rotation = args.rotation

    if camera_type not in ['pointgray', 'econsystems']:
        print('Invalid camera type! Please use "pointgray" or "econsystems". ')
        return 1

    # Load camera parameters from the Parameters.py file if they were not specified on the commandline
    # Note that we can't just put defaults in argparse because we won't know whether to load 'spotter' or 'zoom'
    # settings until *after* parsing the args
    if video_source == 'spotter':
        if fps is None:
            fps = p.FPS_SPOTTER
        if video_port is None:
            video_port = p.VIDEO_PORT_SPOTTER
        if camera_id is None:
            camera_id = p.CAMERA_ID_SPOTTER  # note this may depend on camera plugged in ordering?
        if rotation is None:
            rotation = p.ROTATE_SPOTTER
    elif video_source == 'zoom':
        if fps is None:
            fps = p.FPS_ZOOM
        if video_port is None:
            video_port = p.VIDEO_PORT_ZOOM
        if camera_id is None:
            camera_id = p.CAMERA_ID_ZOOM
        if rotation is None:
            rotation = p.ROTATE_ZOOM
    else:
        print('ERROR: video_source must be "spotter" or "zoom"!')
        return 1

    signal.signal(signal.SIGINT, sigint_handler)

    context = zmq.Context()
    if camera_type == 'econsystems':
        # this socket receives refocusing requests
        refocus_socket_sub = context.socket(zmq.SUB)
        refocus_socket_sub.setsockopt(zmq.RCVTIMEO, 0)
        refocus_socket_sub.connect('tcp://%s:%d' % (p.AUTOFOCUS_IP, p.AUTOFOCUS_PORT))
        topicfilter = ''
        refocus_socket_sub.setsockopt_string(zmq.SUBSCRIBE, topicfilter)

        focus_roi_sub = context.socket(zmq.SUB)
        focus_roi_sub.setsockopt(zmq.CONFLATE, 1)
        focus_roi_sub.setsockopt(zmq.RCVTIMEO, 0)
        focus_roi_sub.connect('tcp://%s:%d' % (p.FOCUS_ROI_IP, p.FOCUS_ROI_PORT))
        topicfilter = ''
        focus_roi_sub.setsockopt_string(zmq.SUBSCRIBE, topicfilter)

       # this socket publishes the current focus setting of the liquid lens camera
        socket_cfocus_pub = context.socket(zmq.PUB)
        socket_cfocus_pub.bind('tcp://*:%s' % p.CURRENT_FOCUS_PORT)

    socket_img = context.socket(zmq.PUB)
    socket_img.bind('tcp://*:%s' % video_port)

    if camera_type == "pointgray":
        cam = PGCamera(camera_id)
        cam.start_capture()
    elif camera_type == "econsystems":
        cam = cv2.VideoCapture(camera_id)
        time.sleep(.5)
        cam.set(cv2.CAP_PROP_FPS, fps)
        time.sleep(.5)
        print('FPS: %f' % fps)
        fps_check = cam.get(cv2.CAP_PROP_FPS)
        print('FPS check: %f' % fps_check)
        time.sleep(.5)
        new_focus = cam.get(cv2.CAP_PROP_FOCUS)
        expected_sharpness = 1e6
        focus_roi_lr = None
        focus_roi_ul = None

        # Doesn't look like we can set resolution??
        #cam.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        #time.sleep(1)
        #cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 1280)
    else:
        print('Please select valid camera type!')
        return 1

    while keep_running:

        if camera_type == 'econsystems' and p.AUTOFOCUS_ENABLE:
            # do autofocus stuff
            print('Sending focus: %d' % new_focus)
            socket_cfocus_pub.send_string('%d' % new_focus)

            try:
                roi_msg = focus_roi_sub.recv_pyobj()
                if type(roi_msg) == m.SetFocusROI:
                    focus_roi_ul = roi_msg.ul
                    focus_roi_lr = roi_msg.lr
                else:
                    print('Malformed Focus ROI message of type %s' % (type(roi_msg)))
            except zmq.Again:
                # don't update ROI
                pass

            try:
                af_msg = refocus_socket_sub.recv_pyobj()
                if type(af_msg) == m.AutofocusMessage:
                    print('Got Autofocus Request')
                    (new_focus, expected_sharpness) = autofocus_LL(cam, rotation, socket_img, af_msg.focus_min, af_msg.focus_max, af_msg.focus_step, focus_roi_ul, focus_roi_lr)
                elif type(af_msg) == m.SetFocusMessage:
                    print('Got SetFocus Request')
                    new_focus = af_msg.focus
                    cam.set(cv2.CAP_PROP_FOCUS, new_focus)
                else:
                    print('Received malformed autofocus message of type %s' % type(af_msg))
            except zmq.Again:
                # no need to autofocus
                (new_focus_obs, best_sharpness) = autofocus_LL(cam, rotation, socket_img, int(new_focus - 15), int(new_focus + 15), int(1), focus_roi_ul, focus_roi_lr)
                new_focus = p.AUTOFOCUS_IIR_DECAY * new_focus + (1-p.AUTOFOCUS_IIR_DECAY) * new_focus_obs
                if best_sharpness < p.EXPECTED_SHARPNESS_THRESHOLD * expected_sharpness:
                    (new_focus, expected_sharpness) = autofocus_LL(cam, rotation, socket_img, 0, 255, 5, focus_roi_ul, focus_roi_lr)


if __name__ == '__main__':
    main()

