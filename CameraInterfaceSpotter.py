#!/usr/bin/python3
import argparse
import time
import signal
import numpy as np
import cv2

import zmq
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


def autofocus_LL(cam, rotation, socket, socket_state, af_min, af_max, af_step, roi_ul, roi_lr):
    af_min = int(max(af_min, 0))
    af_max = int(min(af_max, 255))
    sharpness_list = []
    cam.set(cv2.CAP_PROP_FOCUS, af_min)
    time.sleep(.1)
    for focus_ix in range(af_min, af_max, af_step):
        focus = focus_ix
        cam.set(cv2.CAP_PROP_FOCUS, focus)
        ret, frame = cam.read()
        send_img(socket, np.ascontiguousarray(np.rot90(frame, k=rotation)))
        socket_state.send_string('FOCUSING')
        if not (roi_ul is None or roi_lr is None):
            frame = frame[roi_ul[1]:roi_lr[1], roi_ul[0]:roi_lr[0]]
        # calculate sharpness
        frame_blur = cv2.GaussianBlur(frame, (9, 9), 0)
        frame_sharpness = np.abs(cv2.Laplacian(frame_blur, ddepth=3, ksize=5))
        frame_sharpness_flat = frame_sharpness.flatten()
        sharpness = np.mean(np.sort(frame_sharpness_flat)[-int(frame.size*.2):])
        sharpness_list.append(sharpness)

    print('Best sharpness: %f' % max(sharpness_list))
    new_focus = np.argmax(sharpness_list) * af_step + af_min
    cam.set(cv2.CAP_PROP_FOCUS, new_focus)
    return (new_focus, max(sharpness_list))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--fps', '-r', type=float, help='Desired video fps. Defaults to setting in Parameters.py.')
    parser.add_argument('--video_port', '-p', type=float, help='Port to publish images. Defaults to setting in Parameters.py.')
    parser.add_argument('--camera_id', '-c', type=int, help='Device ID of camera. Defaults to setting in Parameters.py.')
    parser.add_argument('--rotation', type=int, help='Number of 90 degree rotations to perform. e.g. -1, 0, 1, or 2.')

    args = parser.parse_args()
    fps = args.fps
    video_port = args.video_port
    camera_id = args.camera_id
    rotation = args.rotation

    # Load camera parameters from the Parameters.py file if they were not specified on the commandline
    # Note that we can't just put defaults in argparse because we won't know whether to load 'spotter' or 'zoom'
    # settings until *after* parsing the args
    if fps is None:
        fps = p.FPS_SPOTTER
    if video_port is None:
        video_port = p.VIDEO_SPOTTER_PORT
    if camera_id is None:
        camera_id = p.CAMERA_ID_SPOTTER  # note this may depend on camera plugged in ordering?
    if rotation is None:
        rotation = p.ROTATE_SPOTTER

    signal.signal(signal.SIGINT, sigint_handler)

    context = zmq.Context()

    # this socket receives refocusing requests
    refocus_socket_sub = context.socket(zmq.SUB)
    refocus_socket_sub.setsockopt(zmq.RCVTIMEO, 0)
    refocus_socket_sub.connect('tcp://%s:%d' % (p.AUTOFOCUS_IP, p.AUTOFOCUS_PORT))
    topicfilter = ''
    refocus_socket_sub.setsockopt_string(zmq.SUBSCRIBE, topicfilter)

    # listen for ROI to focus
    focus_roi_sub = context.socket(zmq.SUB)
    focus_roi_sub.setsockopt(zmq.CONFLATE, 1)
    focus_roi_sub.setsockopt(zmq.RCVTIMEO, 0)
    focus_roi_sub.connect('tcp://%s:%d' % (p.FOCUS_ROI_IP, p.FOCUS_ROI_PORT))
    topicfilter = ''
    focus_roi_sub.setsockopt_string(zmq.SUBSCRIBE, topicfilter)

    # this socket publishes the current focus setting of the liquid lens camera
    socket_cfocus_pub = context.socket(zmq.PUB)
    socket_cfocus_pub.bind('tcp://*:%s' % p.CURRENT_FOCUS_PORT)

    socket_focus_state_pub = context.socket(zmq.PUB)
    socket_focus_state_pub.bind('tcp://*:%s' % p.SPOTTER_FOCUS_STATE_PORT)

    socket_img = context.socket(zmq.PUB)
    socket_img.bind('tcp://*:%s' % video_port)

    cam = cv2.VideoCapture(camera_id)
    time.sleep(1)
    _, frame = cam.read()  # some stuff doesn't work until you read a frame (?)
    #time.sleep(.5)
    #cam.set(cv2.CAP_PROP_FPS, 30)
    #time.sleep(.5)
    #print('FPS: %f' % fps)
    fps_check = cam.get(cv2.CAP_PROP_FPS)
    print('FPS check: %f' % fps_check)
    time.sleep(.5)
    new_focus = cam.get(cv2.CAP_PROP_FOCUS)
    focus_roi_lr = None
    focus_roi_ul = None

    while keep_running:

        t_loop = time.time()
        # do autofocus stuff
        print('Sending focus: %.3f' % new_focus)
        socket_cfocus_pub.send_string('%.3f' % new_focus)
        socket_focus_state_pub.send_string('FIXED')

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
                (new_focus, expected_sharpness) = autofocus_LL(cam, rotation, socket_img, socket_focus_state_pub, af_msg.focus_min, af_msg.focus_max, af_msg.focus_step, focus_roi_ul, focus_roi_lr)
            elif type(af_msg) == m.SetFocusMessage:
                print('Got SetFocus Request')
                new_focus = af_msg.focus
                cam.set(cv2.CAP_PROP_FOCUS, new_focus)
            else:
                print('Received malformed autofocus message of type %s' % type(af_msg))
        except zmq.Again:
            # No autofocus request, just send an image
            ret, frame = cam.read()
            send_img(socket_img, np.ascontiguousarray(np.rot90(frame, k=rotation)))

            t_wait = 1/fps - (time.time() - t_loop)
            if t_wait > 0:
                time.sleep(t_wait)
            else:
                print('Running too slow! %f' % t_wait)


if __name__ == '__main__':
    main()
