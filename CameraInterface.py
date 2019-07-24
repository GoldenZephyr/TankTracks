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


def autofocus_LL(cam, rotation, socket, af_min, af_max, af_step):
    sharpness_list = []
    for focus_ix in range(af_min, af_max, af_step):
        focus = focus_ix
        cam.set(cv2.CAP_PROP_FOCUS, focus)
        ret, frame = cam.read()
        send_img(socket, np.ascontiguousarray(np.rot90(frame, k=rotation)))
        frame_blur = cv2.GaussianBlur(frame, (3, 3), 0)
        sharpness = np.mean(cv2.convertScaleAbs(cv2.Laplacian(frame_blur, 3)))
        sharpness_list.append(sharpness)

    new_focus = np.argmax(sharpness_list) * af_step + af_min
    cam.set(cv2.CAP_PROP_FOCUS, new_focus)
    return new_focus


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
            video_port = p.VIDEO_SPOTTER_PORT
        if camera_id is None:
            camera_id = p.CAMERA_ID_SPOTTER  # note this may depend on camera plugged in ordering?
        if rotation is None:
            rotation = p.ROTATE_SPOTTER
    elif video_source == 'zoom':
        if fps is None:
            fps = p.FPS_ZOOM
        if video_port is None:
            video_port = p.VIDEO_ZOOM_PORT
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
            socket_cfocus_pub.send_string('%d' % new_focus)

            try:
                af_msg = refocus_socket_sub.recv_pyobj()
                if type(af_msg) == m.AutofocusMessage:
                    print('Got Autofocus Request')
                    new_focus = autofocus_LL(cam, rotation, socket_img, af_msg.focus_min, af_msg.focus_max, af_msg.focus_step)
                elif type(af_msg) == m.SetFocusMessage:
                    print('Got SetFocus Request')
                    new_focus = af_msg.focus
                    cam.set(cv2.CAP_PROP_FOCUS, new_focus)
                else:
                    print('Received malformed autofocus message of type %s' % type(af_msg))
            except zmq.Again:
                # no need to autofocus
                pass


        t_loop = time.time()
        frame_ok, frame = cam.read()
        #if camera_type == 'econsystems':
        #    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Rotate the frame if necessary
        # Note that unless rotation has been explicitly asked for, rotation=0 and therefore no rotation happens
        frame = np.ascontiguousarray(np.rot90(frame, k=rotation))

        send_img(socket_img, frame)

        t_wait = 1/fps - (time.time() - t_loop)
        if t_wait > 0:
            time.sleep(t_wait)
        else:
            print('Running too slow! %f' % t_wait)


if __name__ == '__main__':
    main()
