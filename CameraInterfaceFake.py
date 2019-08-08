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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('videoname', type=str, help='Path to video file for replay')
    parser.add_argument('--fps', '-r', type=float, help='Desired video fps. Defaults to setting in Parameters.py.')
    parser.add_argument('--video_port', '-p', type=float, help='Port to publish images. Defaults to setting in Parameters.py.')
    parser.add_argument('--rotation', type=int, help='Number of 90 degree rotations to perform. e.g. -1, 0, 1, or 2.')

    args = parser.parse_args()
    fps = args.fps
    video_port = args.video_port
    rotation = args.rotation

    # Load camera parameters from the Parameters.py file if they were not specified on the commandline
    # Note that we can't just put defaults in argparse because we won't know whether to load 'spotter' or 'zoom'
    # settings until *after* parsing the args
    if fps is None:
        fps = p.FPS_SPOTTER
    if video_port is None:
        video_port = p.VIDEO_SPOTTER_PORT
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

    socket_img = context.socket(zmq.PUB)
    socket_img.bind('tcp://*:%s' % video_port)

    cam = cv2.VideoCapture(args.videoname)

    while keep_running:

        t_loop = time.time()

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

        ret, frame = cam.read()
        send_img(socket_img, np.ascontiguousarray(np.rot90(frame, k=rotation)))

        t_wait = 1/fps - (time.time() - t_loop)
        if t_wait > 0:
            time.sleep(t_wait)
        else:
            print('Running too slow! %f' % t_wait)


if __name__ == '__main__':
    main()
