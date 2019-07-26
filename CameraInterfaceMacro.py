#!/usr/bin/python3
import argparse
import time
import signal
import numpy as np

import zmq
import cv2

from video import PGCamera
import Parameters as p

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
        fps = p.FPS_ZOOM
    if video_port is None:
        video_port = p.VIDEO_ZOOM_PORT
    if camera_id is None:
        camera_id = p.CAMERA_ID_ZOOM
    if rotation is None:
        rotation = p.ROTATE_ZOOM

    signal.signal(signal.SIGINT, sigint_handler)

    context = zmq.Context()

    socket_img = context.socket(zmq.PUB)
    socket_img.bind('tcp://*:%s' % video_port)

    socket_sharpness = context.socket(zmq.PUB)
    socket_sharpness.bind('tcp://*:%s' % p.MACRO_SHARPNESS_PORT)

    cam = PGCamera(camera_id)
    cam.start_capture()

    sharpness_est = 0
    n_hist = 15
    sharpness_delta_history = np.zeros((1, n_hist))
    hist_ix = 0
    while keep_running:

        t_loop = time.time()
        frame_ok, frame = cam.read()

        # Rotate the frame if necessary
        # Note that unless rotation has been explicitly asked for, rotation=0 and therefore no rotation happens
        frame = np.ascontiguousarray(np.rot90(frame, k=rotation))
        send_img(socket_img, frame)
        frame_blur = cv2.GaussianBlur(frame, (3, 3), 0)
        frame_sharpness = np.abs(cv2.Laplacian(frame_blur, ddepth=3, ksize=3))
        #frame_sharpness_flat = frame_sharpness.flatten()
        #sharpness = np.mean(np.sort(frame_sharpness_flat)[-int(frame.size*.2):])
        sharpness = np.mean(frame_sharpness)

        if hist_ix == 0:
            sharpness_last = sharpness
        sharpness_delta_history[0, hist_ix % n_hist] = sharpness - sharpness_last
        avg_sharpness_delta = np.mean(sharpness_delta_history)
        #print(avg_sharpness_delta)
        print(sharpness)
        socket_sharpness.send_string(str(avg_sharpness_delta))
        sharpness_last = sharpness
        hist_ix += 1

        t_wait = 1/fps - (time.time() - t_loop)
        if t_wait > 0:
            time.sleep(t_wait)
        else:
            print('Running too slow! %f' % t_wait)


if __name__ == '__main__':
    main()
