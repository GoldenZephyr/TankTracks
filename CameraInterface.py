#!/usr/bin/python3
import argparse
import time
import signal
import numpy as np

import zmq
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
    parser.add_argument('video_source', type=str, help='Use "spotter" for wide lens, or "zoom" for focused/micro lens')
    args = parser.parse_args()
    if args.video_source.lower() == 'spotter':
        fps = p.FPS_SPOTTER
        video_port = p.VIDEO_PORT_SPOTTER
        camera_id = p.CAMERA_ID_SPOTTER  # note this may depend on camera plugged in ordering?
    elif args.video_source.lower() == 'zoom':
        fps = p.FPS_ZOOM
        video_port = p.VIDEO_PORT_ZOOM
        camera_id = p.CAMERA_ID_ZOOM
    else:
        print('ERROR: video_source must be "spotter" or "zoom"!')
        return 1

    signal.signal(signal.SIGINT, sigint_handler)

    context = zmq.Context()
    socket = context.socket(zmq.PUB)
    socket.bind('tcp://*:%s' % video_port)

    cam = PGCamera(camera_id, disp_width=1072, disp_height=1920)
    cam.start_capture()

    t_loop = time.time() 
    while keep_running:
        t1 = time.time()
        frame = cam.get_frame()
        if args.video_source and p.ROTATE_SPOTTER:  # we might need to rotate the image
            frame = np.ascontiguousarray(np.rot90(frame))
            print(frame.shape)

        #print('\nFrame grab took %f seconds' % (time.time() - t1))

        t2 = time.time()
        send_img(socket, frame)
        #print('send_img took %f seconds' % (time.time() - t2))

        # this is the line where we want to send the image (i.e. before sleep) ->
        t_wait = 1/fps - (time.time() - t_loop)
        if t_wait > 0:
            time.sleep(t_wait)
        else:
            print('Running too slow! %f' % t_wait)
        t_loop = time.time()


if __name__ == '__main__':
    main()
