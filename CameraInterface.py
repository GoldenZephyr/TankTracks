#!/usr/bin/python3

import time
import signal
import os

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

def get_next_vid_ix(directory='videos'):
    files = os.listdir(directory)
    if len(files) > 0:
        vid_files = [f for f in files if f.split('.')[-1] == 'mp4']
        fn_to_ix = lambda s: int(s.split('.')[0])
        indices = map(fn_to_ix, vid_files)
        return max(indices) + 1
    else:
        return 0

def main():
    signal.signal(signal.SIGINT, sigint_handler)
    do_save = True
    fps = p.FPS
    sz = (p.IMG_WIDTH, p.IMG_HEIGHT)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    vout = cv2.VideoWriter()
    vid_fn = get_next_vid_ix()
    vout.open('videos/%04d.mp4' % vid_fn, fourcc, fps, sz)

    context = zmq.Context()
    socket = context.socket(zmq.PUB)
    socket.bind('tcp://*:%s' % p.VIDEO_PORT)

    cam = PGCamera(0)
    cam.start_capture()

    t_loop = time.time() 
    while keep_running:
        t1 = time.time()
        frame = cam.get_frame()
        #print('\nFrame grab took %f seconds' % (time.time() - t1))

        t2 = time.time()
        send_img(socket, frame)
        #print('send_img took %f seconds' % (time.time() - t2))

        if do_save:
            t3 = time.time()
            vout.write(frame)
            #print('Frame write took %f seconds' % (time.time() - t3))
        # this is the line where we want to send the image (i.e. before sleep) ->
        t_wait = 1/fps - (time.time() - t_loop)
        if t_wait > 0:
            time.sleep(t_wait)
        else:
            print('Running too slow! %f' % t_wait)
        t_loop = time.time()
    vout.release()

if __name__ == '__main__':
    main()
