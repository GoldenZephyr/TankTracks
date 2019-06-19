import os
import signal

import cv2
import numpy as np
import zmq
import time

import Parameters as p

keep_going = True

def sigint_handler(_signo, _stack_frame):
    global keep_going
    keep_going = False

def get_next_vid_ix(directory='videos'):
    files = os.listdir(directory)
    if len(files) > 0:
        vid_files = [f for f in files if f.split('.')[-1] == 'mp4']
        fn_to_ix = lambda s: int(s.split('.')[0])
        indices = map(fn_to_ix, vid_files)
        return max(indices) + 1
    else:
        return 0


def recv_img(socket, flags=0, copy=True, track=False):
    """recv a numpy array"""
    msg = socket.recv(flags=flags, copy=copy, track=track)
    buf = memoryview(msg)
    A = np.frombuffer(buf, np.uint8)
    #return A.reshape((p.IMG_HEIGHT, p.IMG_WIDTH, 3))
    return A.reshape((p.IMG_HEIGHT, p.IMG_WIDTH))


def main():
    signal.signal(signal.SIGINT, sigint_handler)
    context = zmq.Context()
    video_socket = context.socket(zmq.SUB)
    video_socket.setsockopt(zmq.CONFLATE, 1)
    video_socket.setsockopt(zmq.RCVTIMEO, 1000)
    video_socket.connect('tcp://%s:%d' % (p.VIDEO_IP, p.VIDEO_PORT))
    topicfilter = ''
    video_socket.setsockopt_string(zmq.SUBSCRIBE, topicfilter)

    sz = (p.IMG_WIDTH, p.IMG_HEIGHT)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    vout = cv2.VideoWriter()
    vid_fn = get_next_vid_ix()
    vout.open('videos/%04d.mp4' % vid_fn, fourcc, p.FPS, sz, False)
    while keep_going:
        try:
            frame = recv_img(video_socket)
        except zmq.Again as e:
            print('Timed Out!')
            time.sleep(1)
            continue
        vout.write(frame)
    vout.release()



if __name__ == '__main__':
    main()