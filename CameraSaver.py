import os
import signal

import argparse
import cv2
import numpy as np
import zmq
import time
import base64

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


def recv_img(socket, array_size, flags=0, copy=True, track=False):
    """recv a numpy array"""
    msg = socket.recv(flags=flags, copy=False, track=track)
    #msg = socket.recv()
    buf = memoryview(msg)
    img_flat = np.frombuffer(buf, np.uint8)
    img_reshaped = img_flat.reshape(array_size)
    return img_reshaped


def main():
    parser = argparse.ArgumentParser('CameraSaver.py')
    parser.add_argument('video_source', type=str, help='Choose from "zoom" or "spotter". ')
    args = parser.parse_args()
    video_source = args.video_source.lower()
    if video_source == 'zoom':
        img_width = p.IMG_WIDTH_ZOOM
        img_height = p.IMG_HEIGHT_ZOOM
        color_depth = p.COLOR_CHAN_ZOOM
        fps = p.FPS_ZOOM
        video_port = p.VIDEO_PORT_ZOOM
        video_ip = p.VIDEO_IP_ZOOM
    elif video_source == 'spotter':
        img_width = p.IMG_WIDTH_SPOTTER
        img_height = p.IMG_HEIGHT_SPOTTER
        color_depth = p.COLOR_CHAN_SPOTTER
        fps = p.FPS_SPOTTER
        video_port = p.VIDEO_PORT_SPOTTER
        video_ip = p.VIDEO_IP_SPOTTER
    else:
        print('video_source must be either "zoom" or "spotter". ')

    signal.signal(signal.SIGINT, sigint_handler)

    context = zmq.Context()
    video_socket = context.socket(zmq.SUB)
    video_socket.setsockopt(zmq.RCVTIMEO, 1000)
    video_socket.connect('tcp://%s:%d' % (video_ip, video_port))
    topicfilter = ''
    video_socket.setsockopt_string(zmq.SUBSCRIBE, topicfilter)

    array_size = (img_height, img_width, color_depth)
    frame_size_wh = (img_width, img_height)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    vout = cv2.VideoWriter()
    vid_fn = get_next_vid_ix('videos/%s' % video_source)
    vout.open('videos/%s/%04d.mp4' % (video_source, vid_fn), fourcc, fps, frame_size_wh, color_depth > 1)


    # TODO: Need to check if *average* rate is fast enough
    prev_loop_start = time.time() + 1
    while keep_going:
        loop_time = time.time() - prev_loop_start
        prev_loop_start = time.time()
        #if loop_time > 1 / fps:
        #    print("Going too slowly! %f" % (1/fps - loop_time))
        #else:
        #    print('ok %f' % (1/fps - loop_time))
        try:
            frame = recv_img(video_socket, array_size)
        except zmq.Again as e:
            print('Timed Out!')
            time.sleep(1)
            continue
        vout.write(frame)
    vout.release()


if __name__ == '__main__':
    main()
