import cv2
import numpy as np
import Parameters as p

import zmq
import matplotlib.pyplot as plt
import time


def get_vid_frame(cap, frame):
    # get a new frame from the video
    ret, new_frame = cap.read()
    if not ret:
        return False, []
    new_frame = cv2.resize(new_frame, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_LINEAR)
    return True, new_frame


def send_img(socket, A, flags=0, copy=True, track=False):
    """send a numpy array with metadata"""
    return socket.send(A, flags, copy=copy, track=track)


def main():
    context = zmq.Context()
    socket = context.socket(zmq.PUB)
    socket.bind('tcp://*:%s' % p.VIDEO_PORT)

    cap = cv2.VideoCapture(r'C:\Users\aray\desktop\tomopteris1.mp4')
    frame = np.zeros((p.IMG_HEIGHT, p.IMG_WIDTH), np.uint8)
    while(True):
        ret, frame = get_vid_frame(cap, frame)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if not ret:
            break
        send_img(socket, frame)
        time.sleep(.1)

if __name__ == '__main__':
    main()
