import signal
import numpy as np
import zmq
import cv2
import Parameters as p

keep_running = True


def sigint_handler(signo, stack_frame):
    global keep_running
    keep_running = False


def recv_img(socket, flags=0, copy=True, track=False):
    """recv a numpy array"""
    msg = socket.recv(flags=flags, copy=copy, track=track)
    buf = memoryview(msg)
    A = np.frombuffer(buf, np.uint8)
    return A.reshape((p.IMG_HEIGHT_ZOOM, p.IMG_WIDTH_ZOOM))


def main():
    signal.signal(signal.SIGINT, sigint_handler)
    context = zmq.Context()
    # Receive video frames from camera
    video_socket = context.socket(zmq.SUB)
    video_socket.setsockopt(zmq.CONFLATE, 1)
    video_socket.setsockopt(zmq.RCVTIMEO, 1000)
    video_socket.connect('tcp://%s:%d' % (p.VIDEO_ZOOM_IP, p.VIDEO_ZOOM_PORT))
    topicfilter = ''
    video_socket.setsockopt_string(zmq.SUBSCRIBE, topicfilter)

    while keep_running:
        try:
            frame = recv_img(video_socket)
        except zmq.Again:
            print('Recv time out!')
            continue
        scale_percent = 25  # percent of original size
        width = int(frame.shape[1] * scale_percent / 100)
        height = int(frame.shape[0] * scale_percent / 100)
        dim = (width, height)
        # resize image
        resized = cv2.resize(frame, dim)
        cv2.imshow('Macro Lens', resized)
        cv2.waitKey(1)


if __name__ == '__main__':
    main()
