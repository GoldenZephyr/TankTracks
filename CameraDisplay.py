#!/usr/bin/python3
import time
import signal

import zmq
import cv2
import numpy as np

from EnhancedWindow import EnhancedWindow
import cvui
import Parameters as p

keep_running = True


def draw_settings(ctrl_frame, settings, low_threshold, high_threshold):
        settings.begin(ctrl_frame)
        if not settings.isMinimized():
            cvui.trackbar(settings.width() - 20, low_threshold, 5, 150)
            cvui.trackbar(settings.width() - 20, high_threshold, 80, 300)
            cvui.space(20) # add 20px of empty space
        settings.end()


def apply_canny(frame, high, low, kernel):
    frame_ret = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame_ret = cv2.Canny(frame_ret, low, high, kernel)
    frame_ret = cv2.cvtColor(frame_ret, cv2.COLOR_GRAY2BGR)
    return frame_ret


def recv_img(socket, flags=0, copy=True, track=False):
    """recv a numpy array"""
    msg = socket.recv(flags=flags, copy=copy, track=track)
    buf = memoryview(msg)
    A = np.frombuffer(buf, np.uint8)
    return A.reshape((p.IMG_HEIGHT, p.IMG_WIDTH, 3))


def sigint_handler(signo, stack_frame):
    global keep_running
    keep_running = False


def main():
    signal.signal(signal.SIGINT, sigint_handler)
    frame = np.zeros((p.IMG_HEIGHT, p.IMG_WIDTH,3), np.uint8)
    resize_scale = 0.3
    frame_rescaled = np.zeros((int(p.IMG_HEIGHT * resize_scale), int(p.IMG_WIDTH * resize_scale), 3), np.uint8)
    ctrl_frame= np.zeros((p.IMG_HEIGHT, p.IMG_WIDTH, 3), np.uint8)

    settings = EnhancedWindow(10, 50, 270, 270, 'Settings')
    control = EnhancedWindow(10, 300, 270, 100, 'Control')
    cvui.init(p.CTRL_WINDOW_NAME)
    cvui.init(p.VIDEO_WINDOW_NAME)

    context = zmq.Context()
    video_socket = context.socket(zmq.SUB)
    video_socket.setsockopt(zmq.CONFLATE, 1)
    video_socket.setsockopt(zmq.RCVTIMEO, 1000)
    video_socket.connect('tcp://localhost:%d' % p.VIDEO_PORT)
    topicfilter = ''
    video_socket.setsockopt_string(zmq.SUBSCRIBE, topicfilter)

    track_socket = context.socket(zmq.PUB)
    track_socket.bind('tcp://*:%s' % p.TRACK_PORT)

    low_threshold = [50]
    high_threshold = [150]
    target_pos = np.array([1,1])
    target_track_ok = False

    while keep_running:
        try:
            frame = recv_img(video_socket)
        except zmq.Again as e:
            print('Timed Out!')
            time.sleep(1)
            continue

        frame_blur = cv2.GaussianBlur(frame, (0,0), 13)
        frame = cv2.addWeighted(frame, 1.5, frame_blur, -0.5, 0)
        frame_canny = apply_canny(frame, low_threshold[0], high_threshold[0], 3)
        frame_canny = cv2.cvtColor(frame_canny, cv2.COLOR_BGR2GRAY)
        contours, hierarchy = cv2.findContours(frame_canny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        cvui.context(p.VIDEO_WINDOW_NAME)
        if cvui.mouse(cvui.IS_DOWN):
            target_pos = np.array((cvui.mouse().x/resize_scale, cvui.mouse().y/resize_scale))
            target_track_ok = True

        bounding_rects = []
        br_centers = np.zeros((len(contours),2))
        for indx,c in enumerate(contours):
            if cv2.contourArea(c) > p.BBOX_AREA_THRESH:
                br = cv2.boundingRect(c)
                x,y,w,h = cv2.boundingRect(c)
                cv2.circle(frame, (int(x + w/2), int(y+h/2)), 50, (0,255,0),3)
                #cv2.rectangle(frame, br, (0, 255, 0), 3)
                bounding_rects.append(br)
                br_centers[indx,:] = (x + w/2, y + h/2)

    
        if len(br_centers) > 0: # if we found any bounding boxes, determine their distances from target pt
            br_dists = np.linalg.norm(br_centers - target_pos, axis=1)
            target_pos_obs = br_centers[np.argmin(br_dists),:]

        if len(br_centers) > 0 and np.linalg.norm(target_pos_obs - target_pos) < p.TARGET_JUMP_THRESH:
            # (we threw out anything that's *really* off, now we are going to low pass filter
            # we'll just do IIR because the jump threshold will prevent crazy outliers
            target_pos = target_pos * p.LP_IIR_DECAY + target_pos_obs * (1 - p.LP_IIR_DECAY)
            cv2.circle(frame, (int(target_pos[0]), int(target_pos[1])), 75, (255,0,0),3)

        # send track position
        if target_track_ok: # this means the track has been initialized
            dx = target_pos[0] - p.IMG_WIDTH / 2
            dy = target_pos[1] - p.IMG_HEIGHT / 2
            print('%f, %f' % (dx, dy))
            track_socket.send_string('%f %f' % (dx, dy)) # 'wasteful', but easier debugging for now


        #frame_rescaled = cv2.resize(frame, (int(p.IMG_WIDTH * resize_scale), int(p.IMG_HEIGHT * resize_scale)))
        frame_rescaled = cv2.resize(frame, (int(p.IMG_WIDTH * resize_scale), int(p.IMG_HEIGHT * resize_scale)))
        cvui.update(p.VIDEO_WINDOW_NAME)
        cv2.imshow(p.VIDEO_WINDOW_NAME, frame_rescaled)

        cvui.context(p.CTRL_WINDOW_NAME)
        draw_settings(ctrl_frame, settings, low_threshold, high_threshold)
        cvui.update(p.CTRL_WINDOW_NAME)
        cv2.imshow(p.CTRL_WINDOW_NAME, ctrl_frame)
        cv2.waitKey(20)

if __name__ == '__main__':
    main()
