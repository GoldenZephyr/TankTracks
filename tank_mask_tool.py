import cv2
import numpy as np
import zmq
import time

import cvui
from EnhancedWindow import EnhancedWindow
import Parameters as p


def recv_img(socket, flags=0, copy=True, track=False):
    """recv a numpy array"""
    msg = socket.recv(flags=flags, copy=copy, track=track)
    buf = memoryview(msg)
    A = np.frombuffer(buf, np.uint8)
    return A.reshape((p.IMG_HEIGHT_SPOTTER, p.IMG_WIDTH_SPOTTER, 3))


keep_running = True


def main():

    context = zmq.Context()
    # Receive video frames from camera
    video_socket = context.socket(zmq.SUB)
    video_socket.setsockopt(zmq.CONFLATE, 1)
    video_socket.setsockopt(zmq.RCVTIMEO, 1000)
    video_socket.connect('tcp://%s:%d' % (p.VIDEO_SPOTTER_IP, p.VIDEO_SPOTTER_PORT))
    topicfilter = ''
    video_socket.setsockopt_string(zmq.SUBSCRIBE, topicfilter)

    # Receive updates from stage
    stage_sub = context.socket(zmq.SUB)
    stage_sub.setsockopt(zmq.CONFLATE, 1)
    stage_sub.setsockopt(zmq.RCVTIMEO, 0)
    stage_sub.connect('tcp://%s:%d' % (p.STAGE_POSITION_IP, p.STAGE_POSITION_PORT))
    topicfilter = ''
    stage_sub.setsockopt_string(zmq.SUBSCRIBE, topicfilter)


    # TODO: We need to connect here instead of binding if we want to use this at the same time as CameraDisplaySpotter
    # Publish tracking deltas
    track_socket = context.socket(zmq.PUB)
    track_socket.bind('tcp://*:%s' % p.TRACK_PORT)

    intrinsic = np.load('intrinsic_calibration/ll_65/intrinsic.npy')
    cvui.init('MaskWindow1')

    # Need to wait for stage to be up and running:
    while 1:
        try:
            stage_msg = stage_sub.recv_string()
            (stage_x, stage_y, stage_z) = [float(x) for x in stage_msg.split(' ')]
            break
        except zmq.Again:
            print('Waiting for stage controller node')
            time.sleep(1)

    stage_zero_offset = np.array([stage_x, stage_y, -stage_z], ndmin=2).T
    np.save('tank_corners_offset.npy', stage_zero_offset)

    while keep_running:

        MODE = '1'

        # phase 1: First position

        corner_points = np.zeros((3, 8))  # first 4 for front, second 4 for back
        corner_ix = 0
        print('Click on corners at front of tank')
        while MODE == '1':

            try:
                frame = recv_img(video_socket)
            except zmq.Again:
                print('No new frame!')
                continue

            for pt_ix in range(corner_ix):
                center = (int(corner_points[0, pt_ix]), int(corner_points[1, pt_ix]))
                if pt_ix < 4:
                    color = (0,0,255)
                else:
                    color = (255,0,0)
                cv2.circle(frame, center, 5, color, -1)

            cvui.update('MaskWindow1')
            cv2.imshow('MaskWindow1', frame)
            cv2.waitKey(1)
            cvui.context('MaskWindow1')
            if cvui.mouse(cvui.CLICK):
                print('CLICK')
                corner_points[0,corner_ix] = cvui.mouse().x
                corner_points[1,corner_ix] = cvui.mouse().y
                corner_points[2,corner_ix] = 1

                corner_ix += 1
                if corner_ix == 4:
                    print('Click on corners at back of tank')
            if corner_ix == 8:
                MODE = '2'

        # Move stage
        # TODO: Add a REP/REQ instead of PUB/SUB for one-off reliably stage movements
        stage_delta = 15.0
        track_socket.send_string('%f 0 0' % (stage_delta * p.X_MOVE_SCALE))
        print('Waiting for stage motion')
        time.sleep(2)

        E = np.array([[0, 0, 0], [0, 0, -stage_delta], [0, stage_delta, 0]])  # essential matrix - only X motion
        F = np.linalg.inv(intrinsic).T @ E @ np.linalg.inv(intrinsic)  # Fundamental Matrix

        # phase 2: Click corners in second position

        corner_points2 = np.zeros((3, 8))
        corner_ix2 = 0
        print('Click on corners at front of tank')
        while MODE == '2':
            try:
                frame = recv_img(video_socket)
            except zmq.Again:
                print('No new frame!')

            # Plot the epipolar line for the current point
            # i.e. p_r^T * F * p_l^T = 0 where F is fundamental matrix and p_r and p_l are points in (homogeneous) image space
            # y0 and y1 are the y coordinates of this line for x=0 and x=img_width
            im1_pt = corner_points[:, corner_ix2:corner_ix2+1]
            d = im1_pt.T @ F
            y0 = int(-d[0, 2] / d[0, 1])
            y1 = int((-d[0, 0]*p.IMG_WIDTH_SPOTTER - d[0, 2]) / d[0, 1])
            cv2.line(frame, (0, y0), (640, y1), (0, 255, 0))

            for pt_ix in range(corner_ix2):
                center = (int(corner_points2[0, pt_ix]), int(corner_points2[1, pt_ix]))
                if pt_ix < 4:
                    color = (0, 0, 255)
                else:
                    color = (255, 0, 0)

                cv2.circle(frame, center, 5, color, -1)

            cvui.update('MaskWindow1')
            cv2.imshow('MaskWindow1', frame)
            cv2.waitKey(1)
            cvui.context('MaskWindow1')
            if cvui.mouse(cvui.CLICK):
                corner_points2[0,corner_ix2] = cvui.mouse().x
                corner_points2[1,corner_ix2] = cvui.mouse().y
                corner_points2[2,corner_ix2] = 1

                corner_ix2 += 1
                if corner_ix2 == 4:
                    print('Click on corners at back of tank')
            if corner_ix2 == 8:
                MODE = '3'

        # Intermezzo: Compute the 3d location of the tank corners
        world_rays1 = np.linalg.inv(intrinsic) @ corner_points
        world_rays2 = np.linalg.inv(intrinsic) @ corner_points2

        world_points = np.zeros((3, 8))
        A = np.zeros((2, 2))
        B = np.zeros((2, 1))
        for point_ix in range(8):
            p1 = np.zeros((3, 1))
            p2 = np.zeros((3, 1))
            v1 = world_rays1[:, point_ix:point_ix+1]
            v2 = world_rays2[:, point_ix:point_ix+1]
            p2[0] = -stage_delta

            A[0, 0] = -np.inner(v1.T, v1.T)
            A[0, 1] = np.inner(v1.T, v2.T)

            B[0, 0] = -(np.inner(p2.T, v1.T) - np.inner(p1.T, v1.T))

            A[1,0] = -np.inner(v1.T, v2.T)
            A[1,1] = np.inner(v2.T, v2.T)

            B[1,0] = -(np.inner(p2.T, v2.T) - np.inner(p1.T, v2.T))

            sol = np.linalg.solve(A, B)

            point1 = p1 + sol[0] * v1
            point2 = p2 + sol[1] * v2
            mp = (point2 - point1) / 2.0 + point1
            world_points[:, point_ix] = mp.squeeze()

        print(world_points)
        np.save('tank_corners.npy', world_points)

        # phase 3: Compute world-frame corners and update as stage moves
        stage_pos_cur = np.zeros((3, 1))
        while MODE == '3':

            try:
                frame = recv_img(video_socket)
            except zmq.Again:
                print('No new frame!')

            try:
                stage_msg = stage_sub.recv_string()
                (stage_x, stage_y, stage_z) = [float(x) for x in stage_msg.split(' ')]
                stage_pos_cur[0] = stage_x
                stage_pos_cur[1] = stage_y
                stage_pos_cur[2] = -stage_z
            except zmq.Again:
                pass

            stage_delta = stage_pos_cur - stage_zero_offset
            corners_translated = world_points - stage_delta

            full_corners_img = np.zeros((2,8))
            for pt_ix in range(8):
                corner_img = intrinsic @ corners_translated[:, pt_ix]
                corner_img = corner_img / corner_img[2]
                full_corners_img[0,pt_ix] = corner_img[0]
                full_corners_img[1,pt_ix] = corner_img[1]
                center = (int(corner_img[0]), int(corner_img[1]))
                if pt_ix < 4:
                    color = (0,0,255)
                else:
                    color = (255,0,0)

                cv2.circle(frame, center, 5, color, -1)
            cv2.imshow('MaskWindow1', frame)

            print([np.array([full_corners_img[:,0], full_corners_img[:,1], full_corners_img[:,2], full_corners_img[:,3]])])

            poly_frame_front = np.zeros(frame.shape, dtype=np.uint8)
            poly_frame_back = np.zeros(frame.shape, dtype=np.uint8)
            poly_frame_front = cv2.fillPoly(poly_frame_front, [np.array([full_corners_img[:,0], full_corners_img[:,1], full_corners_img[:,2], full_corners_img[:,3]], dtype='int32')], (255, 255, 255))
            poly_frame_back = cv2.fillPoly(poly_frame_back, [np.array([full_corners_img[:,4], full_corners_img[:,5], full_corners_img[:,6], full_corners_img[:,7]], dtype='int32')], (255, 255, 255))

            clipped_frame = cv2.bitwise_and(frame, poly_frame_front)
            clipped_frame = cv2.bitwise_and(clipped_frame, poly_frame_back)
            cv2.imshow('MaskWindow2', clipped_frame)
            cv2.waitKey(1)


if __name__ == '__main__':
    main()
