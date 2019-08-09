import cv2
import numpy as np
from cvui_utils import roi_tool


class ThresholdTracker:
    """ Encapsulate all of the parameters/state necessary for the KCF tracker and estimation"""
    def __init__(self):

        self.initialized = False
        self.threshold = None
        self.target_pos = None
        self.roi = None  # x1, y1, w, h
        self.box_anchor = None


def update_threshold_tracker(frame_disp, tracker_state):

    frame = frame_disp
    anchor, thresh_roi, drawing, new_bb = roi_tool(frame, tracker_state.box_anchor, tracker_state.roi)

    tracker_state.roi = thresh_roi

    if drawing:
        tracker_state.initialized = False

    if new_bb:
        tracker_state.initialized = True
        target_pos_obs = None
        roi = None
    elif tracker_state.initialized:

        # clip to bounding box
        roi_mask = np.zeros(frame.shape, dtype=np.uint8)
        x = int(tracker_state.roi.x)
        y = int(tracker_state.roi.y)
        w = int(tracker_state.roi.width)
        h = int(tracker_state.roi.height)
        roi = ((x, y), (x + w, y + h))
        roi_mask[y:y+h, x:x+w] = 255
        frame = cv2.bitwise_and(frame, roi_mask)

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        _, frame = cv2.threshold(frame, tracker_state.threshold[0], 255, cv2.THRESH_BINARY)  # + cv2.THRESH_OTSU ?

        kernel_dilate = np.ones((5, 5), np.uint8)
        kernel_erode = np.ones((5, 5), np.uint8)
        frame = cv2.erode(frame, kernel_erode, iterations=2)
        frame = cv2.dilate(frame, kernel_dilate, iterations=2)
        contours, hierarchy = cv2.findContours(frame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        frame = cv2.fillPoly(frame, contours, color=255)

        M = cv2.moments(frame, binaryImage=True)
        if M['m00'] != 0:
            cX = int(M['m10'] / M['m00'])
            cY = int(M['m01'] / M['m00'])
            target_pos_obs = np.array([cX, cY])
            tracker_state.roi.x = cX - int(tracker_state.roi.width/2)
            tracker_state.roi.y = cY - int(tracker_state.roi.height/2)
            #frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            cv2.circle(frame, (cX, cY), 50, (0, 255, 0), 1)
        else:
            target_pos_obs = None
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

        #frame_disp[:] = frame[:]

    else:
        target_pos_obs = None
        roi = None

    return target_pos_obs, roi, tracker_state

