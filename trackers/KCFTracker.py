import numpy as np
import cvui
import cv2
from cvui_utils import roi_tool


class KCFTracker:
    """ Encapsulate all of the parameters/state necessary for the KCF tracker and estimation"""
    def __init__(self):
        self.kcf_box_anchor = None
        self.kcf_roi = None
        self.kcf_tracker_init = None
        self.kcf_tracker = None


def update_kcf_tracker(frame, kcf_tracker_state):
    """ Update the kcf tracker based on new frame and previous state """

    anchor, kcf_roi, drawing, new_bb = roi_tool(frame, kcf_tracker_state.kcf_box_anchor, kcf_tracker_state.kcf_roi)

    target_pos_obs = None
    roi = None

    if drawing:
        kcf_tracker_state.kcf_tracker_init = False

    if new_bb:
        kcf_tracker_state.kcf_tracker_init = True
        kcf_tracker_state.kcf_tracker = cv2.TrackerKCF_create()
        kcf_tracker_state.kcf_tracker.init(frame, (kcf_roi.x, kcf_roi.y, kcf_roi.width, kcf_roi.height))
    elif kcf_tracker_state.kcf_tracker_init:
        track_ok, new_roi = kcf_tracker_state.kcf_tracker.update(frame)
        x1 = kcf_roi.x
        y1 = kcf_roi.y
        w = kcf_roi.width
        h = kcf_roi.height
        if track_ok:
            target_pos_obs = np.array([x1 + w/2., y1 + h/2])
        else:
            target_pos_obs = None
        kcf_tracker_state.kcf_roi = cvui.Rect(new_roi[0], new_roi[1], new_roi[2], new_roi[3])
        roi = ((x1, y1), (x1 + w, y1 + h))
    return target_pos_obs, roi, kcf_tracker_state
