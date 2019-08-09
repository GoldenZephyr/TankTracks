import numpy as np
import cv2
import Parameters as p


class CannyTracker:
    """ Encapsulate all of the parameters/state necessary for the Canny tracker and estimation"""
    def __init__(self):
        self.canny_low = None
        self.canny_high = None
        self.target_pos = None


def determine_roi(frame, contours, target_pos):
    bounding_rects = []
    br_centers = []
    for indx, c in enumerate(contours):
        if cv2.contourArea(c) > p.BBOX_AREA_THRESH:
            br = cv2.boundingRect(c)
            x, y, w, h = cv2.boundingRect(c)
            cv2.rectangle(frame, br, (0, 255, 0), 3)
            bounding_rects.append(br)
            br_centers.append((x + w/2, y + h/2))
    br_centers = np.array(br_centers)

    if len(br_centers) > 0:  # if we found any bounding boxes, determine their distances from target pt
        br_dists = np.linalg.norm(br_centers - target_pos, axis=1)
        best_bbox_ix = np.argmin(br_dists)
        target_pos_obs = br_centers[best_bbox_ix, :]
        x, y, w, h = bounding_rects[best_bbox_ix]
        bbox_ul = (x, y)
        bbox_lr = (x + w, y + h)
        roi = (bbox_ul, bbox_lr)
    else:
        roi = None  # Don't focus on any particular ROI
        target_pos_obs = None

    return br_centers, target_pos_obs, roi


def process_contours(frame, canny_thresh_low, canny_thresh_high):
    frame_canny = apply_canny(cv2.GaussianBlur(frame, (0, 0), 3), canny_thresh_low, canny_thresh_high, 3)
    kernel = np.ones((5, 5), np.uint8)
    frame_canny = cv2.dilate(frame_canny, kernel, iterations=1)
    contours, _ = cv2.findContours(frame_canny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours


def apply_canny(frame, high, low, kernel):
    frame_ret = cv2.Canny(frame, low, high, kernel)
    return frame_ret


def update_canny_tracker(frame, canny_tracker_state):

    contours = process_contours(frame, canny_tracker_state.canny_low[0], canny_tracker_state.canny_high[0])

    (br_centers, target_pos_obs, roi) = determine_roi(frame, contours, canny_tracker_state.target_pos)

    return target_pos_obs, roi, canny_tracker_state
