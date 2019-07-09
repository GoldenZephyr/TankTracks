import cv2
import numpy as np
import time
from video import PGCamera
import Parameters as p
from lts300 import LTS300

cam = PGCamera(0)
cam.start_capture()

z_stage = LTS300(p.LTS300_Z)
z_stage.initialize()

sharpness_list = []

for ix in range(p.N_AUTOFOCUS_JOG):
    frame = cam.get_frame()
    cv2.imshow(p.VIDEO_WINDOW_NAME, frame)
    cv2.waitKey(1)
    frame = cv2.GaussianBlur(frame, (3, 3), 0)
    sharpness = np.mean(cv2.convertScaleAbs(cv2.Laplacian(frame, 3)))
    print(sharpness)
    sharpness_list.append(sharpness)
    z_stage.jog(p.AUTOFOCUS_JOG, min=20, max=250, max_wait=5000)

best_sharpness_ix = np.argmax(sharpness_list)
jog_back_amount = -(p.N_AUTOFOCUS_JOG - (best_sharpness_ix + 1)) * p.AUTOFOCUS_JOG
print('Best sharpness: %f' % sharpness_list[best_sharpness_ix])
print('Sharpness ix: %f' % best_sharpness_ix)
print('jogging back: %f' % jog_back_amount)
time.sleep(3)
z_stage.jog(jog_back_amount, max_wait=4000)
time.sleep(3)
frame = cam.get_frame()
cv2.imshow(p.VIDEO_WINDOW_NAME, frame)
cv2.waitKey(0)
