import os
import cv2
import numpy as np

f_name= "face_data.npy"
if os.path.exists(f_name):
    old = np.load(f_name)
    data = np.vstack([old,data])

np.save(f_name, data)
