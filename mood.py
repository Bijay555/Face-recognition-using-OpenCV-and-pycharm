import cv2
import dlib
import numpy as np
import os

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

cap = cv2.VideoCapture(0)

mood = input("Enter name of expression: ")

frames=[]
output=[]

while True:

    ret,frame = cap.read()
    faces = detector(frame)

    for face in faces:
        landmarks = predictor(frame, face)
        # print(landmarks.parts())
        # nose = landmarks.parts()[27]
        # print(nose.x, nose.y)
        expression = np.array([[point.x - face.left(), point.y - face.top()] for point in landmarks.parts()[:17]])

    # print(faces)

    if ret:
        cv2.imshow("My screen",frame)

    key = cv2.waitKey(1)
    if key == ord('q'):
        break
    if key == ord('c'):
        # cv2.imwrite(name+".jpg",frame)
        frames.append(expression.flatten())
        output.append([mood])

X = np.array(frames)
y = np.array(output)

data = np.hstack([y, X])

f_name="mood_data.npy"

if os.path.exists(f_name):
    old = np.load(f_name)
    data = np.vstack([old,data])

np.save(f_name, data)


cap.release()
cv2.destroyAllWindows()