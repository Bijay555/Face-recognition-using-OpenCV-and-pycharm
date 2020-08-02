import cv2
import numpy as np
from sklearn.neighbors import KNeighborsClassifier

data=np.load("face_data.npy")

print(data.shape)

X = data[:,1:].astype(int)
y = data[:,0]

model = KNeighborsClassifier()
model.fit(X,y)


import os

cap = cv2.VideoCapture(0)

detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

while True:

    ret,frame = cap.read()

    if ret:
        faces = detector.detectMultiScale(frame)
        for face in faces:
            x,y,w,h = face
            cut = frame[y:y+h,x:x+w]
            fix = cv2.resize(cut, (200,200))
            gray = cv2.cvtColor(fix, cv2.COLOR_BGR2GRAY)

            out= model.predict([gray.flatten()])

            cv2.rectangle(frame,(x,y),(x+w,y+h), [0,255,0], 2 )
            cv2.putText(frame,str(out[0]), (x,y-10), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, [0,255,0], 2 )
            print(out)

            cv2.imshow("My face", gray)
        cv2.imshow("My screen",frame)


    key = cv2.waitKey(1)

    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()