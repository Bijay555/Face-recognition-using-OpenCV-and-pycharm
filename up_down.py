import cv2
import dlib
import keyboard

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

cap = cv2.VideoCapture(0)

while True:

    ret,frame = cap.read()
    faces = detector(frame)

    for face in faces:
        landmarks = predictor(frame, face)
        # print(landmarks.parts())
        # nose = landmarks.parts()[27]
        # print(nose.x, nose.y)
        mouth_up = landmarks.parts()[62].y
        mouth_down = landmarks.parts()[66].y

        if mouth_down - mouth_up >5:
            keyboard.press('up')
        else:
            keyboard.press('down')
    # print(faces)

    if ret:
        cv2.imshow("My screen",frame)

    key = cv2.waitKey(1)
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()