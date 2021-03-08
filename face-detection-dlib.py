#coding=utf-8

import cv2
import dlib

path = "img/face-example.jpg"
img = cv2.imread(path)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#face detector
detector = dlib.get_frontal_face_detector()
#get predictor
predictor = dlib.shape_predictor(
    "D:\\python3\\Lib\\site-packages\\dlib-data\\shape_predictor_68_face_landmarks.dat"
)

dets = detector(gray, 1)
for face in dets:
    shape = predictor(img, face) # finds 68 points on a face
    #iterates all the points, prints their coorinates, and circle out
    for pt in shape.parts():
        pt_pos = (pt.x, pt.y)
        cv2.circle(img, pt_pos, 2, (0, 255, 0), 1)
    cv2.imshow("image", img)

cv2.waitKey(0)
cv2.destroyAllWindows()
