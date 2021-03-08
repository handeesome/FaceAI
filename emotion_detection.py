#coding=utf-8
#表情识别

import cv2
from keras.models import load_model
import numpy as np
import datetime

startTime = datetime.datetime.now()
emotion_classifier = load_model(
    'trained_models/emotion_models/simple_CNN.530-0.65.hdf5')
endTime = datetime.datetime.now()
print(endTime - startTime)


emotion_labels = {
    0: 'angry',
    1: 'disgust',
    2: 'fear',
    3: 'happy',
    4: 'sad',
    5: 'surprise',
    6: 'neutral'
}

img = cv2.imread("img/face_recognition/sb.jpg")
face_classifier = cv2.CascadeClassifier(
    "trained_models/detection_models/haarcascade_frontalface_default.xml"
)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
faces = face_classifier.detectMultiScale(
    gray, scaleFactor=1.2, minNeighbors=3, minSize=(40, 40))
color = (255, 0, 0)

for (x, y, w, h) in faces:
    gray_face = gray[(y):(y + h), (x):(x + w)]
    gray_face = cv2.resize(gray_face, (48, 48))
    gray_face = gray_face / 255.0
    gray_face = np.expand_dims(gray_face, 0)
    gray_face = np.expand_dims(gray_face, -1)
    emotion_label_arg = np.argmax(emotion_classifier.predict(gray_face))
    emotion = emotion_labels[emotion_label_arg]
    cv2.rectangle(img, (x + 10, y + 10), (x + h - 10, y + w - 10),
                  (255, 255, 255), 2)
    font = cv2.FONT_HERSHEY_DUPLEX
    cv2.putText(img, emotion, (x+h, y), font, 1.0, color)

cv2.imshow("Image", img)
cv2.waitKey(0)
cv2.destroyAllWindows()