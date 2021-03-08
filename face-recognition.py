import cv2
import face_recognition
import os

path = "img/face_recognition"  # model data image directory
cap = cv2.VideoCapture(0)
total_image_name = []
total_face_encoding = []
for fn in os.listdir(path):  # fn for file name
    print(path + "/" + fn)
    total_face_encoding.append(
        face_recognition.face_encodings(
            face_recognition.load_image_file(path + "/" + fn))[0])
    fn = fn[:(len(fn) - 4)]  # take only the image name
    total_image_name.append(fn)  # image name list
while (1):
    ret, frame = cap.read()
    # for knowing every face and face_encodings
    face_locations = face_recognition.face_locations(frame)
    face_encodings = face_recognition.face_encodings(frame, face_locations)
    # iterate every face in one frame
    for (top, right, bottom, left), face_encoding in zip(
            face_locations, face_encodings):
        # check for match
        for i, v in enumerate(total_face_encoding):
            match = face_recognition.compare_faces(
                [v], face_encoding, tolerance=0.5)
            name = "Unknown"
            if match[0]:
                name = total_image_name[i]
                break
        # rectangle face
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
        # lebel under face
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255),
                      cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0,
                    (255, 255, 255), 1)
    # result showing
    cv2.imshow('Video', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()