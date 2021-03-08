import cv2

def discern(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cap = cv2.CascadeClassifier(
        "D:\python3\Lib\site-packages\opencv\data\haarcascades\haarcascade_frontalface_default.xml"
    )
    faceRects = cap.detectMultiScale(
        gray, scaleFactor=1.2, minNeighbors=3, minSize=(50, 50))
    if len(faceRects):
        for faceRect in faceRects:
            x, y, w, h = faceRect
            cv2.rectangle(img, (x, y), (x + h, y + w), (0, 255, 0), 2)  # rectangle face
    cv2.imshow("Image", img)

# 0 means first camera
cap = cv2.VideoCapture(0)
while (1):  # for each frame
    ret, img = cap.read()
#     cv2.imshow("Image", img)
    discern(img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()  # release camera
cv2.destroyAllWindows()  # release windows

