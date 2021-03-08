import cv2

filepath = "img/group/group5.jpg"
img = cv2.imread(filepath) # read img
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # convert gray

#OpenCV facial recognition classifiser
classifier = cv2.CascadeClassifier(
    "D:\python36\Lib\site-packages\opencv\data\haarcascades\haarcascade_frontalface_default.xml"
)
color = (0, 255, 0) # define color
#call facial recognition
faceRects = classifier.detectMultiScale(
    gray, scaleFactor=1.2, minNeighbors=3, minSize=(32, 32))
if len(faceRects):   # detects face if greater than 0
    for faceRect in faceRects: # separate rectangle one face
        x, y, w, h = faceRect
        #rectangle face
        cv2.rectangle(img, (x, y), (x+h, y+w), color, 2)
        #left eye
        cv2.circle(img, (x + w//4, y + h//4 + 30), min(w//8, h//8), color)
        #right eye
        cv2.circle(img, (x + 3*w//4, y + h//4 +30), min(w//8, h//8), color)
        #mouth
        cv2.rectangle(img, (x + 3*2//8, y + 3*h//4), 
                    (x + 5*2//8, y+7*h//8), color)

cv2.imshow("image", img) # show img
c = cv2.waitKey(0)

cv2.waitKey(0)
cv2.destroyAllWindows()