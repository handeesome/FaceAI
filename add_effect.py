import cv2

# OpenCV facial regonization classifier
classifier = cv2.CascadeClassifier(
    "D:\python39\Lib\site-packages\opencv\data\haarcascades\haarcascade_frontalface_default.xml"
)

img = cv2.imread("img/face_recognition/shuai.jpg")  # read image
imgCompose = cv2.imread("img/compose/maozi.png") # this picture must be black as background

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # convert to gray
color = (0, 255, 0)  # define color
faceRects = classifier.detectMultiScale(
    gray, scaleFactor=1.2, minNeighbors=3, minSize=(32, 32))
if len(faceRects):  # detect face if greater than 0
    for faceRect in faceRects:  
        x, y, w, h = faceRect
        sp = imgCompose.shape
        imgComposeSizeH = int(sp[0]/sp[1]*w)
        if imgComposeSizeH>(y-20):
            imgComposeSizeH=(y-20)
        imgComposeSize = cv2.resize(imgCompose,(w, imgComposeSizeH), interpolation=cv2.INTER_NEAREST)
        top = (y-imgComposeSizeH-20)
        if top<=0:
            top=0
        rows, cols, channels = imgComposeSize.shape
        roi = img[top:top+rows,x:x+cols]

        # Now create a mask of logo and create its inverse mask also
        img2gray = cv2.cvtColor(imgComposeSize, cv2.COLOR_RGB2GRAY)
        ret, mask = cv2.threshold(img2gray, 10, 255, cv2.THRESH_BINARY) 
        mask_inv = cv2.bitwise_not(mask)

        # Now black-out the area of logo in ROI
        img1_bg = cv2.bitwise_and(roi, roi, mask=mask_inv)

        # Take only region of logo from logo image.
        img2_fg = cv2.bitwise_and(imgComposeSize, imgComposeSize, mask=mask)

        # Put logo in ROI and modify the main image
        dst = cv2.add(img1_bg, img2_fg)
        img[top:top+rows, x:x+cols] = dst

cv2.imshow("image", img) 
cv2.waitKey(0)
cv2.destroyAllWindows()
