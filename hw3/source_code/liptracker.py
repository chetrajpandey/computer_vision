import numpy as np
import cv2

# multiple cascades: https://github.com/Itseez/opencv/tree/master/data/haarcascades
faceCascade = cv2.CascadeClassifier('data/haarcascade_frontalface_default.xml')
smileCascade = cv2.CascadeClassifier('data/haarcascade_mcs_mouth.xml')
 
cap = cv2.VideoCapture(0)
cap.set(3,1080) # set Width
cap.set(4,720) # set Height

while True:
    ret, img = cap.read()
    img = cv2.flip(img, 1)

    grayscaled_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(grayscaled_img, minSize=(100,100))

    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
        roi_gray =  grayscaled_img[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]
        
        lips = smileCascade.detectMultiScale(roi_gray, scaleFactor= 1.5, minNeighbors=15, minSize=(40,40))
        
        for (xx, yy, ww, hh) in lips:
            # cv2.rectangle(roi_color, (xx, yy), (xx + ww, yy + hh), (0, 0, 255), 2)
            center = (int(xx + ww/2), int(yy + hh/2))
            axes = (int(ww/2), int(hh/2))

            cv2.ellipse(roi_color, center, axes, 0, 0, 360, (0, 0, 255), 5)
               
        cv2.imshow('FaceLipTracker', img)

    k = cv2.waitKey(30) & 0xff
    if k == 27: # press 'ESC' to quit
        break

cap.release()
cv2.destroyAllWindows()