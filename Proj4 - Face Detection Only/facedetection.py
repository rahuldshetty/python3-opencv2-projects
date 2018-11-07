import numpy as np
import cv2

cap=cv2.VideoCapture(0)
face_cas = cv2.CascadeClassifier('face.xml')


while(True):
    ret,frame=cap.read()

    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

    

    faces = face_cas.detectMultiScale(gray,1.3,5)

    for (x,y,w,h) in faces:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
        
        
    cv2.imshow('frame',frame)

    if cv2.waitKey(1) & 0xff==ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
