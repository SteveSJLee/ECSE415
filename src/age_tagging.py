
# coding: utf-8

# In[ ]:


import numpy as np
import cv2

def age_tagging(img_gray):
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x,y,w,h) in faces:
        cv2.rectangle(img_gray,(x,y),(x+w,y+h),(0,0,255),10)
        face_crop = img_gray[x:x+w,y:y+h]
        WKS = []
        
        eyes,left_eye,right_eye = eye_detect(face_crop)
        mouth_pos,nose_pos,mouth = nose_mouth_loc(face_crop, left_eye,right_eye)
        wrinkles, R_em, R_enm = feature_extraction(face_crop, eyes, left_eye, right_eye, nose_pos, mouth_pos, mouth)
        WKS.append(np.array(wrinkles).flatten())
        age = scl.train_model.predict(WKS)
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(img,age,(x,y), font, 1, (0,0,255), 10, cv2.LINE_AA)
        
img = cv2.imread('group.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
age_tagging(gray)
plt.imshow(gray, cmap="gray")
plt.show()

