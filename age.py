from keras.models import load_model
from time import sleep
#from keras.preprocessing.image import img_to_array
from tensorflow.keras.utils import img_to_array

from keras.preprocessing import image
import cv2
import numpy as np

face_classifier=cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_default.xml')
age_model = load_model('age_model_32epochs.h5')
gender_model = load_model('gender_model_32epochs.h5')

gender_labels = ['Male', 'Female']


frame=cv2.imread("Mynuddin.jpg")
#labels=[]
Ages=[]
gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
faces=face_classifier.detectMultiScale(gray,1.3,5)

for (x,y,w,h) in faces:
    cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
   
    #Gender
    roi_color=frame[y:y+h,x:x+w]
    roi_color=cv2.resize(roi_color,(200,200),interpolation=cv2.INTER_AREA)
    gender_predict = gender_model.predict(np.array(roi_color).reshape(-1,200,200,3))
    gender_predict = (gender_predict>= 0.5).astype(int)[:,0]
#    print(gender_predict)
    gender_label=gender_labels[gender_predict[0]]
   

#    gender_label_position=(x+50,y-10)
#    cv2.putText(frame,gender_label,gender_label_position,cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
   
    #Age
    age_predict = age_model.predict(np.array(roi_color).reshape(-1,200,200,3))
    age = round(age_predict[0,0])
#    Ages.append(age)
#    age_label_position=(x+40,y+h+40)
#    cv2.putText(frame,"Age="+str(age),age_label_position,cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
   

#    cv2.imwrite('family_Age_Gender_Detect.jpg', frame)

#cv2.imshow('Age  Detect', frame)

print(age)
print(gender_label)
#print(gender_predict)