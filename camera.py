import cv2
from imutils.video import WebcamVideoStream
from imutils.video import FPS
import imutils
import numpy as np
import time
import dlib
import cv2
import os
import math
import face_recognition
import pickle
from collections import deque
import pandas as pd
from datetime import datetime
import mediapipe as mp
import requests
import psutil

class VideoCamera(object):
    def __init__(self):
        # Using OpenCV to capture from device 0. If you have trouble capturing
        # from a webcam, comment the line below out and use a video file
        # instead.
        url = "rtsp://admin:admin321!!@192.168.10.33:554/ch01/0"
        self.video = WebcamVideoStream(src=0).start()
        print("Thread Started")
        threads_count = psutil.cpu_count() / psutil.cpu_count(logical=False)
        print("No:"+str(threads_count))
        self.pTime = 0
        self.users={}

        path = "/home/mynuddin-workstation/face_rec/Training_images"


        self.known_face_encodings=[]
        self.known_face_names=[]
        if not os.path.exists("/home/mynuddin-workstation/face_rec/encodings-m.pkl"):
            #my_list = os.listdir('/home/face_rec/known_images')
            MyList = os.listdir(path)
            for i in range(len(MyList)):
                if(MyList[i]!=".ipynb_checkpoints"):
                    #image=face_recognition.load_image_file("/home/face_rec/known_images/"+my_list[i]+"/01.jpg")
                    currentimage = cv2.imread(f'{path}/{MyList[i]}')
                    print(MyList[i])
                    img = cv2.cvtColor(currentimage, cv2.COLOR_BGR2RGB)
                    encode_img = face_recognition.face_encodings(img)[0]
                    #face_encoding = face_recognition.face_encodings(image,num_jitters=100)[0]
                    self.known_face_encodings.append(encode_img)
                    self.known_face_names.append(os.path.splitext(MyList[i])[0])

            with open('/home/mynuddin-workstation/face_rec/encodings-m.pkl','wb') as f:
                pickle.dump([self.known_face_encodings,self.known_face_names], f)
        else:
            with open('/home/mynuddin-workstation/face_rec/encodings-m.pkl', 'rb') as f:
                self.known_face_encodings ,self.known_face_names = pickle.load(f)

        self.mpFaceDetection = mp.solutions.face_detection
		# mpDraw = mp.solutions.drawing_utils
        self.faceDetection = self.mpFaceDetection.FaceDetection(model_selection=1)
        self.distances = []

        self.tTime = 0.0
        self.pTime = 0
        self.pName = {}
        self.timer = 0.0
        self.isRequest = False


    def __del__(self):
        #self.video.release()
        self.video.stopped=True

    def picture_from_frame(self,frame,name = "unknown"):
        this_time = datetime.now().isoformat(timespec='minutes')
        known_dir="/home/mynuddin-workstation/face_rec/captured_known_images/"+name
        # cap = gen_capture(url=0)
        if not (os.path.isdir(known_dir)):
            mode = 0o777
            os.makedirs(known_dir,mode)

        file_path = known_dir+'/'+this_time+'.jpg'
        # print()
        cv2.imwrite(file_path,frame)
        return file_path

    def get_frame(self):
        img = self.video.frame
        if (time.time()-self.timer)>=2:
            self.pName = []
        
        imgResize = cv2.resize(img,(0,0) , None , 0.25, 0.25)
        imgResize_RGB = cv2.cvtColor(imgResize, cv2.COLOR_BGR2RGB)

        face_Currentframe = face_recognition.face_locations(imgResize_RGB)
        encode_Currentframe = face_recognition.face_encodings(imgResize_RGB , face_Currentframe)

        font = cv2.FONT_HERSHEY_DUPLEX
        # cv2.putText(self.video.frame, "HI", (50 + 6, 50 - 6), font, 1.0, (255, 255, 255), 1)
        cTime = time.time()
        fps = 1/(cTime-self.pTime)
        self.pTime = cTime

        cv2.putText(img, "FPS: {:.2f}".format(fps), (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        

        for encodeface , faceLoc in zip(encode_Currentframe,face_Currentframe):
            print(faceLoc)
            matches = face_recognition.compare_faces(self.known_face_encodings, encodeface)
            print(matches)
            faceDis = face_recognition.face_distance(self.known_face_encodings, encodeface)
        
            confidence=min(faceDis)
            if min(faceDis) < 0.45: 
        
                matchIndex = np.argmin(faceDis)
                if matches[matchIndex]:
                    
                    name = self.known_face_names[matchIndex].upper()
                    if name not in self.users:
                        self.users[name]=1
                    else:
                        self.users[name]=self.users[name]+1
                    

                    print(self.users)
                    print(confidence)
                    
                    
               

                
                print(self.users[name])
                if(int(self.users[name])>=7):
                    try:
                        print('requesting... for name - {} id - {}'.format(name.split('-')[0],name.split('-')[1]))
                        self.users[name]=0
                    
                        #self.picture_from_frame(img, name = name, confidence = faceDis[matchIndex])
                        requests.get('http://192.168.10.87:8080?id={}'.format(name.split('-')[1]))

                    except Exception as ex:
                        template = "An exception of type {0} occurred. Arguments:\n{1!r}"
                        message = template.format(type(ex).__name__, ex.args)
                        print(message)
                        pass
                    
                    
                y1,x2,y2,x1=faceLoc
                y1,x2,y2,x1 = y1*4,x2*4,y2*4,x1*4
                cv2.rectangle(img, (x1,y2-35), (x2,y2), (0,255,0), cv2.FILLED)
                cv2.putText(img, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
                cv2.putText(img, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
            


            #cv2.imshow('Webcam', img)
            # key = cv2.waitKey(1)
            # if key == ord("q"):
                #         break
        # We are using Motion JPEG, but OpenCV defaults to capture raw images,
        # so we must encode it into JPEG in order to correctly display the
        # video stream.
        ret, jpeg = cv2.imencode('.jpg', img)
        print(ret)
        #jpeg = cv2.resize(img, (640,480))
        return jpeg.tobytes()