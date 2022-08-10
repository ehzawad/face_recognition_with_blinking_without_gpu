from flask import Flask, render_template, Response
from camera import VideoCamera
from asyncore import write
from glob import glob
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
import datetime
import mediapipe as mp
import requests



def twoArgs(arg1):
    arg1.release()

def gen_capture(stream = None, url = 0,fps=None):
    # time.sleep(1.0)
    if(fps!=None):
        fps.stop()
        print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
        print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))
    if stream is None:
        stream = cv2.VideoCapture(url)
        stream.set(cv2.CAP_PROP_BUFFERSIZE,1)
        return stream
    else:
        stream.release()
        stream = cv2.VideoCapture(url)
        stream.set(cv2.CAP_PROP_BUFFERSIZE,1)
        return stream

known_face_encodings=[]
known_face_names=[]
if not os.path.exists("encodings.pkl"):
    my_list = os.listdir('known_images')
    for i in range(len(my_list)):
        if(my_list[i]!=".ipynb_checkpoints"):

            image=face_recognition.load_image_file("known_images/"+my_list[i]+"/01.jpg")
            print(my_list[i])
            face_encoding = face_recognition.face_encodings(image,num_jitters=100)[0]
            known_face_encodings.append(face_encoding)
            known_face_names.append(my_list[i])

    with open('encodings.pkl','wb') as f:
        pickle.dump([known_face_encodings,known_face_names], f)
else:
    with open('encodings.pkl', 'rb') as f:
        known_face_encodings ,known_face_names = pickle.load(f)
mpFaceDetection = mp.solutions.face_detection
mpDraw = mp.solutions.drawing_utils
faceDetection = mpFaceDetection.FaceDetection()
distances = []


app = Flask(__name__)

@app.route('/')
def index():
    
    return render_template('index.html')

def gen(camera):
    n=0
    while True:
        frame = camera
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

def genResponse(frame):
    # if not rgb_frame:
    #     frame  = gen(VideoCamera())
    # else:
    #     frame = rgb_frame
        
    resp = Response(gen(frame),
                    mimetype='multipart/x-mixed-replace; boundary=frame')
    resp.headers['Access-Control-Allow-Origin'] = '*'
    return resp


def face_recognize(url = "rtsp://admin:admin321!!@192.168.10.33:554/ch01/0"):
    testDf = pd.DataFrame(columns=known_face_names)
    video_capture = gen_capture(url=url)
    out = cv2.VideoWriter('output.mp4', cv2.VideoWriter_fourcc('M','J','P','G'), 200.0, (int(video_capture.get(3)), int(video_capture.get(4))))
    tTime = 0.0
    pTime = 0
    pName = []
    timer = 0.0
    isRequest = False
    while True:

        ret, frame = video_capture.read()
        if frame is not None:
            frame = cv2.resize(frame, (640,480))   
            # frame = cv2.resize(frame, (800,600))
            # frame = cv2.resize(frame, (0, 0), fx=0.3, fy=0.3)

        #time out
        if not ret:
            # time.sleep(1.0)
            video_capture = gen_capture(stream=video_capture,url=url)
            ret, frame = video_capture.read()
            frame = cv2.resize(frame, (640,480))

        rgb_frame = frame        
        results = faceDetection.process(rgb_frame)
        face_locations = []
        if (time.time()-timer)>=2:
            print('clearing array')
            pName = []
        if results.detections:            
            timer = time.time()
            if tTime == 0.0:
                tTime = time.time()
            for id,detection in enumerate(results.detections):
                bBoxC=detection.location_data.relative_bounding_box
                ih,iw,ic=rgb_frame.shape
                bBox = int(bBoxC.xmin*iw),int(bBoxC.ymin*ih),int(bBoxC.width*iw),int(bBoxC.height*ih)
                left,top,right,bottom = bBox[1],bBox[0]+bBox[2],bBox[1]+bBox[3],bBox[0]
                tup=(left,top,right,bottom)
                face_locations.append(tup)


        # print(face_locations)

        cTime = time.time()
        fps = 1/(cTime-pTime)
        pTime = cTime

        cv2.putText(rgb_frame, "FPS: {:.2f}".format(fps), (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        # Find all the faces and face enqcodings in the frame of video
        # face_locations = face_recognition.face_locations(rgb_frame,number_of_times_to_upsample=1)

        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)


        # Loop through each face in this frame of video
        faces = []
        count = 0
        dTime = time.time()


        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            # See if the face is a match for the known face(s)
            # count+=1         
            print(face_locations)
            name = "Unknown"
            #single face
            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            if face_distances[best_match_index] < 0.6:
                name1 = known_face_names[best_match_index]
                if len(pName)>=30:
                    print("array clearence")
                    # for _ in range(21): pName.pop(0)
                    pName = pName[-1:-9:-1]

                pName.append(name1)
            
            #add timer to best optimization
            if(len(pName)>=5):
                name = max(pName, key=pName.count)
                # print(len(pName))
                print(name)

            print('len {}'.format(len(pName)))
            if(len(pName)==7):
                try:
                    # employee_id = name.split('-')[-1]
                    # print('time needed before request {}'.format(time.time()-timer))
                    # print('this time before request {}'.format(datetime.datetime.now()))
                    print('requesting... for name - {} id - {}'.format(name.split('-')[0],name.split('-')[-1]))
                    requests.get('http://192.168.10.87:8080?id={}'.format(name.split('-')[-1]))

                except:
                    pass
                
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

            # Draw a label with a name below the face
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            just_name = name.split('-')[0]
            cv2.putText(frame, just_name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)
            ret, jpeg = cv2.imencode('.jpg', frame)
         .   vid = cv2.flip(frame,1)
            out.write(vid)

        cv2.imshow('FaceDetector', rgb_frame)
        #resp = genResponse(frame)
        
        # res = yield (b'--frame\r\n'
        #        b'Content-Type: image/jpeg\r\n\r\n' + cv2.get_frame() + b'\r\n\r\n')
        # resp = Response(res,
        #             mimetype='multipart/x-mixed-replace; boundary=frame')
        # resp.headers['Access-Control-Allow-Origin'] = '*'
        # return resp
        key=cv2.waitKey(1)
    #     # Hit 'q' on the keyboard to quit!
        if key%256 == 27:
            cv2.destroyAllWindows()

            break

    # # Release handle to the webcam

    video_capture.release()
    out.release()
    cv2.destroyAllWindows()


# @app.route('/video_feed')
# def video_feed():    
#     url = "rtsp://admin:admin321!!@192.168.10.33:554/ch01/0"
#     video_capture = cv2.VideoCapture(url)
#     while True:
#         grabbed, frame = video_capture.read()

#         image = frame
#         ret, jpeg = cv2.imencode('.jpg', image)
#         re = yield(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + 
# 			bytearray(jpeg) + b'\r\n')
#         resp = Response(re,
#                     mimetype='multipart/x-mixed-replace; boundary=frame')
#         resp.headers['Access-Control-Allow-Origin'] = '*'
#         return resp

# if __name__ == '__main__':
#     app.run(host='192.168.10.89', debug=False)