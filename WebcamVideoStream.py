# import the necessary packages
from threading import Thread
import cv2
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

class WebcamVideoStream:
	
	def __init__(self, src=0):


		self.self.known_face_encodings=[]
		self.self.known_face_names=[]
		if not os.path.exists("encodings.pkl"):
			my_list = os.listdir('known_images')
			for i in range(len(my_list)):
				if(my_list[i]!=".ipynb_checkpoints"):

					image=face_recognition.load_image_file("known_images/"+my_list[i]+"/01.jpg")
					print(my_list[i])
					face_encoding = face_recognition.face_encodings(image,num_jitters=100)[0]
					self.known_face_encodings.append(face_encoding)
					self.known_face_names.append(my_list[i])

			with open('encodings.pkl','wb') as f:
				pickle.dump([self.known_face_encodings,self.known_face_names], f)
		else:
			with open('encodings.pkl', 'rb') as f:
				self.known_face_encodings ,self.known_face_names = pickle.load(f)
		self.mpFaceDetection = mp.solutions.face_detection
		# mpDraw = mp.solutions.drawing_utils
		self.faceDetection = self.mpFaceDetection.FaceDetection()
		self.distances = []

		self.tTime = 0.0
		self.pTime = 0
		self.pName = []
		self.timer = 0.0
		self.isRequest = False

		# initialize the video camera stream and read the first frame
		# from the stream


		self.stream = cv2.VideoCapture(src)
		(self.grabbed, self.frame) = self.stream.read()
		font = cv2.FONT_HERSHEY_DUPLEX
		cv2.putText(self.frame, "HI", (50 + 6, 50 - 6), font, 1.0, (255, 255, 255), 1)

		# initialize the variable used to indicate if the thread should
		# be stopped
		self.stopped = False
	def start(self):
		# start the thread to read frames from the video stream
		Thread(target=self.update, args=()).start()
		return self

	def update(self):
		# keep looping infinitely until the thread is stopped
		while True:
			# if the thread indicator variable is set, stop the thread
			if self.stopped:
				return
			# otherwise, read the next frame from the stream
			(self.grabbed, self.frame) = self.stream.read()
			if self.frame is not None:
				self.frame = cv2.resize(self.frame, (640,480))
			
			rgb_frame = self.frame 
			results = self.faceDetection.process(rgb_frame)
			face_locations = []
			if (time.time()-timer)>=2:
				print('clearing array')
				self.pName = []
			if results.detections:            
				timer = time.time()
				if self.tTime == 0.0:
					self.tTime = time.time()
				for id,detection in enumerate(results.detections):
					bBoxC=detection.location_data.relative_bounding_box
					ih,iw,ic=rgb_frame.shape
					bBox = int(bBoxC.xmin*iw),int(bBoxC.ymin*ih),int(bBoxC.width*iw),int(bBoxC.height*ih)
					left,top,right,bottom = bBox[1],bBox[0]+bBox[2],bBox[1]+bBox[3],bBox[0]
					tup=(left,top,right,bottom)
					face_locations.append(tup)


			# print(face_locations)

			cTime = time.time()
			fps = 1/(cTime-self.pTime)
			self.pTime = cTime

			cv2.putText(rgb_frame, "FPS: {:.2f}".format(fps), (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

			# Find all the faces and face enqcodings in the frame of video
			# face_locations = face_recognition.face_locations(rgb_frame,number_of_times_to_upsample=1)

			## have to work for multi face
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
				face_distances = face_recognition.face_distance(self.known_face_encodings, face_encoding)
				best_match_index = np.argmin(face_distances)
				if face_distances[best_match_index] < 0.6:
					name1 = self.known_face_names[best_match_index]
					if len(self.pName)>=30:
						print("array clearence")
						# for _ in range(21): self.pName.pop(0)
						self.pName = self.pName[-1:-9:-1]

					self.pName.append(name1)
				
				#add timer to best optimization
				if(len(self.pName)>=5):
					name = max(self.pName, key=self.pName.count)
					# print(len(self.pName))
					print(name)

				print('len {}'.format(len(self.pName)))
				if(len(self.pName)==7):
					try:
						# employee_id = name.split('-')[-1]
						# print('time needed before request {}'.format(time.time()-timer))
						# print('this time before request {}'.format(datetime.datetime.now()))
						print('requesting... for name - {} id - {}'.format(name.split('-')[0],name.split('-')[-1]))
						requests.get('http://192.168.10.87:8080?id={}'.format(name.split('-')[-1]))

					except:
						pass
					
				cv2.rectangle(self.frame, (left, top), (right, bottom), (0, 0, 255), 2)

				# Draw a label with a name below the face
				cv2.rectangle(self.frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
				font = cv2.FONT_HERSHEY_DUPLEX
				just_name = name.split('-')[0]
				cv2.putText(self.frame, just_name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)
				ret, jpeg = cv2.imencode('.jpg', self.frame)
				vid = cv2.flip(self.frame,1)
				# out.write(vid)

			#cv2.imshow('FaceDetector', rgb_frame)
			# self.frame = rgb_frame
            
	def read(self):
		# return the frame most recently read
		return self.frame
	def stop(self):
		# indicate that the t
		return self.frame