import grovepi
import time
import picamera
import pigpio
import socket
import sys
import time
from imutils.object_detection import non_max_suppression
import numpy as np
import imutils
import cv2
import sys
import Queue
import os

sys.path.append('/usr/local/lib/python2.7/site-packages')

ultrasonic_ranger = [2, 3, 7, 8]	#ultrasonic ports
camera_number = 4				   #number of cameras
ultrasonic_number = 4
buzzer = 0							#buzzer ports
approach_pics = 8							#pictures taken each time
passing_pics = 4
leaving_pics = 2
video_time = 2						#length of video taken each time
history_data_max = 5
history_diff_num = history_data_max - 2
ultrasonic_average = 5			  #average of 5 reads
img_cnt = 0
vid_cnt = 0
cam = 1								#default camera number
PORT = 8080							#Socket server port
BUF_SIZE=1024
ul_history = [[0 for n in range(history_data_max)] for m in range(camera_number)]

# initialize the HOG descriptor/person detector
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
face_cascade = cv2.CascadeClassifier("/usr/share/opencv/haarcascades/haarcascade_frontalface_default.xml")
event = 1
myname = socket.getfqdn(socket.gethostname())
myaddr = socket.gethostbyname(myname)
HOST= myaddr
ADDR=(HOST,PORT)
print("Machine address is %s:%s" %(HOST, PORT))
server = socket.socket(socket.AF_INET,socket.SOCK_STREAM)
try:
	server.bind(ADDR)
except socket.error as msg:
	print("Bind failed, error code: %s:%s" %(msg[0], msg[1]))
	sys.exit()
server.listen(1)

grovepi.pinMode(buzzer,"OUTPUT")
pigpio.exceptions = False
pi = pigpio.pi()
pi.set_mode(4, pigpio.OUTPUT)
pi.set_mode(17, pigpio.OUTPUT)
pi.set_mode(18, pigpio.OUTPUT)
pi.write(4, 0)
pi.write(17, 0)
pi.write(18, 1)

with picamera.PiCamera() as camera:

	def ultrasonic_history(ranger_number):
		ul_history [ranger_number][history_data_max - 1] = ultrasonicRead(ranger_number)
		return ul_history[ranger_number]

	def ultrasonicRead(ranger_number):
		data = [0 for n in range(ultrasonic_average)]
		for n in range(0, ultrasonic_average):
			time.sleep(0.02)
			try:
				data[n] = grovepi.ultrasonicRead(ultrasonic_ranger[ranger_number - 1])
			except TypeError:
				print ("Ultrasonic on D%d: Typr error" % ultrasonic_ranger[ranger_number - 1])
				return None
			except IOError:
				print ("Ultrasonic on D%d: Read error" % ultrasonic_ranger[ranger_number - 1])
				return None
		data.sort()
		print ("Ultrasonic on D%d: Distance is: %dcm" % (ultrasonic_ranger[ranger_number - 1], data[2]))
		return data[2]

	def detectHuman(image_name):
		# load the image and resize it to (1) reduce detection time
		# and (2) improve detection accuracy
		image = cv2.imread(image_name)
		image = imutils.resize(image, width=min(400, image.shape[1]))
		orig = image.copy()
		# detect people in the image
		(rects, weights) = hog.detectMultiScale(image, winStride=(4, 4), padding=(8, 8), scale=1.05)
		# apply non-maxima suppression to the bounding boxes using a
		# fairly large overlap threshold to try to maintain overlapping
		# boxes that are still people
		rects = np.array([[x, y, x + w, y + h] for (x, y, w, h) in rects])
		pick = non_max_suppression(rects, probs=None, overlapThresh=0.65)
		cv2.waitKey(0)
		return pick #include xA, yA, xB, yB

	def detectFaces(image_name):
		img = cv2.imread(image_name)
		if img.ndim == 3:
			gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
		else:
			gray = img
		faces = face_cascade.detectMultiScale(gray, 1.2, 5)#1.3 and 5 are the smallest and biggest window for detected feature
		result = []
		for (x,y,width,height) in faces:
			result.append((x,y,x+width,y+height))
		return result

	def cam_change(camera_n = 0):
		global cam
		if camera_n == 0 :
			cam += 1
			if cam > 4 : #4:
				cam = 1
		else:
			if cam == camera_n:
				return
			cam = camera_n
		time.sleep(0.007)   # SD Card Bandwidth Correction Delay
		if cam == 1:
			# CAM 1 for A Jumper Setting
			pi.write(4, 0)
			pi.write(17, 0)
			pi.write(18, 1)
		elif cam == 2:
			# CAM 2 for A Jumper Setting
			pi.write(4, 1)
			pi.write(17, 0)
			pi.write(18, 1)
		elif cam == 3:
			# CAM 3 for A Jumper Setting
			pi.write(4, 0)
			pi.write(17, 1)
			pi.write(18, 0)
		elif cam == 4:
			# CAM 4 for A Jumper Setting
			pi.write(4, 1)
			pi.write(17, 1)
			pi.write(18, 0)
		time.sleep(0.007)   # SD Card Bandwidth Correction Delay

	def image_filenames(camera_n, frames):
		global img_cnt
		frame = 0
		while frame < frames:
			img_cnt += 1
			cam_change(camera_n)		# Switching Camera
			yield "record/image%02d_c%d.jpg" % (img_cnt, camera_n)
			frame += 1

	def video_filenames(camera_n):
		global vid_cnt
		vid_cnt += 1
		cam_change(camera_n)		# Switching Camera
		return "record/video%02d_c%d.mp4" % (vid_cnt, camera_n)

	def isApproach(ranger_number):
		his = ultrasonic_history(ranger_number)
		for i in range(history_diff_num):
			if (his[history_data_max - i - 1] - his[history_data_max - i - 2]) > 0:
				return False
		return True

	def isHuman(camera_number):
		camera.resolution = (1280, 720)
		camera.framerate = 30
		#camera.capture_sequence(image_filenames(i, approach_pics), use_video_port=False)
		camera.capture("record/temp.jpg", format = "jpeg", use_video_port=False)
		if detectHuman("record/temp.jpg") == None and detectFaces("record/temp.jpg") == None:
			human = False
		else:
			human = True
		os.remove("record/temp.jpg")
		return human
		
	def isPassingby(camera_number):
		detect = MotionDetectorInstantaneous(doRecord=True)
		detect.MotionDetector.run()

	print("Initializing camera..")
	camera.resolution = (1280, 720)
	camera.framerate = 30
	camera.start_preview()

	print("Waiting for alarm client connection, Please run SocketClient.py on host PC")
	conn,addr = server.accept()
	print("Alarm server connected by:%s:%s" %(addr[0], addr[1]))

	while True:
		for i in range(1, camera_number + 1):
			#loop_start = float(time.time() * 1000)

			ultrasonic_distance = ultrasonic_history(i)

			if (event == 3 or event == 4) and isApproach() and isHuman():
				event = 4
				print("Suspecious target in front of camera %d, take video for %.1fs" %(i, video_time))
				camera.resolution = (1280, 720)
				camera.framerate = 30
				print("Sending alarm to host PC")
				conn.send("Alarm from camera %d: Object is %dcm in the front" %(i, ultrasonic_distance))
				respon = conn.recv(BUF_SIZE)
				if respon:
					print("Host PC:%s" %respon)
				grovepi.digitalWrite(buzzer,1)
				camera.start_recording(video_filenames(i), format='h264')
				time.sleep(video_time)
				camera.stop_recording()
				grovepi.digitalWrite(buzzer,0)
			elif event != 4 and isApproach():
				event = 3
				print("Object in front of camera %d is approaching, take %d pictures" %(i, approach_pics)
				camera.resolution = (1280, 720)
				camera.framerate = 30
				#start = time.time()
				camera.capture_sequence(image_filenames(i, approach_pics), use_video_port=False)
				#finish = time.time()
				#print('Captured %d frames at total %.2ffps' % (approach_pics, approach_pics / (finish - start)))
			elif (event == 1 or event == 2 or event == 5) and isPassingby():
				event = 2
				print("Object in front of camera %d is passing by, take %d pictures" %(i, passing_pics)
				camera.resolution = (640, 360)
				camera.framerate = 30
				#start = time.time()
				camera.capture_sequence(image_filenames(i, passing_pics), use_video_port=False)
				#finish = time.time()
				#print('Captured %d frames at total %.2ffps' % (passing_pics, passing_pics / (finish - start)))
			elif event != 1 and isLeaving():
				event = 5
				print("Object in front of camera %d is leaving, take %d pictures" %(i, leaving_pics)
				camera.resolution = (640, 360)
				camera.framerate = 30
				#start = time.time()
				camera.capture_sequence(image_filenames(i, leaving_pics), use_video_port=False)
				#finish = time.time()
				#print('Captured %d frames at total %.2ffps' % (approach_pics, approach_pics / (finish - start)))
			elif event != 3 and event != 4:
				event = 1
				print("Nothing is in front of camera %d" %i)
				
			#loop_end = float(time.time() * 1000)
			#print("Loop execution time is %fms" %(loop_end-loop_start))
conn.close()
pi.stop()
