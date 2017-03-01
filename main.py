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
import os
from PIL import Image,ImageDraw

sys.path.append('/usr/local/lib/python2.7/site-packages')

print "============================VIDEO SURVEILLANCE SYSTEM============================"

ultrasonic_ranger = [2, 3, 8, 7]	#ultrasonic ports
camera_number = 4					#number of cameras
ultrasonic_number = 4				#total ultrasonic number
buzzer = 4							#port number of buzzer
approach_pics = 4#8					#pictures taken each for approaching 
passing_pics = 2#4					#... 					passingby
leaving_pics = 1#2					#... 					leaving
video_time = 2						#length of video taken each time
history_data_max = 5				#length of ultrasonic history
history_diff_num = history_data_max - 2	#2 difference of ul his to determine approach or leave
ultrasonic_average = 3				#average of 5 reads
img_cnt = 0							#image file number start value
vid_cnt = 0							#video ...
cam = 1								#default camera number
PORT = 8080							#Socket server port
BUF_SIZE=1024
#motion_record_time = 1				#useless
#motion_frame_jump = 10				#useless
hd_resolution = (1280, 720)			# can be (1920, 1080)
sd_resolution = (640, 360)			#for low res
hd_frame_rate = 30
sd_frame_rate = 30
threshold = 5						#threshold for motion detect
pics = ["record/temp1.jpg", "record/temp2.jpg", "record/temp3.jpg"]					#temp picture name
im = 0								#temp value indicates number of motion detection loops

ul_history = [[0 for n in range(history_data_max)] for m in range(camera_number)]	#init ul his
events = [1 for m in range(camera_number)]											#init events
passs = [False for n in range(camera_number)]		#init blah blah
apps = [False for n in range(camera_number)]
hums = [False for n in range(camera_number)]
leas = [False for n in range(camera_number)]
alarms = [False for n in range(camera_number)]
info_lines = 5										#lines for info display
infos = [["\t\t" for n in range(info_lines)] for m in range(camera_number)]	#still init
global_debug = ""									#global string for debug print

# initialize the HOG descriptor/person detector
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
#face_cascade = cv2.CascadeClassifier("/usr/share/opencv/haarcascades/haarcascade_frontalface_default.xml")
face_cascade = cv2.CascadeClassifier("Dependencies/haarcascade_frontalface_default.xml")

#Socket connection
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

#init GPIO
grovepi.pinMode(buzzer,"OUTPUT")
pigpio.exceptions = False
pi = pigpio.pi()
pi.set_mode(4, pigpio.OUTPUT)
pi.set_mode(17, pigpio.OUTPUT)
pi.set_mode(18, pigpio.OUTPUT)
pi.write(4, 0)
pi.write(17, 0)
pi.write(18, 1)

def ultrasonicHistory(ranger_number, update = True):
	if update :
		u = None
		while u == None:
			u = ultrasonicRead(ranger_number)
		ul_history[ranger_number - 1][0:history_data_max - 1] = ul_history[ranger_number - 1][1:history_data_max]		#shift to left
		ul_history[ranger_number - 1][history_data_max - 1] = u
	return ul_history[ranger_number - 1]


def ultrasonicRead(ranger_number):
	data = [0 for x in range(ultrasonic_average)]		#samples for get each value reduced from 5 to 3 
	for y in range(0, ultrasonic_average):
		time.sleep(0.02)
		try:
			data[y] = grovepi.ultrasonicRead(ultrasonic_ranger[ranger_number - 1])
		except TypeError:
			#print ("Ultrasonic on D%d: Typr error" % ultrasonic_ranger[ranger_number - 1])
			return None
		except IOError:
			#print ("Ultrasonic on D%d: Read error" % ultrasonic_ranger[ranger_number - 1])
			return None
	data.sort()
	#print ("Ultrasonic on D%d: Distance is: %dcm" % (ultrasonic_ranger[ranger_number - 1], data[int(ultrasonic_average/2)-1]))
	return data[int(ultrasonic_average/2)-1]

def detectHuman(image_name):
	# load the image and resize it to (1) reduce detection time
	# and (2) improve detection accuracy
	image = cv2.imread(image_name)
	image = imutils.resize(image, width=min(300, image.shape[1]))		#reduced from 400 to 300
	orig = image.copy()
	# detect people in the image
	(rects, weights) = hog.detectMultiScale(image, winStride=(4, 4), padding=(8, 8), scale=1.05)
	# apply non-maxima suppression to the bounding boxes using a
	# fairly large overlap threshold to try to maintain overlapping
	# boxes that are still people
	rects = np.array([[x, y, x + w, y + h] for (x, y, w, h) in rects])
	pick = non_max_suppression(rects, probs=None, overlapThresh=0.65)
	#cv2.waitKey(0)
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

def processImage(frame, frame1gray, frame2gray, res):
	global im
	cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY, frame2gray, 0)
	
	cv2.imwrite("record/debug/frame2gray_new_%d.jpg" %im, frame2gray)
	# Absdiff to get the difference between to the frames
	cv2.absdiff(frame1gray, frame2gray, res)
	cv2.imwrite("record/debug/res_diff_%d.jpg" %im, res)

	# Remove the noise and do the threshold
	cv2.blur(res, (5, 5), res) #
	#cv2.imwrite("record/debug/res_blur_%d.jpg" %im, res)

	kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
	cv2.morphologyEx(res, cv2.MORPH_OPEN, kernel, res)
	#cv2.imwrite("record/debug/res_m1_%d.jpg" %im, res)
	cv2.morphologyEx(res, cv2.MORPH_CLOSE, kernel, res)
	#cv2.imwrite("record/debug/res_m2_%d.jpg" %im, res)

	cv2.threshold(res, 25, 255, cv2.THRESH_BINARY_INV, res)		#threshold change from 10 to 25
	cv2.imwrite("record/debug/res_th_%d.jpg" %im, res)

def drawFaces(faces, image_name):
	global im
	if faces:
		img = Image.open(image_name)
		draw_instance = ImageDraw.Draw(img)
		for (x1,y1,x2,y2) in faces:
			draw_instance.rectangle((x1,y1,x2,y2), outline=(255, 0,0))
		img.save('record/debug/face%d.jpg' %im)

def somethingHasMoved(res, width, height):
	global global_debug
	nb = 0  # Will hold the number of black pixels
	#nb_pixels = width * height
	#print cv2.countNonZero(res)
	nb = res.size - cv2.countNonZero(res)		#change from for loop to this, same effect but much faster
	#for x in range(int(height)):  # Iterate the hole image
	#	for y in range(int(width)):
	#		if res[x, y] == 0.0:  # If the pixel is black keep it
	#			nb += 1
	avg = (nb * 100.0) / res.size  # Calculate the average of black pixel in the image
	global_debug += "\n" + ("Motion detetion black field: %d, Avg : %f, Threshold: %f" %(nb, avg, threshold))
	if avg > threshold:  # If over the ceiling trigger the alarm
		return True
	else:
		return False
'''
def detectMotion(video_name):
	capture = cv2.VideoCapture(video_name)
	#os.system("raspvid -o" + video_name)
	#capture = cv2.imread(video_name)
	#frame = []
	#for b in range( int (capture.get(cv2.CAP_PROP_FRAME_COUNT) / motion_frame_jump) ):
	for b in range( int(capture.get(cv2.CAP_PROP_FRAME_COUNT))):
		ret, frame = capture.read()
		#frame.append(f_temp)
		#capture.set(cv2.CAP_PROP_POS_FRAMES, capture.get(cv2.CAP_PROP_POS_FRAMES) + motion_frame_jump - 1)

		if b == 0:
			width = capture.get(cv2.CAP_PROP_FRAME_WIDTH)
			height = capture.get(cv2.CAP_PROP_FRAME_HEIGHT)
			#frame1gray = np.array((np.zeros(weight, height), np.uint8)
			frame1gray = np.zeros((height,width), np.uint8)
			#frame1gray = cv2.cv.CreateMat(height, width, cv2.CV_8U)  # Gray frame at t-1
			#frame1gray = np.array((height, width), np.uint8)
			#frame1gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
			cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY, frame1gray, 0)
			#res = cv2.CreateMat(height, width, cv2.CV_8U)
			res = np.zeros((height,width), np.uint8)
			#frame2gray = cv2.CreateMat(height, width, cv2.CV_8U)  # Gray frame at t
			#frame2gray = np.array((height, width, 1), np.uint8)#np.array((height, width), np.uint8)
			frame2gray = np.zeros((height,width), np.uint8)
		#frame2gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
		processImage(frame, frame1gray, frame2gray, res)

		if b!= 0 and somethingHasMoved(res, width, height):
			return False
		#cv2.Copy(frame2gray, frame1gray)
		frame1gray = frame2gray
	return True
'''
def detectMotion(pictures):
	global im
	for b in pictures:
		im += 1
		#loop_1 = float(time.time() * 1000)
		#print b
		frame = cv2.imread(b, cv2.IMREAD_COLOR)
		if b == pictures[0]:
			height, width, channels = frame.shape
			res = np.zeros((height,width), np.uint8)
			frame1gray = np.zeros((height,width), np.uint8)
			frame2gray = np.zeros((height,width), np.uint8)
			cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY, frame1gray, 0)
			
		#frame2gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
		processImage(frame, frame1gray, frame2gray, res)
		#cv2.imwrite("record/debug/orignal_%d.jpg" %(im), frame)
		#cv2.imwrite("record/debug/frame1gray_%d.jpg" %(im), frame1gray)
		#cv2.imwrite("record/debug/frame2gray_%d.jpg" %(im), frame2gray)
		#cv2.imwrite("record/debug/res_%d.jpg" %(im), res)
		#loop_2 = float(time.time() * 1000)
		#print ("motion1: " + str(loop_2 - loop_1))

		if b!= pictures[0] and somethingHasMoved(res, width, height):
			#loop_3 = float(time.time() * 1000)
			#print ("motion2: " + str(loop_3 - loop_2))
			#print ("motion_total: " + str(loop_3 - loop_1))
			return True
		frame1gray = frame2gray.copy()
		#loop_3 = float(time.time() * 1000)
		#print ("motion2: " + str(loop_3 - loop_2))
		#print ("motion_total: " + str(loop_3 - loop_1))
	return False

def camChange(camera_n = 0):
	global cam
	if camera_n == 0 :
		cam += 1
		if cam > 4 : #4:
			cam = 1
	else:
		if cam == camera_n:		#do not change when current camera = target camera
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

def imageFilenames(camera_n, frames):
	global img_cnt
	frame = 0
	while frame < frames:
		img_cnt += 1
		camChange(camera_n)		# Switching Camera
		yield "record/image%02d_c%d.jpg" % (img_cnt, camera_n)		#filename++
		frame += 1

def videoFilenames(camera_n):
	global vid_cnt
	vid_cnt += 1
	camChange(camera_n)		# Switching Camera
	return "record/video%02d_c%d.mp4" % (vid_cnt, camera_n)

def isApproach(ranger_number):
	his = ul_history[ranger_number - 1]
	if his[-1]<20:
		return True
	for z in range(history_diff_num):
		if (his[history_data_max - z - 1] - his[history_data_max - z - 2]) > 0:
			return False
	return True

def isLeaving(ranger_number):
	his = ul_history[ranger_number - 1]
	if his[-1]<20:
		return False
	for w in range(history_diff_num):
		if (his[history_data_max - w - 1] - his[history_data_max - w - 2]) < 0:
			return False
	return True

def isHuman(camera_number):
	global global_debug
	#camSD()
	#camChange(camera_number)
	#camera.capture("record/temp.jpg", format = "jpeg", use_video_port=True)
	#loop_1 = float(time.time() * 1000)
	detecth = detectHuman(pics[1])
	#loop_2 = float(time.time() * 1000)
	detectf = detectFaces(pics[1])
	drawFaces(detectf, pics[1])
	#loop_3 = float(time.time() * 1000)
	#if detecth == [] and detectf == []:
	if len(detecth) == 0 and len(detectf) == 0:
		human = False
	else:
		human = True
	#loop_4 = float(time.time() * 1000)
	#print loop_2 - loop_1
	#print loop_3 - loop_2
	#print loop_4 - loop_3
	global_debug += "\n" + ("human: " + str(detecth) + ", face: " + str(detectf) + ", ret: " + str(human))
	return human
	
def isPassingby(camera_number):
	#loop_1 = float(time.time() * 1000)
	#camSD()
	#camChange(camera_number)
	#camRecord(camera_number, motion_record_time, "record/temp.mp4")
	#ret = detectMotion("record/temp.mp4")
	#camera.capture_sequence(pics, use_video_port=True)
	ret = detectMotion(pics)
	#loop_4 = float(time.time() * 1000)
	#print ("total: " + str(loop_4 - loop_1))
	return ret
	
def UI(camera_number, event, ul_data, pas, app, hum, lea, ala, info):
	os.system("clear")
	arrow = ([["+", " ", " ", " "], [" ", "+", " ", " "], [" ", " ", "+", " "], [" ", " ", " ", "+"]])[camera_number - 1]
	event_dic = {1:"Clear scene", 2:"Passing by", 3:"Approaching", 4:"Suspecious", 5:"Leaving"}
	app_dic = {True:"APP", False:"   "}
	pas_dic = {True:"PAS", False:"   "}
	hum_dic = {True:"HUM", False:"   "}
	lea_dic = {True:"LEA", False:"   "}
	passs[camera_number - 1] = pas
	apps[camera_number - 1] = app
	hums[camera_number - 1] = hum
	leas[camera_number - 1] = lea
	alarms[camera_number - 1] = ala
	if len(info) < 15:
		infos[camera_number - 1][0] = info[0:len(info)]
		infos[camera_number - 1][0] += "	"
		if len(info) < 7:
			infos[camera_number - 1][0] += "	"
	if len(info) >= 15:
		infos[camera_number - 1][0] = info[0:15]
	if len(info) > 15 and len(info) < 30:
		infos[camera_number - 1][1] = info[15:len(info)]
		infos[camera_number - 1][1] += "	"
		if len(info) < 22:
			infos[camera_number - 1][0] += "	"
	if len(info) >= 30:
		infos[camera_number - 1][1] = info[15:30]
	if len(info) > 30 and len(info) < 45:
		infos[camera_number - 1][2] = info[30:len(info)]
		infos[camera_number - 1][2] += "	"
		if len(info) < 37:
			infos[camera_number - 1][0] += "	"
	if len(info) >= 45:
		infos[camera_number - 1][2] = info[30:45]
	if len(info) > 45 and len(info) < 60:
		infos[camera_number - 1][3] = info[45:len(info)]
		infos[camera_number - 1][3] += "	"
		if len(info) < 52:
			infos[camera_number - 1][0] += "	"
	if len(info) >= 60:
		infos[camera_number - 1][3] = info[45:60]
	if len(info) > 60 and len(info) < 75:
		infos[camera_number - 1][4] = info[60:len(info)]
		infos[camera_number - 1][4] += "	"
		if len(info) < 67:
			infos[camera_number - 1][0] += "	"
	if len(info) >= 75:
		infos[camera_number - 1][4] = info[60:75]

	print "============================VIDEO SURVEILLANCE SYSTEM============================"
	print "|			%s		%s		%s		%s	|" %(arrow[0], arrow[1], arrow[2], arrow[3])#arrow
	print "|Camera		|	1	|	2	|	3	|	4	|"
	print "--------------------------------------------------------------------------------"
	print "|Event		|%s	|%s	|%s	|%s	|" %(event_dic[events[0]], event_dic[events[1]], event_dic[events[2]], event_dic[events[3]])
	print "--------------------------------------------------------------------------------"
	print "|Distance	|	%dcm	|	%dcm	|	%dcm	|	%dcm	|" %(ul_history[0][history_data_max - 1], ul_history[1][history_data_max - 1], ul_history[2][history_data_max - 1], ul_history[3][history_data_max - 1])
	print "--------------------------------------------------------------------------------"
	print "|Detection	|%s|%s|%s|%s|%s|%s|%s|%s|%s|%s|%s|%s|%s|%s|%s|%s|" %(pas_dic[passs[0]], app_dic[apps[0]], hum_dic[hums[0]], lea_dic[leas[0]], pas_dic[passs[1]], app_dic[apps[1]], hum_dic[hums[1]], lea_dic[leas[1]], pas_dic[passs[2]], app_dic[apps[2]], hum_dic[hums[2]], lea_dic[leas[2]], pas_dic[passs[3]], app_dic[apps[3]], hum_dic[hums[3]], lea_dic[leas[3]])
	print "--------------------------------------------------------------------------------"
	print "|Alarm		|	%r	|	%r	|	%r	|	%r	|" %(alarms[0], alarms[1], alarms[2], alarms[3])
	print "--------------------------------------------------------------------------------"
	print "|		|%s|%s|%s|%s|" %(infos[0][0], infos[1][0], infos[2][0], infos[3][0])
	print "|		|%s|%s|%s|%s|" %(infos[0][1], infos[1][1], infos[2][1], infos[3][1])
	print "|Info		|%s|%s|%s|%s|" %(infos[0][2], infos[1][2], infos[2][2], infos[3][2])
	print "|		|%s|%s|%s|%s|" %(infos[0][3], infos[1][3], infos[2][3], infos[3][3])
	print "|		|%s|%s|%s|%s|" %(infos[0][4], infos[1][4], infos[2][4], infos[3][4])
	print "================================================================================="


with picamera.PiCamera() as camera:

	def camHD():
		camera.resolution = hd_resolution
		camera.framerate = hd_frame_rate

	def camSD():
		camera.resolution = sd_resolution
		camera.framerate = sd_frame_rate

	def camRecord(defination, camera_number, record_time, filename):
		if defination == "HD":
			camHD()
		elif defination == "SD":
			camSD()
		else:
			pass
		camChange(camera_number)
		camera.start_recording(filename, format='h264')
		time.sleep(record_time)
		camera.stop_recording()

	def camCapture(defination, camera_number, pics_number):
		#start = time.time()
		if defination == "HD":
			camHD()
		elif defination == "SD":
			camSD()
		else:
			pass
		camChange(camera_number)
		camera.capture_sequence(imageFilenames(camera_number, pics_number), use_video_port=True)
		#finish = time.time()
		#print('Captured %d frames at total %.2ffps' % (pics_number, pics_number / (finish - start)))

	def createTemp(camera_number):
		camSD()
		camChange(camera_number)
		camera.capture_sequence(pics, use_video_port=True)

	def deleteTemp():
		for y in pics:
			os.remove(y)

	# main function
	print("Initializing camera..")
	camHD()
	camera.start_preview()
	print("Waiting for alarm client connection, Please run SocketClient.py on host PC")
	conn,addr = server.accept()
	print("Alarm server connected by:%s:%s" %(addr[0], addr[1]))

	while True:
		for i in range(1, camera_number + 1):
			alarm = False
			info = ""
			#if i != 3:
			#	continue
			loop_start = float(time.time() * 1000)
			ultrasonic_distance = (ultrasonicHistory(i, True))[history_data_max - 1]
			createTemp(i)
			approaching  = isApproach(i)
			leaving = isLeaving(i)
			human = isHuman(i)
			passing = isPassingby(i)
			event = events[i - 1]
			old_event = event

			if (event == 3 or event == 4) and human:
				event = 4
				info += ("Suspecious target appears, take video for %.1fs" %(video_time)) + " "
				#print("Sending alarm to host PC")
				alarm = True
				conn.send("Alarm from camera %d: Object is %dcm in the front" %(i, ultrasonic_distance))
				respon = conn.recv(BUF_SIZE)
				if respon:
					info += ("Host PC:%s" %respon) + " "
				grovepi.digitalWrite(buzzer,1)
				camRecord("HD", i, video_time, videoFilenames(i))
				grovepi.digitalWrite(buzzer,0)
			elif approaching:
				event = 3
				info += ("Object is approaching, take %d pictures" %(approach_pics)) + " "
				camCapture("HD", i, approach_pics)
			elif (event == 1 or event == 2 or event == 5) and passing:
				event = 2
				info += ("Object is passing by, take %d pictures" %(passing_pics)) + " "
				camCapture("HD", i, passing_pics)
			elif event != 1 and leaving:
				event = 5
				info += ("Object is leaving, take %d pictures" %(leaving_pics)) + " "
				camCapture("HD", i, leaving_pics)
			elif event != 4:
				event = 1
				info += ("Nothing is in front of camera") + " "
			else:
				#print("State machine error on camera %d" %i)
				#deleteTemp()
				#sys.exit()
				event = 1
				info += ("Nothing is in front of camera") + " "
			
			events[i - 1] = event
			deleteTemp()
			loop_end = float(time.time() * 1000)
			UI(i, event, ultrasonic_distance, passing, approaching, human, leaving, alarm, info)
			print "\n\nFor debugging:"
			print "ultrasonicHistory: ", ul_history
			print "events list: ", events
			print "\n\ncamera", i, ":"
			print "event change from event", old_event, " to event", event
			print("Loop execution time is %f ms" %(loop_end-loop_start))
			print global_debug
			global_debug = ""
			alarm = False
conn.close()
pi.stop()
