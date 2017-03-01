import grovepi  
import time
import picamera
import pigpio
import socket

#remote host name and port
HOST='172.20.45.26'
PORT=8081

ADDR=(HOST,PORT)
BUF_SIZE = 1024

ultrasonic_ranger = [2, 3, 7, 8]
buzzer = 1
grovepi.pinMode(buzzer,"OUTPUT")

pigpio.exceptions = False
pi = pigpio.pi()

pi.set_mode(4, pigpio.OUTPUT)
pi.set_mode(17, pigpio.OUTPUT)
pi.set_mode(18, pigpio.OUTPUT)  

pi.write(4, 0)
pi.write(17, 0)
pi.write(18, 1)

frames = 4
video_time = 2
img_cnt = 0
vid_cnt = 0
cam = 1

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
	#gp.setmode(gp.BOARD)
	if cam == 1:
		# CAM 1 for A Jumper Setting
		# print("camera number 1")
		pi.write(4, 0)
		pi.write(17, 0)
		pi.write(18, 1)

	elif cam == 2:
		# CAM 2 for A Jumper Setting
		# print("camera number 2")
		pi.write(4, 1)
		pi.write(17, 0)
		pi.write(18, 1)

	elif cam == 3:
		# CAM 3 for A Jumper Setting
		# print("camera number 3")
		pi.write(4, 0)
		pi.write(17, 1)
		pi.write(18, 0)

	elif cam == 4:
		# CAM 4 for A Jumper Setting
		# print("camera number 4")
		pi.write(4, 1)
		pi.write(17, 1)
		pi.write(18, 0)

	time.sleep(0.007)   # SD Card Bandwidth Correction Delay

def image_filenames(camera_n):
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
	return "record/video%02d_c%d.mp4" % (img_cnt, camera_n)

with picamera.PiCamera() as camera:
	print("Initializing camera..")
	camera.resolution = (1920, 1080)
	camera.framerate = 25
	camera.start_preview()

	print("Waiting for alarm server connection...")
	conn = socket.socket(socket.AF_INET,socket.SOCK_STREAM)
	conn.connect(ADDR)

	while True:
		j = 0
		for i in ultrasonic_ranger:
			j += 1
			if j > 4:
				j = 1
			try:
				ultrasonic_distance = grovepi.ultrasonicRead(i)
				print ("Ultrasonic on D%d: Distance is: %dcm" % (i, ultrasonic_distance))
			except TypeError:
				print ("Ultrasonic on D%d: Typr error" %i)
			except IOError:
				print ("Ultrasonic on D%d: Read error" %i)

			if (ultrasonic_distance < 150) and (ultrasonic_distance > 50):
				print("Object in front of camera %d is closer than 1.5m" %j);
				print("Taking %d pictures with camera %d" %(frames,j))
				start = time.time()
				camera.capture_sequence(image_filenames(j), use_video_port=False)
				finish = time.time()
				print('Captured %d frames at total %.2ffps' % (frames, frames / (finish - start))) 
			if ultrasonic_distance < 50:
				print("Object in front of camera %d is closer than 0.5m" %j)
				print("Taking video with camera %d for %.1fs" %(j, video_time))
				print("Sending alarm to host PC")

				conn.send("Alarm from camera %d: Object is %dcm in the front" %(j, ultrasonic_distance))
				respon = conn.recv(BUF_SIZE)
				if respon:
					print("Host PC:%s" %respon)
					
				grovepi.digitalWrite(buzzer,1)
				camera.start_recording(video_filenames(j), format='h264')
				time.sleep(video_time)
				camera.stop_recording()
				grovepi.digitalWrite(buzzer,0)

conn.close()
pi.stop()

