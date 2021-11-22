#----------------------------Dependencies-----------------------------------
from centroidtracker import CentroidTracker
import cv2
from detect import detect
from datetime import datetime
from matplotlib import pyplot as plt
import matplotlib
matplotlib.use('TkAgg')
#----------------------------Argument parsers----------------------------
# ap = argparse.ArgumentParser()
# ap.add_argument("--source", required=True)
# ap.add_argument("--skipframes")
# ap.add_argument("--classnameslist")
# ap.add_argument("--classIDlist")
# ap.add_argument("--conf")
# ap.add_argument("--parklimit")
# args = vars(ap.parse_args())

#-------------------------------Variables----------------------------------
#------------USER CHANGEBLE VARIABLES----------- 
video = "Videos/illegalparking1.mp4" #args['source']
detectclassname = ['car', 'bus', 'truck'] #args["classnameslist"]
detectclassid=[2,5,7] #args["classIDlist"]
illegal_park_time_limit = 20 #args["parklimit"]
skip_frames = 50 #args["skipframes"]
confidence_threshold=0.4 #args["conf"]

#---------------fixed supported variables----------
totalFrames=0
min_x = 0
min_y = 0
max_x = 0
max_y = 0
upd = 0
flag_plot = 0
virtual_bound = 20
cancel_key = "a"
warningsignimage=cv2.imread("dataset/no_parking1.png")


#-------------- fixed supported dictionaries-------
object_time = {}
updated_centroid_flag={}
updated_centroid = {}
parkingviolationcount = {}
parkingviolcnt_interval = {}
classID = {}
timeinterval_done = {}
count_done = {}
entry = {}
exit = {}
parktime={}
blink_circle = {}

#------------------fixed supported lists-------------
IDS=[]
timestamp=[]
trackers = []
park_temp = []
illegal_parking_entry_ID = []

#-------------------------------------------------------
start_time = datetime.now()
vs = cv2.VideoCapture(video)
ct = CentroidTracker(maxDisappeared=40, maxDistance=50)


#-------------------------helper functions----------------------------------------------------

def make_virtual_bound(objectID):
	global min_x,min_y,max_x,max_y
	min_x = updated_centroid[objectID][0] - virtual_bound
	min_y = updated_centroid[objectID][1] - virtual_bound
	max_x = updated_centroid[objectID][0] + virtual_bound
	max_y = updated_centroid[objectID][1] + virtual_bound

def blink_violation_image():
	global warningsignimage
	warningsignimage = cv2.resize(warningsignimage, (28, 28))
	warningsignimage_height,warningsignimage_width,_=warningsignimage.shape
	frame[centroid[1]-30:(centroid[1]-30+warningsignimage_height),centroid[0]-50:(centroid[0]-50 +warningsignimage_width)]=warningsignimage


def data_preparation_for_timestamp_visualization():
	updated_time=datetime.now()
	elapsed_time=int((updated_time-start_time).total_seconds())
	for eachclass in detectclassname:
			if parkingviolationcount.get(eachclass)!=None:
				park_temp.append(parkingviolationcount[eachclass])
			else:
				park_temp.append(0)
			if timeinterval_done.get(elapsed_time)==None:
				timeinterval_done[elapsed_time]=1
			parkingviolcnt_interval[str(updated_time.strftime("%I"))+":"+str(updated_time.strftime("%M")) + ":" + str(updated_time.strftime("%S"))] = park_temp
#-------------------ROI--------------------------------
ret,frame=vs.read()
frame = cv2.resize(frame, (1080, 640))
x1,y1,w,h=cv2.selectROI(frame)
x2=x1+w
y2=y1+h
#------------------------------Start of while loop--------------------------------------------------------

while True:
	# grab the next frame and handle if we are reading from either
	# VideoCapture or VideoStream
	ret, frame = vs.read()
	if ret==0:
		break
	frame = cv2.resize(frame, (1080, 640))
	frame=frame[y1:y2,x1:x2]
	if ret == 0:
		break
	rects = []
	rectclass = []
	if totalFrames % skip_frames == 0:
		trackers = cv2.MultiTracker_create()
		trackerclass = []
		success, detection, frame = detect(image_to_be_classified=frame, classes= detectclassid, conf_thres=confidence_threshold)
		if success == 1:
			number_of_detection = detection.shape[0]
			for i in range(number_of_detection - 1):
				startX = int(float(detection[i + 1][0]))
				startY = int(float(detection[i + 1][1]))
				endX = int(float(detection[i + 1][2]))-startX
				endY = int(float(detection[i + 1][3]))-startY
				box = (startX, startY, endX, endY)
				tracker = cv2.TrackerCSRT_create()
				trackers.add(tracker, frame, box)
				trackerclass.append(detection[i + 1][4])
	else:
		iteration = -1
		(success,boxes)=trackers.update(frame)
		for box in boxes:
			iteration += 1
			(x, y, w, h) = [int(v) for v in box]
			cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
			rects.append((x, y, (x + w), (y + h)))
			rectclass.append(trackerclass[iteration])
	objects, classes, classID = ct.update(rects,rectclass)

	for (objectID, centroid) in objects.items():


		if updated_centroid_flag.get(objectID)==None:
			updated_centroid[objectID] = centroid
			updated_centroid_flag[objectID]=1

		else:
			make_virtual_bound(objectID)
			# check if the vehicle is moving or not
			if centroid[0] >= min_x and centroid[0] <= max_x and centroid[1] >= min_y and centroid[1] <= max_y:
				# this means vehicle is illegally parked, so we start the timer
				if object_time.get(objectID, None) == None:
					ts1 = datetime.now()
					entry[objectID]=str(ts1.strftime("%I")) + ":" + str(ts1.strftime("%M")) + ":" + str(ts1.strftime("%S"))
					object_time[objectID] = ts1
				ts2 = datetime.now()
				exit[objectID] = str(ts2.strftime("%I")) + ":" + str(ts2.strftime("%M")) + ":" + str(ts2.strftime("%S"))
				parktime[objectID] = int((ts2 - object_time[objectID]).total_seconds())

				if parktime[objectID] >= illegal_park_time_limit:

					if count_done.get(objectID) == None:
						count_done[objectID] = 1
						if parkingviolationcount.get(classes[objectID])==None:
							parkingviolationcount[classes[objectID]]=1
						else:
							parkingviolationcount[classes[objectID]]+=1

					if blink_circle.get(objectID)==None:
						blink_circle[objectID]=1
					else:
						blink_circle[objectID]+=1
					if(blink_circle[objectID]%2==0):
						blink_violation_image()

					cv2.rectangle(frame,(centroid[0]-110,centroid[1]+15),(centroid[0]+256,centroid[1]+35),(116, 20, 0),-1)
					cv2.putText(frame, f"ID {objectID} illegally parked for {parktime[objectID]:.2f} seconds", (centroid[0]-100, centroid[1]+30),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255),1, cv2.LINE_AA)

			else:
				updated_centroid[objectID] = centroid
				object_time[objectID] = None


		text = "{}".format(objectID)
		cv2.rectangle(frame, (centroid[0] - 10, centroid[1] - 25), (centroid[0] + 50, centroid[1] - 3), (44, 0, 116), -1)
		cv2.putText(frame, text, (centroid[0] - 10, centroid[1] - 10), cv2.cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255),1, cv2.LINE_AA)

	data_preparation_for_timestamp_visualization()

	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1)
	if key == ord(cancel_key):
		break
	totalFrames += 1



#---------------------------------------Visualization setup-----------------------------------------------

def configuring_plot_details():
	plt.figure(facecolor='#1B2631')
	ax = plt.axes()
	ax.set_facecolor("#1B2631")
	ax.tick_params(axis='x', colors='#F2F4F4', rotation = 90)
	ax.tick_params(axis='y', colors='#F2F4F4')
	plt.title("Illegal parking detection",color='#E74C3C',fontweight="bold")
	plt.xlabel("Vehicle ID -->",color='#FDFEFE',fontweight="bold")
	plt.ylabel("Time-stamp (HR:MIN:SEC) -->",color='#FDFEFE',fontweight="bold")

def prepare_data_for_visualization():
	illegal_parking_entry_ID=list(entry.keys())
	valid_count=0
	for ID in illegal_parking_entry_ID:
		if parktime[ID]> illegal_park_time_limit:
			valid_count+=1
			IDS.append(ID)
			IDS.append(ID)
			timestamp.append(entry[ID])
			timestamp.append(exit[ID])
	return valid_count

def creating_data_visualization(valid_count):
	global upd, flag_plot
	for i in range(valid_count):
		particularID = IDS[upd:upd + 2]
		entry_exit = timestamp[upd:upd + 2]
		upd += 2
		if parktime.get(particularID[0]) > illegal_park_time_limit:
			plt.plot(particularID, entry_exit, color= "#F20F0F",linewidth=10)
		else:
			plt.plot(particularID, entry_exit, color= "#0FF256",linewidth=10)
		if flag_plot==0:
			plt.plot([particularID[0]], [entry_exit[0]], marker='v', markerfacecolor='black', markeredgecolor='black',label='Entry Time')
			plt.plot([particularID[1]], [entry_exit[1]], marker='^', markerfacecolor='black', markeredgecolor='black',label='Exit Time')
			flag_plot=1
		else:
			plt.plot([particularID[0]], [entry_exit[0]], marker='v', markerfacecolor='black', markeredgecolor='black')
			plt.plot([particularID[1]], [entry_exit[1]], marker='^', markerfacecolor='black', markeredgecolor='black')

		plt.text(particularID[0],entry_exit[0],f"Park time = {parktime[particularID[0]]} sec",color='#E74C3C',verticalalignment='center',fontweight="bold")

def ID_vs_violation():
	configuring_plot_details()
	valid_count = prepare_data_for_visualization()
	creating_data_visualization(valid_count)
	plt.legend()
	plt.show()

ID_vs_violation()

