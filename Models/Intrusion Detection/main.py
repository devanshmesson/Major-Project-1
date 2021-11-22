#----------------------------Dependencies-----------------------------------

from centroidtracker import CentroidTracker
from trackableobject import TrackableObject
from matplotlib import pyplot as plt
import matplotlib
import cv2
from detect import detect
from datetime import datetime
matplotlib.use('TkAgg')

#----------------------------Argument parsers----------------------------
# ap = argparse.ArgumentParser()
# ap.add_argument("--source", required=True)
# ap.add_argument("--skipframes")
# ap.add_argument("--classnameslist")
# ap.add_argument("--classIDlist")
# ap.add_argument("--conf")
# args = vars(ap.parse_args())

#-------------------------------Variables----------------------------------

#------------USER CHANGEBLE VARIABLES----------- 
detectclassname = ['person'] #args["classnameslist"]
detectclassID = [0] #args["classIDlist"]
video = "Videos/Intrusion detection 8_Trim.mp4" #args["source"]
skip_frames =2 #args["skipframes"]
confidence_threshold=0.4 #args["conf"]

#---------------fixed supported variables----------------------
color = ['#F4D03F', '#E74C3C', '#7D3C98', '#27AE60']
totalFrames = 0
cancel_key = "a"
flag = 0
upd=0
index=-1
flag_plot=0
firstframe=None
frame = None
x1 = None 
y1=None
y2=None
x2=None
start_time = datetime.now()
vs = cv2.VideoCapture(video)
ct = CentroidTracker(maxDisappeared=40, maxDistance=50)

#-------------- fixed supported dictionaries----------------------
wait = {}
trackableObjects = {}
timeinterval_done={}
classID = {}
entry={}
exit={}
dwelltime={}
intruder_starttime={} 

#------------------fixed supported lists------------------------------

IDS=[]
timestamp=[]

#------------------------------Helper Function----------------------------------------------

def objectID_dwell_time_calculated(objects):
	for (objectID, centroid) in objects.items():
		if entry.get(objectID) == None:
			entry[objectID] = str(updated_time.strftime("%I")) + ":" + str(updated_time.strftime("%M")) + ":" + str(updated_time.strftime("%S"))
			intruder_starttime[objectID]=updated_time
		else:
			exit[objectID] = str(updated_time.strftime("%I")) + ":" + str(updated_time.strftime("%M")) + ":" + str(updated_time.strftime("%S"))
			dwelltime[objectID] = int((updated_time - intruder_starttime[objectID]).total_seconds())
		to = trackableObjects.get(objectID, None)
		if to is None:
			to = TrackableObject(objectID, centroid, classes[objectID])
		to.counted = True
		to.centroids.append(centroid)
		trackableObjects[objectID] = to
		text = "{}".format(objectID)
		cv2.rectangle(frame, (centroid[0] - 10, centroid[1] - 25), (centroid[0] + 60, centroid[1] - 3), (0, 0, 255), -1)
		cv2.putText(frame, text, (centroid[0] - 10, centroid[1] - 10), cv2.cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255),2, cv2.LINE_AA)


#-------------------------------------ROI SELECT--------------------
ret, frame = vs.read()
x1, y1, w, h = cv2.selectROI(frame)
x2=x1+w
y2=y1+h


#------------------------------Start of while loop--------------------------------------------------------

while True:
	ret, frame = vs.read()
	if ret == 0:
		break
	frame=frame[y1:y2,x1:x2]

	if flag==0:
		firstframe=frame
		flag=1

	rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
	rects = []
	rectclass = []
	if totalFrames % skip_frames == 0:
		trackers = cv2.MultiTracker_create()
		trackerclass = []
		success, detection, frame = detect(image_to_be_classified=frame, classes=detectclassID, conf_thres=confidence_threshold)
		if success == 1:
			number_of_detection = detection.shape[0]
			for i in range(number_of_detection - 1):
				startX = int(float(detection[i + 1][0]))
				startY = int(float(detection[i + 1][1]))
				endX = int(float(detection[i + 1][2])) - startX
				endY = int(float(detection[i + 1][3])) - startY
				box = (startX, startY, endX, endY)
				tracker = cv2.TrackerCSRT_create()
				trackers.add(tracker, frame, box)
				trackerclass.append(detection[i + 1][4])
	else:
		iteration = -1
		(success, boxes) = trackers.update(frame)
		for box in boxes:
			iteration += 1
			(x, y, w, h) = [int(v) for v in box]
			cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
			rects.append((x, y, (x + w), (y + h)))
			rectclass.append(trackerclass[iteration])

	objects, classes, classID = ct.update(rects, rectclass)
	updated_time=datetime.now()
	elapsed_time=int((updated_time-start_time).total_seconds())+1

	objectID_dwell_time_calculated(objects)	
	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1)
	if key == ord(cancel_key):
		break
	totalFrames += 1
vs.release()
cv2.destroyAllWindows()

#------------------------------Visualization Setup---------------------------------

def prepare_data_for_visualization():
	intruder_entry_ID=list(entry.keys())
	for ID in intruder_entry_ID:
		IDS.append(ID)
		IDS.append(ID)
		timestamp.append(entry[ID])
		timestamp.append(exit[ID])

def print_intruder_path():
	flag=0
	for intruder in entry.keys():  #for all intruders
		centroidlist=trackableObjects[intruder].centroids #for each Intruder's centroids list
		for intruder_centroid in range(len(centroidlist)-1):
			cv2.line(firstframe, tuple(centroidlist[intruder_centroid]), tuple(centroidlist[intruder_centroid+1]), (60,76,231), thickness=3)
			if flag==0:
				cv2.putText(firstframe, f"Intruder({intruder}) Enters", (centroidlist[intruder_centroid][0],centroidlist[intruder_centroid][1]+1), cv2.FONT_HERSHEY_COMPLEX, 0.5, (3,255,149), 2)
				flag=1
		cv2.putText(firstframe, f"Intruder({intruder}) Exits",(centroidlist[len(centroidlist)-1][0], centroidlist[len(centroidlist)-1][1] + 1),cv2.FONT_HERSHEY_COMPLEX, 0.5, (3,255,149), 2)
	cv2.imshow("window", firstframe)

def setting_color():
	ax = plt.axes()
	ax.set_facecolor("#1B2631")
	ax.tick_params(axis='x', colors='#F2F4F4')
	ax.tick_params(axis='y', colors='#F2F4F4')
	plt.title("Every intruder's entry and exit time",color='#E74C3C',fontweight="bold")
	plt.xlabel("Intruder's ID -->",color='#FDFEFE',fontweight="bold")
	plt.ylabel("Time-stamp (HR:MIN:SEC) -->",color='#FDFEFE',fontweight="bold")

def intrusion_duration():
	global upd, index
	plt.figure(facecolor='#1B2631')
	setting_color()
	upd=0
	index=-1
	flag_plot=0
	for i in range(len(entry.keys())):
		particularID = IDS[upd:upd + 2]
		entry_exit = timestamp[upd:upd + 2]
		upd += 2
		index+=1
		plt.plot(particularID, entry_exit, color=color[index%5],linewidth=10)
		if flag_plot==0:
			plt.plot([particularID[0]], [entry_exit[0]], marker='v', markerfacecolor='black', markeredgecolor='black',label='Entry Time')
			plt.plot([particularID[1]], [entry_exit[1]], marker='^', markerfacecolor='black', markeredgecolor='black',label='Exit Time')
			flag_plot=1
		else:
			plt.plot([particularID[0]], [entry_exit[0]], marker='v', markerfacecolor='black', markeredgecolor='black')
			plt.plot([particularID[1]], [entry_exit[1]], marker='^', markerfacecolor='black', markeredgecolor='black')

		plt.text(particularID[0],entry_exit[0],f"Duration = {dwelltime[particularID[0]]} sec",color='#E74C3C',verticalalignment='center',fontweight="bold")
	plt.legend()
	plt.show()

print_intruder_path()
prepare_data_for_visualization()
intrusion_duration()
#---------------------------------------END----------------------------------------