#----------------------------Dependencies-----------------------------------
from centroidtracker import CentroidTracker
from matplotlib import pyplot as plt
import matplotlib
import numpy as np
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

#------------------------User customizable variables,list-----------------------------
detectclassname = ['person', 'car', 'bus', 'truck','motorcycle'] #args["classnameslist"]
detectclassID = [0, 2, 5, 7,3]#args["classIDlist"]
video = "Videos/Speed Violation.mp4" #args["source"]
skip_frames = 10
confidence_threshold=0.6 #args["conf"]

#------------------------Fixed variables,list, dict, etc-----------------------------
classID = {}
vs = cv2.VideoCapture(video)
ct = CentroidTracker(maxDisappeared=40, maxDistance=50)
totalFrames = 0
start_time = datetime.now()
count_in_intervals={}
timeinterval_done={}
#------------------------------ROI------------------------------------------
ret,frame=vs.read()
x1,y1,w,h=cv2.selectROI(frame)
x2=x1+w
y2=y1+h
#------------------------------Start of while loop----------------------------------------------

while True:
	image = np.zeros((512, 512, 3))
	ret, frame = vs.read()
	if ret == 0:
		break
	# frame = frame[306:720, 4:1101]
	frame=frame[y1:y2,x1:x2]
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
	elapsed_time=int((updated_time-start_time).total_seconds())

	if elapsed_time%60==0 and elapsed_time>0 and timeinterval_done.get(elapsed_time)==None and classID!=None:
		temp=[]
		for i in detectclassname:
			if classID.get(i,None)!=None:
				temp.append(int(classID[i].split(" ")[1]))
			else:
				temp.append(0)
		if timeinterval_done.get(elapsed_time)==None:
			timeinterval_done[elapsed_time]=1

		count_in_intervals[str(updated_time.strftime("%I"))+":"+str(updated_time.strftime("%M"))]=temp




	for (objectID, centroid) in objects.items():
		cv2.rectangle(frame, (centroid[0] - 10, centroid[1] - 25), (centroid[0] + 60, centroid[1] - 3), (0, 0, 255), -1)
		cv2.putText(frame, "{}".format(objectID), (centroid[0] - 10, centroid[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
					(0, 255, 255), 2, cv2.LINE_AA)
		ycoord = 0
		for i in range(len(detectclassname)):
			if classID.get(detectclassname[i], None) != None:
				countofclass = classID[detectclassname[i]].split(" ")[1]
				text = f"{detectclassname[i]} : {countofclass}"
				cv2.putText(image, text, (0, ycoord + 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
				ycoord += 30

	cv2.imshow("Frame", frame)
	cv2.imshow("Count", image)
	key = cv2.waitKey(1)
	if key == ord("a"):
		break
	totalFrames += 1
	countclass = [0] * len(detectclassID)
	index = 0
	for i in range(len(detectclassID)):

		if classID.get(detectclassname[i], None) != None:
			t = classID[detectclassname[i]]
			t = t.split(" ")[1]
			countclass[index] = int(t)
			index += 1
# -----------------------End of while loop-------------------------------
vs.release()
cv2.destroyAllWindows()

#------------------------------Visualizations---------------------------------

values=list(count_in_intervals.values())
keys=list(count_in_intervals.keys())

for i in reversed(range(len(values))):
	if i==0:
		continue
	for j in range(len(detectclassname)):
		values[i][j]-=values[i-1][j]
		count_in_intervals[keys[i]][j]=values[i][j]



def setting_color():
	ax = plt.axes()
	ax.set_facecolor("#1B2631")
	ax.tick_params(axis='x', colors='#F2F4F4')
	ax.tick_params(axis='y', colors='#F2F4F4')

color = ['#F4D03F', '#E74C3C', '#7D3C98', '#27AE60','#3DFF33']

def no_of_vehicles_bar():
	yaxis = detectclassname
	xaxis = countclass
	plt.figure(num=1,facecolor='#1B2631')
	setting_color()
	plt.barh(yaxis, xaxis, 0.3,color=color)
	plt.xlabel("Number of vehicles",color='#F4D03F',fontweight="bold")
	plt.ylabel("Vehicles",color='#F4D03F',fontweight="bold")
	plt.title("Number of vehicles by class",color='#F2F4F4',fontweight="bold")
	for i in range(len(xaxis)):
		plt.text(xaxis[i],i,f"{xaxis[i]}",verticalalignment='center',color='#FDFEFE',fontsize = 11,fontweight="bold")


def percentage_of_vehicles_pie():
	plt.figure(num=2,facecolor='#1B2631')
	setting_color()
	plt.title("Class-wise counting proportion",color='#F2F4F4')
	patches, texts, pcts=plt.pie(countclass, labels=detectclassname, colors=color, autopct='%.2f%%',wedgeprops={'linewidth': 3.0, 'edgecolor': '#1B2631'})
	for i, patch in enumerate(patches):
		texts[i].set_color(patch.get_facecolor())
	plt.setp(pcts, color='white')
	plt.setp(texts, fontweight=600)
	plt.legend()

def no_of_vehicles_bytime_line():
	plt.figure(num=3,facecolor='#1B2631')
	setting_color()
	plt.title("Number of types of vehicles by time",color='#F2F4F4')
	ids = list(count_in_intervals.keys())
	countbytime = list(count_in_intervals.values())
	# plt.plot(ids, countbytime)
	markerscolorlist = ['#F4D03F', '#229954', '#E74C3C', '#3498DB','#3DFF33']
	for i in range(len(ids)):
		for j in range(len(detectclassname)):
			plt.text(ids[i],countbytime[i][j]+float(0.5),f"{countbytime[i][j]}",verticalalignment='center',color=markerscolorlist[j],fontweight="bold")

	plt.xlabel("Time stamp",color='#F4D03F',fontweight="bold")
	plt.ylabel("Number of Vehicles at a particular time-stamp",color='#F4D03F',fontweight="bold")


	ax = plt.subplot()
	markershapelist=['o','v','^','*',cv2.MARKER_CROSS]

	for eachclass in range(len(detectclassname)):
		countofparticularclass=[]
		eachtimestamp=[]
		iteration=0

		for countofeachclass in values:
			countofparticularclass.append(countofeachclass[eachclass])
			eachtimestamp.append(keys[iteration])
			iteration+=1
		ax.plot(eachtimestamp,countofparticularclass,label=detectclassname[eachclass],marker=markershapelist[eachclass],markerfacecolor=markerscolorlist[eachclass],markeredgecolor=markerscolorlist[eachclass],color=markerscolorlist[eachclass],markersize=12)

	plt.legend()

no_of_vehicles_bar()
percentage_of_vehicles_pie()
no_of_vehicles_bytime_line()
plt.show()
#---------------------------------------END----------------------------------------


# ': [7, 8, 1, 0], '01:17': [7, 17, 3, 0], '01:18': [10, 22, 5, 0], '01:19': [10, 33, 5, 0], '01:20': [10, 38, 6, 0], '01:21': [12, 49, 6, 0], '01:22': [12, 62, 6, 0]}