# ----------------------------Dependencies-----------------------------------
from centroidtracker import CentroidTracker
from matplotlib import pyplot as plt
import cv2
from detect import detect
from datetime import datetime
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import matplotlib
import pandas as pd
from EasyROI import EasyROI
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
detectclassname = ['car', 'motorcycle', 'bus', 'truck'] #args["classnameslist"]
detectclassID = [2, 3, 5, 7] #args["classIDlist"]
video = "Videos/redlight crossing 3.mov" #args["source"]
skip_frames = 20 #args["skipframes"]
confidence_threshold=0.6 #args["conf"]

# ----------------------Fixed variables,list,dictionaries,objects-------------------------------------------------------------------------------------------------------------------
classID = {}
timeinterval_done = {}
vs = cv2.VideoCapture(video)
ct = CentroidTracker(maxDisappeared=40, maxDistance=50)
totalFrames = 0
start_time = datetime.now()
count_in_intervals = {}
timeinterval_done = {}
redlightviolationcount = {}
previously_below_line = {}
above_line = {}
blink_circle = {}
redlightviolcnt_interval = {}
count_done = {}
# -----------------------------Select ROI, offsetline, trafficlight, red_portion_of_traffic_light----------------------------------------------
ret, frame = vs.read()
#select_region_of_interest
x1, y1, w, h = cv2.selectROI(frame)
x2 = x1 + w
y2 = y1 + h
frame_roi_list = [y1, y2, x1, x2]
frame = frame[frame_roi_list[0]:frame_roi_list[1], frame_roi_list[2]:frame_roi_list[3]]
img = frame[y1:y2, x1:x2]

#select_offset_line
roi_helper = EasyROI(verbose=True)
line_roi = roi_helper.draw_line(frame, 1)
frame_temp = roi_helper.visualize_roi(frame, line_roi)
line_coordinates=[line_roi['roi'][0]['point1'],line_roi['roi'][0]['point2']]
cv2.line(frame,line_coordinates[0],line_coordinates[1],(188,255,0),3)

#select_traffic_light
x1, y1, w, h = cv2.selectROI(frame)
x2 = x1 + w
y2 = y1 + h
traffic_red_light_list = [x1, y1, x2, y2]
cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

#select_traffic_light_red_portion
x1, y1, w, h = cv2.selectROI(frame)
x2 = x1 + w
y2 = y1 + h
traffic_red_light_roi_list = [x1, y1, x2, y2]
cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

def blink_warning_sign(frame,centroid):
	warningsignimage = cv2.imread('Videos/traffic-signals.jpg')
	warningsignimage = cv2.resize(warningsignimage, (28, 28))
	warningsignimage_height, warningsignimage_width, _ = warningsignimage.shape
	frame[centroid[1]:(centroid[1] + warningsignimage_height),centroid[0]:(centroid[0] + warningsignimage_width)] = warningsignimage

# --------------------------------------Helper functions-----------------------------------
def check_position_of_vehicle(centroid):
	if centroid[1] > line_coordinates[1][1] and previously_below_line.get(objectID) == None:
		previously_below_line[objectID] = 1

	if previously_below_line.get(objectID) == 1 and centroid[1] < line_coordinates[1][1]:
		above_line[objectID] = 1

def count_traffic_light_violations(objectID):
	if redlightviolationcount.get(classes[objectID]) == None:
		redlightviolationcount[classes[objectID]] = 1
	else:
		redlightviolationcount[classes[objectID]] += 1

def data_preparation_for_timestamp_visualization(elapsed_time):
	if elapsed_time % 5 == 0 and elapsed_time > 0 and timeinterval_done.get(elapsed_time) == None and classID != None:
		redlight_temp = []

		for eachclass in detectclassname:
			if redlightviolationcount.get(eachclass) != None:
				redlight_temp.append(redlightviolationcount[eachclass])
			else:
				redlight_temp.append(0)
		if timeinterval_done.get(elapsed_time) == None:
			timeinterval_done[elapsed_time] = 1

		redlightviolcnt_interval[str(updated_time.strftime("%I")) + ":" + str(updated_time.strftime("%M")) + ":" + str(
			updated_time.strftime("%S"))] = redlight_temp

def identify_red_colour(frame, x1, y1, x2, y2):
	df = pd.read_csv('color_dataset_custom.csv')
	color_count = {'red': 0, 'green': 0, 'yellow': 0}
	whole_count = 0
	# Traversing each pixel and identifying it's color
	final_color = None
	skip_rows = 0
	rowscount = 0
	for row in range(y1, y2 + 1):
		skip_rows += 1
		if (skip_rows % 8 != 0):
			continue
		rowscount += 1
		for column in range(x1, x2 + 1):
			pixel = frame[row:row + 1, column:column + 1]
			B = pixel[0][0][0]
			G = pixel[0][0][1]
			R = pixel[0][0][2]
			minimum_diff = 1000
			dfcount = 0
			for color in range(len(df)):
				whole_count += 1
				dfcount += 1
				diff = abs(df["Red"][color] - R) + abs(df["Blue"][color] - B) + abs(df["Green"][color] - G)
				if diff < minimum_diff:
					minimum_diff = diff
					final_color = df["Name"][color]
			color_count[final_color] += 1
	if color_count["red"] > color_count["green"] and color_count["red"] > color_count["yellow"]:
		return 1
	return 0

# ---------------------------------------Traverse the video frames--------------------------------------------------
while True:
	ret, frame = vs.read()
	if ret == 0:
		break
	frame = frame[frame_roi_list[0]:frame_roi_list[1], frame_roi_list[2]:frame_roi_list[3]]
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
	updated_time = datetime.now()
	elapsed_time = int((updated_time - start_time).total_seconds())
	#Identify colour of traffic light's red portion(weather it's black or red)
	redlight = identify_red_colour(frame, traffic_red_light_roi_list[0], traffic_red_light_roi_list[1],traffic_red_light_roi_list[2], traffic_red_light_roi_list[3])

	for (objectID, centroid) in objects.items():
		cv2.rectangle(frame, (centroid[0] - 10, centroid[1] - 25), (centroid[0] + 50, centroid[1] - 3), (0, 0, 255), -1)
		cv2.putText(frame, "{}".format(objectID), (centroid[0] - 10, centroid[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,(0, 255, 255), 2, cv2.LINE_AA)
		check_position_of_vehicle(centroid)
		if above_line.get(objectID) == 1:
			if count_done.get(objectID) == None:
				count_done[objectID] = 1
				count_traffic_light_violations(objectID)
			if blink_circle.get(objectID)==None:
				blink_circle[objectID]=0
			else:
				blink_circle[objectID]+=1
			if (blink_circle[objectID] % 2 == 0) and redlight == 1:
				blink_warning_sign(frame,centroid)
	updated_time = datetime.now()
	elapsed_time = int((updated_time - start_time).total_seconds())
	data_preparation_for_timestamp_visualization(elapsed_time)
	cv2.line(frame, line_coordinates[0], line_coordinates[1], (188, 255, 0), 3)
	cv2.rectangle(frame, (traffic_red_light_list[0], traffic_red_light_list[1]),(traffic_red_light_list[2], traffic_red_light_list[3]), (0, 255, 0), 2)
	cv2.rectangle(frame, (traffic_red_light_roi_list[0], traffic_red_light_roi_list[1]),(traffic_red_light_roi_list[2], traffic_red_light_roi_list[3]), (0, 255, 0), 2)
	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1)
	if key == ord("a"):
		break
	totalFrames += 1
vs.release()
cv2.destroyAllWindows()
# ----------------------------------------Structuring data and visualizing data---------------------------------

#data_preparation:
values = list(redlightviolcnt_interval.values())
keys = list(redlightviolcnt_interval.keys())
for i in reversed(range(len(values))):
	if i == 0:
		continue
	for j in range(len(detectclassname)):
		values[i][j] -= values[i - 1][j]
	redlightviolcnt_interval[keys[i]] = values[i]

def take_image_in_graph():
	image = "Videos/traffic-signals.jpg"
	zoom = 0.07
	img = plt.imread(image)
	im = OffsetImage(img, zoom=zoom)
	return im

def setting_color():
	ax = plt.axes()
	ax.set_facecolor("#1B2631")
	ax.tick_params(axis='x', colors='#F2F4F4')
	ax.tick_params(axis='y', colors='#F2F4F4')

color = ['#F4D03F', '#E74C3C', '#7D3C98', '#27AE60']

def no_of_traffic_light_violations_bytimestamp_line():
	#Define the plot details
	plt.figure(num=2, facecolor='#1B2631')
	setting_color()
	plt.xticks(rotation=90, fontsize=9)
	plt.title("Number of redlight violations by types of vehicles on a timestamp", color='#F2F4F4')
	plt.xlabel("Time stamp", color='#F4D03F', fontweight="bold")
	plt.ylabel("Number of redlight violations at a particular time-stamp", color='#F4D03F', fontweight="bold")

	#Showing the count
	ids = list(redlightviolcnt_interval.keys())
	countbytime = list(redlightviolcnt_interval.values())
	markerscolorlist = ['#F4D03F', '#229954', '#E74C3C', '#3498DB']
	for i in range(len(ids)):
		for j in range(len(detectclassname)):
			plt.text(ids[i], countbytime[i][j] + float(0.05), f"{countbytime[i][j]}", verticalalignment='center',
					 color=markerscolorlist[j], fontweight="bold")

	#Create data according to line plot and finally plot the data!
	ax = plt.subplot()
	markershapelist = ['o', 'v', '^', '*']
	for eachclass in range(len(detectclassname)):
		countofparticularclass = []
		eachtimestamp = []
		iteration = 0

		for countofeachclass in values:
			countofparticularclass.append(countofeachclass[eachclass])
			eachtimestamp.append(keys[iteration])
			iteration += 1
		ax.plot(eachtimestamp, countofparticularclass, label=detectclassname[eachclass],
				marker=markershapelist[eachclass], markerfacecolor=markerscolorlist[eachclass],
				markeredgecolor=markerscolorlist[eachclass], color=markerscolorlist[eachclass], markersize=12)
	#Defining the legend
	plt.legend()


def ID_vs_violation():
	# Data preparation
	ID = []
	traffic_light_violation = []
	for everyclass in detectclassname:
		if classID.get(everyclass) == None:
			continue
		highestID = classID[everyclass]
		highestID_number = int(highestID.split(" ")[1])
		for idsofeverclass in range(highestID_number):
			classid = str(everyclass) + " " + str(idsofeverclass)
			if above_line.get(classid) == 1:
				traffic_light_violation.append("Traffic red-light violated")
				ID.append(classid)
			elif (previously_below_line.get(classid) == 1):
				traffic_light_violation.append("No Traffic red-light violated")
				ID.append(classid)

	#Define the plot details
	plt.figure(facecolor='#1B2631')
	setting_color()
	plt.xticks(rotation=90, fontsize=9)
	plt.plot(ID, traffic_light_violation)
	plt.title("Class-wise Traffic red-light violation detection", color='#F2F4F4', fontweight="bold", fontsize=13)
	plt.xlabel("Vehicles ID", color='#F4D03F', fontweight="bold")
	plt.ylabel("Traffic red-light violated", color='#F4D03F', fontweight="bold")
	f1 = 0
	f2 = 0
	#Over-write the image on the markers

	#for defining marker labels, over-write the individual points on the same plot and add labels of the marker
	for i in range(len(ID)):
		if traffic_light_violation[i] == "Traffic red-light violated":
			plt.gca().add_artist(AnnotationBbox(take_image_in_graph(), (ID[i], traffic_light_violation[i]), xycoords='data', frameon=False))
			# This is done to add the label only once for a particular marker
			if f1 == 0:
				plt.plot([ID[i]], [traffic_light_violation[i]], marker="^", markerfacecolor='red', markeredgecolor='red',label="Traffic red-light violated")
				f1 = 1
		else:
			plt.plot([ID[i]], [traffic_light_violation[i]], marker="*", markerfacecolor='#F39C12',markeredgecolor='#F39C12', markersize=14)
			# This is done to add the label only once for a particular marker
			if f2 == 0:
				plt.plot([ID[i]], [traffic_light_violation[i]], marker="*", markerfacecolor='#F39C12',markeredgecolor='#F39C12',label="No Traffic red-light violated")
				f2 = 1
	#Define the legend
	plt.legend()

	scale_factor = 1
	xmin, xmax = plt.xlim()
	ymin, ymax = plt.ylim()
	plt.xlim(xmin * scale_factor, xmax * scale_factor)
	plt.ylim(ymin * scale_factor, ymax * scale_factor)



no_of_traffic_light_violations_bytimestamp_line()
ID_vs_violation()
plt.show()
# ---------------------------------------END----------------------------------------



