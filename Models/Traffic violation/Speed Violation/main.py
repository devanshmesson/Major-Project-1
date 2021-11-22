from centroidtracker import CentroidTracker
from trackableobject import TrackableObject
import numpy as np
import cv2
from detect import detect
from datetime import datetime
from matplotlib import pyplot as plt
import matplotlib
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
#----------------------------Argument parsers----------------------------
# ap = argparse.ArgumentParser()
# ap.add_argument("--speedlimit", required=True)
# ap.add_argument("--realdistance", required=True)
# ap.add_argument("--source", required=True)
# ap.add_argument("--skipframes")
# ap.add_argument("--classnameslist")
# ap.add_argument("--classIDlist")
# ap.add_argument("--conf")
# args = vars(ap.parse_args())

#------------------------User customizable variables,list-----------------------------
detectclassname = ['car', 'bus', 'truck',"motorcycle"] #args["classnameslist"]
speed_violation_threshold=20 #args["speedlimit"]
detectclassid=[2,5,7,3] #args["classIDlist"]
real_distance=100 #args["realdistance"]
video="Videos/Speed Violation.mp4" #args["source"]
skip_frames=10 #args["skipframes"]
confidence_threshold=0.62 #args["conf"]

#------------------------variables,list,dictionaries,objects-------------------------------------------------------------------------------------------------------------------
speedviolcnt_interval = {}
timeinterval_done = {}
speedviolationcount = {}
start_time = datetime.now()
speed_estimation_zone = {"A": None, "B": None, "C": None, "D": None}
image = np.zeros((512, 512, 3))
vs = cv2.VideoCapture(video)
ct = CentroidTracker(maxDisappeared=40, maxDistance=50)
trackers = []
trackableObjects = {}
totalFrames = 0
totalDown = 0
totalUp = 0
points = [("A", "B"), ("B", "C"), ("C", "D")]
classID = {}

#----------------------ROI-------------------------------
ret,frame=vs.read()
x1,y1,w,h=cv2.selectROI(frame)
x2=x1+w
y2=y1+h

#-------------------------Traversing the video one by one-------------------------
while True:
    ret, frame = vs.read()
    if ret == 0:
        break
    ts = datetime.now()
    newDate = ts.strftime("%m-%d-%y")
    frame = frame[y1:y2, x1:x2]
    H = frame.shape[0]
    update = 0
    key = 'A'
    #Define speed estimation zone
    for i in range(4):
        speed_estimation_zone[key] = int(update)
        temp = ord(key) + 1
        key = chr(temp)
        update += int((H / 4))

    meterPerPixel = real_distance / H
    rects = []
    rectclass = []
    if totalFrames % skip_frames == 0:
        trackers = cv2.MultiTracker_create()
        trackerclass = []
        success, detection, frame = detect(image_to_be_classified=frame, classes=detectclassid, conf_thres=confidence_threshold)
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

    objects, classes, classID= ct.update(rects, rectclass)

    for (objectID, centroid) in objects.items():

        to = trackableObjects.get(objectID, None)
        if to is None:
            to = TrackableObject(objectID, centroid)
        elif not to.estimated:
            to.centroids.append(centroid)
            if to.direction is None or to.direction == 0.0:
                y = [c[1] for c in to.centroids]
                direction = centroid[1] - np.mean(y)
                to.direction = direction
            if to.direction > 0:

                if to.timestamp["A"] == 0:
                    if centroid[1] > speed_estimation_zone["A"]:

                        to.timestamp["A"] = ts
                        to.position["A"] = centroid[1]
                elif to.timestamp["B"] == 0:
                    if centroid[1] > speed_estimation_zone["B"]:

                        to.timestamp["B"] = ts
                        to.position["B"] = centroid[1]
                elif to.timestamp["C"] == 0:
                    if centroid[1] > speed_estimation_zone["C"]:
                        to.timestamp["C"] = ts
                        to.position["C"] = centroid[1]
                elif to.timestamp["D"] == 0:
                    if centroid[1] > speed_estimation_zone["D"]:
                        to.timestamp["D"] = ts
                        to.position["D"] = centroid[1]
                        to.lastPoint = True


            elif to.direction < 0:
                if to.timestamp["D"] == 0:

                    if centroid[1] < speed_estimation_zone["D"]:
                        to.timestamp["D"] = ts
                        to.position["D"] = centroid[1]

                elif to.timestamp["C"] == 0:

                    if centroid[1] < speed_estimation_zone["C"]:
                        to.timestamp["C"] = ts
                        to.position["C"] = centroid[1]

                elif to.timestamp["B"] == 0:

                    if centroid[1] < speed_estimation_zone["B"]:
                        to.timestamp["B"] = ts
                        to.position["B"] = centroid[1]

                elif to.timestamp["A"] == 0:

                    if centroid[1] < speed_estimation_zone["A"]:
                        to.timestamp["A"] = ts
                        to.position["A"] = centroid[1]

            if to.lastPoint and not to.estimated:
                estimatedSpeeds = []

                for (i, j) in points:
                    d = to.position[j] - to.position[i]
                    distanceInPixels = abs(d)

                    if distanceInPixels == 0:
                        continue
                    t = to.timestamp[j] - to.timestamp[i]
                    timeInSeconds = abs(t.total_seconds())
                    timeInHours = timeInSeconds / (60 * 60)
                    distanceInMeters = distanceInPixels * meterPerPixel
                    distanceInKM = distanceInMeters / 1000
                    estimatedSpeeds.append(distanceInKM / timeInHours)
                to.calculate_speed(estimatedSpeeds)
                to.estimated = True
                show_speed = "Speed of car ID {objectID}: {speed:.2f} KMPH".format(objectID=objectID,speed=to.speedKMPH)
                image = np.zeros((512, 512, 3))
                cv2.putText(image, show_speed, (0, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 100, 50), 1, cv2.LINE_AA)
                if to.speedKMPH >= 20:
                    if speedviolationcount.get(classes[objectID]) == None:
                        speedviolationcount[classes[objectID]] = 1
                    else:
                        speedviolationcount[classes[objectID]] += 1
                    violate = f"Car ID{objectID} is overspeeding!"
                    cv2.putText(image, violate, (0, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 100, 50), 1, cv2.LINE_AA)

        trackableObjects[objectID] = to
        text = "ID {}".format(objectID)
        cv2.rectangle(frame, (centroid[0] - 10, centroid[1] - 25), (centroid[0] + 50, centroid[1] - 3), (0, 0, 255), -1)
        cv2.putText(frame, "{}".format(objectID), (centroid[0] - 10, centroid[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,(0, 255, 255), 2, cv2.LINE_AA)

    updated_time = datetime.now()
    elapsed_time = int((updated_time - start_time).total_seconds())


    if elapsed_time % 60 == 0 and elapsed_time > 0 and timeinterval_done.get(elapsed_time) == None and classID != None:
        speed_temp = []
        # List preparation
        for eachclass in detectclassname:
            if speedviolationcount.get(eachclass) != None:
                speed_temp.append(speedviolationcount[eachclass])
            else:
                speed_temp.append(0)
        if timeinterval_done.get(elapsed_time) == None:
            timeinterval_done[elapsed_time] = 1

        speedviolcnt_interval[str(updated_time.strftime("%I")) + ":" + str(updated_time.strftime("%M"))] = speed_temp

    cv2.imshow("Frame", frame)
    cv2.imshow("show speed", image)
    key = cv2.waitKey(1)
    if key == ord("a"):
        break
    totalFrames += 1

vs.release()
cv2.destroyAllWindows()
#-------------------------------------Data preparation and Visualizations---------------------

# data preparation
values = list(speedviolcnt_interval.values())
keys = list(speedviolcnt_interval.keys())
speed = [int(trackableObjects[eachclass.split(" ")[0] + " " + str(countofeachclass)].speedKMPH) for eachclass in
classID.values() for countofeachclass in range(int(eachclass.split(" ")[1])) if
         trackableObjects[eachclass.split(" ")[0] + " " + str(countofeachclass)].speedKMPH != None]
ID = [eachclass.split(" ")[0] + " " + str(countofeachclass) for eachclass in classID.values() for countofeachclass in
      range(int(eachclass.split(" ")[1])) if
      trackableObjects[eachclass.split(" ")[0] + " " + str(countofeachclass)].speedKMPH != None]

for i in reversed(range(len(values))):
    if i == 0:
        continue
    for j in range(len(detectclassname)):
        values[i][j] -= values[i - 1][j]
    speedviolcnt_interval[keys[i]] = values[i]


#Visualizations
def set_color():
    matplotlib.use('TkAgg')
    plt.figure(facecolor='#1B2631')
    ax = plt.axes()
    ax.set_facecolor("#1B2631")
    ax.tick_params(axis='x', colors='#F2F4F4')
    ax.tick_params(axis='y', colors='#F2F4F4')
    plt.title("Speed Limit Graph(Speed vs Vehicle IDs)", color='#F2F4F4', fontweight="bold", fontsize=13)


def no_of_speed_violations_bytimestamp_line():
    set_color()
    plt.title("Number of speed violations by types of vehicles on a timestamp", color='#F2F4F4')
    ids = list(speedviolcnt_interval.keys())
    countbytime = list(speedviolcnt_interval.values())
    markerscolorlist = ['#F4D03F', '#229954', '#E74C3C', '#3498DB']
    for i in range(len(ids)):
        for j in range(len(detectclassname)):
            plt.text(ids[i], countbytime[i][j] + float(0.05), f"{countbytime[i][j]}", verticalalignment='center',
                     color=markerscolorlist[j], fontweight="bold")

    plt.xlabel("Time stamp", color='#F4D03F', fontweight="bold")
    plt.ylabel("Number of Speed violations at a particular time-stamp", color='#F4D03F', fontweight="bold")

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
    plt.legend()



def line_graph():
    set_color()
    plt.figure(num=1, facecolor='#1B2631')
    plt.xticks(rotation=90, fontsize=9)
    plt.plot(ID, speed)
    a = 20
    plt.axhline(y=a, color='#FF3633', linestyle='--')
    plt.text(0, a + 0.2, f'Speed Limit({a} km/hr)', fontsize=8, color='#FDFEFE', fontweight="bold")
    f1 = 0
    f2 = 0


    # ---------------------------------------------------------
    image = 'dataset/warn.png'
    zoom = 0.07
    img = plt.imread(image)
    im = OffsetImage(img, zoom=zoom)
    # ---------------------------------------------------------
    for i in range(len(ID)):
        if int(trackableObjects[ID[i]].speedKMPH) > a:
            plt.gca().add_artist(AnnotationBbox(im, (ID[i], speed[i]), xycoords='data', frameon=False))
            if f1 == 0:
                plt.plot(ID[i], speed[i], marker="^", markerfacecolor='red', markeredgecolor='red',
                         label="Speed limit violated")
                f1 = 1

        else:
            plt.plot(ID[i], speed[i], marker="o", markerfacecolor='green', markeredgecolor='green', markersize=12)
            if f2 == 0:
                plt.plot(ID[i], speed[i], marker="o", markerfacecolor='green', markeredgecolor='green',
                         label="Under speed limit")
                f2 = 1

    plt.legend()
    for i in range(len(ID)):
        plt.text(ID[i], speed[i] + 0.7, f"{speed[i]}", verticalalignment='center', color='#F4D03F', fontsize=11,
                 fontweight="bold")

    plt.xlabel("Vehicles ID", color='#F4D03F', fontweight="bold")
    plt.ylabel("Speed of the Vehicle", color='#F4D03F', fontweight="bold")

    scale_factor = 1
    xmin, xmax = plt.xlim()
    ymin, ymax = plt.ylim()
    plt.xlim(xmin * scale_factor, xmax * scale_factor)
    plt.ylim(ymin * scale_factor, ymax * scale_factor)


# ------------------------------------------------------------------


line_graph()
no_of_speed_violations_bytimestamp_line()

plt.show()