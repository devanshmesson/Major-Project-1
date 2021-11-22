# import the necessary packages
from scipy.spatial import distance as dist
from collections import OrderedDict
import numpy as np

class CentroidTracker:
	def __init__(self, maxDisappeared=100, maxDistance=50):
		self.nextObjectID = 0
		self.objects = OrderedDict()
		self.objectsclass= OrderedDict()
		self.classID= {}
		self.disappeared = OrderedDict()
		self.maxDisappeared = maxDisappeared
		self.maxDistance = maxDistance

	def register(self, centroid,rectclass):

		if self.classID.get(rectclass,None)==None:
			self.classID[rectclass]=rectclass + " " +"0"

		self.objects[self.classID[rectclass]] = centroid
		self.objectsclass[self.classID[rectclass]]=rectclass
		self.disappeared[self.classID[rectclass]] = 0

		temp=self.classID[rectclass].split(" ")[1]
		temp=int(temp)+1
		self.classID[rectclass]=rectclass+ " " + str(temp)



	def deregister(self, objectID):
		del self.objects[objectID]
		del self.disappeared[objectID]

	def update(self, rects,rectclass):

		if len(rects) == 0:
			for objectID in list(self.disappeared.keys()):
				self.disappeared[objectID] += 1
				if self.disappeared[objectID] > self.maxDisappeared:
					self.deregister(objectID)
			return self.objects, self.objectsclass, self.classID
		inputCentroids = np.zeros((len(rects), 2), dtype="int")

		for (i, (startX, startY, endX, endY)) in enumerate(rects):
			# use the bounding box coordinates to derive the centroid
			cX = int((startX + endX) / 2.0)
			cY = int((startY + endY) / 2.0)
			inputCentroids[i] = (cX, cY)

		if len(self.objects) == 0:
			for i in range(0, len(inputCentroids)):
				self.register(inputCentroids[i],rectclass[i])

		else:
			objectIDs = list(self.objects.keys())
			objectCentroids = list(self.objects.values())
			D = dist.cdist(np.array(objectCentroids), inputCentroids)
			rows = D.min(axis=1).argsort()
			cols = D.argmin(axis=1)[rows]
			usedRows = set()
			usedCols = set()
			for (row, col) in zip(rows, cols):
				if row in usedRows or col in usedCols:
					continue
				if D[row, col] > self.maxDistance:
					continue
				objectID = objectIDs[row]
				self.objects[objectID] = inputCentroids[col]
				self.disappeared[objectID] = 0
				usedRows.add(row)
				usedCols.add(col)
			unusedRows = set(range(0, D.shape[0])).difference(usedRows)
			unusedCols = set(range(0, D.shape[1])).difference(usedCols)
			if D.shape[0] >= D.shape[1]:
				for row in unusedRows:
					objectID = objectIDs[row]
					self.disappeared[objectID] += 1
					if self.disappeared[objectID] > self.maxDisappeared:
						self.deregister(objectID)

			else:
				for col in unusedCols:
					self.register(inputCentroids[col],rectclass[col])
		return self.objects,self.objectsclass,self.classID