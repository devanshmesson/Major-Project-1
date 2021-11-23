import cv2
import numpy as np
from pyzbar.pyzbar import decode

vs  =  cv2.VideoCapture(0)
#----------------traverse through data.txt------------------ 
with open("data.txt") as traverse: 
	qrlist = traverse.read().splitlines()
#-----------------------------------------------------------
while True:
	ret, frame = vs.read()
	for barcode in decode(frame):
		qr_read = barcode.data.decode('utf-8')
		print(qr_read)
		#--------------------------check QR code read data exist in data.txt------------------------------
		pts = np.array([barcode.polygon], np.int32)
		pts = pts.reshape((-1, 1, 2))
		pts2 = barcode.rect
		if qr_read in qrlist:
			print("authorized")
			output = "authorized"
			myColor = (0, 255, 0)
			cv2.polylines(frame, [pts], True, myColor, 5)
			cv2.putText(frame, output, (pts2[0], pts2[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.9, myColor, 2)
			cv2.putText(frame, qr_read, (pts2[0], pts2[1] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.9, myColor, 2)
		else:
			print("unauthorized")
			output = "unauthorized"
			myColor = (0, 0, 255)
			cv2.polylines(frame, [pts], True, myColor, 5)
			cv2.putText(frame, output, (pts2[0], pts2[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.9, myColor, 2)
			cv2.putText(frame, qr_read, (pts2[0], pts2[1] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.9, myColor, 2)
		#---------------------------------------------------------------------------------------------------
	cv2.imshow("dhd", frame)
	key = cv2.waitKey(1)
	if key == ord("a"):
		break


