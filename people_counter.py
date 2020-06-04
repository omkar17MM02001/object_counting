# import the packages
from objsearch.EssentialTrack import EssentialTrack
from objsearch.ObjectsTrackable import ObjectsTrackable
from imutils.video import VideoStream
from imutils.video import FPS
import numpy as np
import argparse
import imutils
import time
import dlib
import cv2

# Arguments 
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--prototxt", required=True,
	help="path to Caffe 'deploy' prototxt file")
ap.add_argument("-m", "--model", required=True,
	help="path to Caffe pre-trained model")
ap.add_argument("-i", "--input", type=str,
	help="path to optional input video file")
ap.add_argument("-o", "--output", type=str,
	help="path to optional output video file")
ap.add_argument("-c", "--confidence", type=float, default=0.4,
	help="minimum probability to filter weak detections")
ap.add_argument("-s", "--skip-frames", type=int, default=30,
	help="# of skip frames between detections")
args = vars(ap.parse_args())

# object which would be detectable 
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
	"bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
	"dog", "horse", "motorbike", "person", "pottedplant", "sheep",
	"sofa", "train", "tvmonitor"]

print("Loading model...")
net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])

if not args.get("input", False):
	print("Starting video stream...")
	vs = VideoStream(src=0).start()
	time.sleep(2.0)

else:
	print("Opening video file...")
	vs = cv2.VideoCapture(args["input"])

writer = None

W = None
H = None

ct = EssentialTrack(maxDisappeared=40, maxDistance=50)
trackers = []
trackableObjects = {}

totalFrames = 0
totalout = 0
totalin = 0

fps = FPS().start()

while True:
	frame = vs.read()
	frame = frame[1] if args.get("input", False) else frame

	if args["input"] is not None and frame is None:
		break

	frame = imutils.resize(frame, width=500)
	rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

	if W is None or H is None:
		(H, W) = frame.shape[:2]

	if args["output"] is not None and writer is None:
		fourcc = cv2.VideoWriter_fourcc(*"MJPG")
		writer = cv2.VideoWriter(args["output"], fourcc, 30,
			(W, H), True)

	status = "Waiting"
	rects = []

	if totalFrames % args["skip_frames"] == 0:
		status = "Detecting"
		trackers = []

		blob = cv2.dnn.blobFromImage(frame, 0.007843, (W, H), 127.5)
		net.setInput(blob)
		detections = net.forward()

		for i in np.arange(0, detections.shape[2]):
			confidence = detections[0, 0, i, 2]

			if confidence > args["confidence"]:
				idx = int(detections[0, 0, i, 1])

				if CLASSES[idx] != "person":
					continue

				box = detections[0, 0, i, 3:7] * np.array([W, H, W, H])
				(startX, startY, endX, endY) = box.astype("int")

				tracker = dlib.correlation_tracker()
				rect = dlib.rectangle(startX, startY, endX, endY)
				tracker.start_track(rgb, rect)

				trackers.append(tracker)

	else:
		for tracker in trackers:
			status = "Tracking"
			tracker.update(rgb)
			pos = tracker.get_position()

			startX = float(pos.left())
			startY = float(pos.top())
			endX = float(pos.right())
			endY = float(pos.bottom())

			rects.append((startX, startY, endX, endY))

	cv2.line(frame, (0, (H //4)*3), (W, (H // 4)*3), (0, 0, 255), 10)

	objects = ct.update(rects)

	for (objectID, centroid) in objects.items():
		to = trackableObjects.get(objectID, None)

		if to is None:
			to = ObjectsTrackable(objectID, centroid)

		else:
			y = [c[1] for c in to.centroids]
			direction = centroid[1] - np.mean(y)
			to.centroids.append(centroid)

			if not to.counted:
				if direction < 0 and centroid[1] < H // 2:
					totalin += 1
					to.counted = True

				elif direction > 0 and centroid[1] > H // 2:
					totalout += 1
					to.counted = True

		trackableObjects[objectID] = to

		text = "ID {}".format(objectID)
		cv2.putText(frame, text, (centroid[0] - 10, centroid[1] - 10),
			cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
		cv2.circle(frame, (centroid[0], centroid[1]), 4, (255, 255, 0), -1)
	info = [
		("IN", totalin),
		("OUT", totalout),
		("Current Status", status),
	]

	for (i, (k, v)) in enumerate(info):
		text = "{}---> {}".format(k, v)
		cv2.putText(frame, text, (60, H - ((i * 20) + 20)),
			cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

	if writer is not None:
		writer.write(frame)

	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF

	if key == ord("q"):
		break

	totalFrames += 1
	fps.update()

fps.stop()
print("Elapsed time: {:.2f}".format(fps.elapsed()))
print("Approx. FPS: {:.2f}".format(fps.fps()))

if writer is not None:
	writer.release()

if not args.get("input", False):
	vs.stop()

else:
	vs.release()

cv2.destroyAllWindows()