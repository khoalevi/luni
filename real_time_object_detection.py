from imutils.video import FileVideoStream
from imutils.video import FPS
import numpy as np
import argparse
import imutils
import time
import cv2

ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", type=str,
                help="path to input video")
ap.add_argument("-p", "--prototxt", type=str,
                default='./MobileNetSSD/MobileNetSSD_deploy.prototxt.txt',
                help="path to Caffe 'deploy' prototxt file")
ap.add_argument("-m", "--model", type=str,
                default='./MobileNetSSD/MobileNetSSD_deploy.caffemodel',
                help="path to Caffe pre-trained model")
ap.add_argument("-c", "--confidence", type=float, default=0.4,
                help="minimum probability to filter weak detections")
args = vars(ap.parse_args())

CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
           "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
           "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
           "sofa", "train", "tvmonitor"]
COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))

print("[INFO] loading model...")
net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])
print("[INFO] starting video stream...")
vs = FileVideoStream(args["video"]).start()
time.sleep(2.0)
fps = FPS().start()

frames = []

while True:
    try:
        frame = vs.read()
        frameClone = frame.copy()
        frame = imutils.resize(frame, width=400)
        ar = frameClone.shape[0] / frame.shape[0]

        (h, w) = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)),
                                    0.007843, (300, 300), 127.5)

        net.setInput(blob)
        detections = net.forward()

        for i in np.arange(0, detections.shape[2]):
            confidence = detections[0, 0, i, 2]

            if confidence > args["confidence"]:
                idx = int(detections[0, 0, i, 1])
                box = detections[0, 0, i, 3:7] * \
                    np.array([w * ar, h * ar, w * ar, h * ar])
                (startX, startY, endX, endY) = box.astype("int")

                tag = CLASSES[idx]
                if tag == "motorbike" or tag == "person":

                    label = "{}: {:.2f}%".format(CLASSES[idx],
                                                confidence * 100)
                    cv2.rectangle(frameClone, (startX, startY), (endX, endY),
                                COLORS[idx], 2)
                    y = startY - 15 if startY - 15 > 15 else startY + 15
                    cv2.putText(frameClone, label, (startX, y),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2)

        cv2.imshow("Frame", frameClone)
        key = cv2.waitKey(1) & 0xFF

        frames.append(frameClone)

        if key == ord("q"):
            break

        fps.update()
    except:
        print("alaba trap")
        break

fps.stop()
print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

cv2.destroyAllWindows()
vs.stop()

out = cv2.VideoWriter('project.avi', cv2.VideoWriter_fourcc(*'DIVX'), 30, frames[0].shape[:2])

for i in range(len(frames)):
    out.write(frames[i])
out.release()
