from levi import ShapeDetector
import argparse
import imutils
import cv2

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="path to input image")
args = vars(ap.parse_args())

image = cv2.imread(args["image"])
resized = imutils.resize(image, width=300)
scaleRatio = image.shape[0] / float(resized.shape[0])

gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (5, 5), 0)
thresh = cv2.threshold(blurred, 60, 255, cv2.THRESH_BINARY)[1]

cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
                        cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)

sd = ShapeDetector()
for cnt in cnts:
    M = cv2.moments(cnt)
    centerX = int((M["m10"] / M["m00"]) * scaleRatio)
    centerY = int((M["m01"] / M["m00"]) * scaleRatio)
    shape = sd.detect(cnt)

    cnt = cnt.astype("float")
    cnt *= scaleRatio
    cnt = cnt.astype("int")

    cv2.drawContours(image, [cnt], -1, (0, 255, 0), 2)
    cv2.putText(image, shape, (centerX, centerY),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    cv2.imshow("Shape", image)
    cv2.waitKey(0)
