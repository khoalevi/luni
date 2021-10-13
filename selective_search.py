import argparse
import random
import time
import cv2

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
                help="path to the input image")
ap.add_argument("-m", "--method", type=str, default="fast",
                choices=["fast", "quality"],
                help="selective search method")
args = ap.parse_args()

image = cv2.imread(args.image)

ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
ss.setBaseImage(image)

if args.method == "fast":
    print("[INFO] using *fast* selective search")
    ss.switchToSelectiveSearchQuality()
else:
    print("[INFO] using *quality* selective search")
    ss.switchToSelectiveSearchQuality()

start = time.time()
boxes = ss.process()
end = time.time()

print("[INFO] selective search took {:.4f} seconds".format(end - start))
print("[INFO] {} total region proposals".format(len(boxes)))

for i in range(0, len(boxes), 100):
    output = image.copy()

    for (x, y, w, h) in boxes[i:i + 100]:
        color = [random.randint(0, 255) for j in range(0, 3)]
        cv2.rectangle(output, (x, y), (x + w, y + h), color, 2)

    cv2.imshow("Output", output)
    key = cv2.waitKey(0) & 0xFF

    if key == ord("q"):
        break
