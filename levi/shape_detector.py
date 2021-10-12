import cv2


class ShapeDetector:
    def __init__(self):
        pass

    def detect(self, contour):
        shape = "unidentified"
        peri = cv2.arcLength(contour, True)
        # second parameter are normally in the range of 1-5%
        approx = cv2.approxPolyDP(contour, 0.03 * peri, True)

        numEdges = len(approx)
        if numEdges == 3:
            shape = "triangle"

        elif numEdges == 4:
            # x, y, w, h
            (_, _, w, h) = cv2.boundingRect(approx)
            aspectRatio = w / float(h)

            shape = "square" if aspectRatio >= 0.95 and aspectRatio <= 1.05 else "rectangle"

        elif numEdges == 5:
            shape = "pentagon"

        else:
            shape = "circle"

        return shape
