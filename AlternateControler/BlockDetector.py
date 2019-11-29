import cv2
import numpy as np

class BlockDetector(object):
    def __init__(self):
        self._initParametersWindow()
        self._capture = None

    def _emptyCallBack(self, arg):
        pass

    def _initParametersWindow(self):
        cv2.namedWindow('Parameters')
        cv2.createTrackbar('Threshold', 'Parameters', 0, 255, self._emptyCallBack)
        cv2.createTrackbar('FillContours', 'Parameters', 0, 1, self._emptyCallBack)

    def FindCountours(self, processedImg, base):
        # Find contours
        contours, h = cv2.findContours(processedImg.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            if cv2.getTrackbarPos('FillContours', 'Parameters') == 1:
                cv2.fillPoly(base, [contour], (0, 0, 255))
            else:
                cv2.drawContours(base, [contour], -1, (0, 0, 255), 2)
        return base

    def CheckFunction(self, frame):
        # convert frame → TO IMPROVE
        # Greyscale
        grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Remove noise
        grey = cv2.fastNlMeansDenoising(grey, None, 5, 7, 21)
        grey = cv2.GaussianBlur(grey, (5, 5), 0)
        # grey = cv2.Canny(grey, 35, 125)
        cv2.imshow('grey', grey)

        # Threshold
        tValue = cv2.getTrackbarPos('Threshold', 'Parameters')
        thresh = cv2.threshold(grey, tValue, 255, cv2.THRESH_BINARY)[1]
        cv2.imshow('grey&threshold', thresh)

        # Fill close pixel by rectangle kernel
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 10))
        closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        # Improve borders
        closed = cv2.erode(closed, None, iterations=4)
        closed = cv2.dilate(closed, None, iterations=4)
        cv2.imshow('closed', closed)
        return closed

    def RunDetection(self):
        self._capture = cv2.VideoCapture(0)
        while True:
            ret, frame = self._capture.read()
            processedImg = self.CheckFunction(frame)
            finalImg = self.FindCountours(processedImg, frame)
            cv2.imshow('Webcam', finalImg)
            key = cv2.waitKey(1)
            if key == 27:
                break
        self._capture.release()
        cv2.destroyAllWindows()

