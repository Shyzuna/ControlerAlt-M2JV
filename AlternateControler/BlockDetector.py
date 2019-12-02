import cv2
import numpy as np
import os
from pathlib import Path

class BlockDetector(object):
    def __init__(self, comPipe):
        self._initParametersWindow()
        self._comPipe = comPipe
        self._capture = None
        self._templateImgs = {}
        self._lastContours = None
        self._currentTemplate = 0
        self._imgNbr = 2  # Do smth in order to start at the right number
        path = Path()
        self._pathToSave = Path.joinpath(path.absolute(), 'data')
        self.LoadTemplateImg(0)
        self.LoadTemplateImg(1)
        self._templateFilledIn = 0.0
        self._templateFilledOut = 0.0
        self._totalPixel = 0.0
        self._templatePixel = 0.0

    def _emptyCallBack(self, arg):
        pass

    def _initParametersWindow(self):
        cv2.namedWindow('Parameters')
        cv2.createTrackbar('Threshold', 'Parameters', 0, 255, self._emptyCallBack)
        cv2.createTrackbar('FillContours', 'Parameters', 0, 1, self._emptyCallBack)
        cv2.createTrackbar('UseTemplate', 'Parameters', 0, 1, self._emptyCallBack)
        #cv2.createTrackbar('Template', 'Parameters', 0, 10, self._emptyCallBack)

    def FindCountours(self, processedImg, base):
        # Find contours
        self._lastContours, h = cv2.findContours(processedImg.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in self._lastContours:
            if cv2.getTrackbarPos('FillContours', 'Parameters') == 1:
                cv2.fillPoly(base, [contour], (255, 0, 0))
            else:
                cv2.drawContours(base, [contour], -1, (255, 0, 0), 2)
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
        closed = thresh
        # Fill close pixel by rectangle kernel
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))
        closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        # Improve borders
        closed = cv2.erode(closed, None, iterations=4)
        closed = cv2.dilate(closed, None, iterations=4)
        cv2.imshow('closed', closed)
        return closed

    def SaveFrameAsTemplate(self, size):
        crtImg = np.zeros(shape=[size[0], size[1], 4])
        filledImg = np.zeros(shape=[size[0], size[1], 3])
        for contour in self._lastContours:
            cv2.drawContours(crtImg, [contour], -1, (0, 0, 255, 255), 2)
            cv2.fillPoly(filledImg, [contour], (0, 0, 255))
        cv2.imshow('crtImg', crtImg)
        cv2.imshow('filledImg', filledImg)
        cv2.imwrite(str(Path.joinpath(self._pathToSave, 'crt{}.png'.format(self._imgNbr))), crtImg)
        cv2.imwrite(str(Path.joinpath(self._pathToSave, 'filled{}.png'.format(self._imgNbr))), filledImg)
        self.LoadTemplateImg(self._imgNbr)
        self._imgNbr += 1

    def LoadTemplateImg(self, number):
        templates = {}
        templates['crt'] = cv2.imread(str(Path.joinpath(self._pathToSave, 'crt{}.png'.format(number))))
        templates['filled'] = cv2.imread(str(Path.joinpath(self._pathToSave, 'filled{}.png'.format(number))))
        self._templateImgs[str(number)] = templates

    def CompareInOutValues(self, templateNbr, processedImg):
        #  Improve compute time → cython / dll or other algo
        templateImg = self._templateImgs[str(templateNbr)]['filled']
        for contour in self._lastContours:
            cv2.fillPoly(processedImg, [contour], (0, 0, 255))
        h = templateImg.shape[0]
        w = templateImg.shape[1]
        totalP = w * h
        templateP = 0
        inP = 0
        outP = 0
        for y in range(0, h):
            for x in range(0, w):
                if np.all(templateImg[y, x] == (0, 0, 255)):
                    templateP += 1
                    if np.all(processedImg[y, x] == (0, 0, 255)):
                        inP += 1
                elif np.all(processedImg[y, x] == (0, 0, 255)):
                    outP += 1
        print('template : {}\nin : {}\nout : {}\ntotal : {}'.format(templateP, inP, outP, totalP))
        self._templateFilledIn = float(inP)
        self._templateFilledOut = float(outP)
        self._totalPixel = float(totalP)
        self._templatePixel = float(templateP)
        #self._comPipe.put_nowait(self._templateFilledIn)
        self._comPipe.send(inP)

    def RunDetection(self):
        self._capture = cv2.VideoCapture(0)
        while True:
            ret, frame = self._capture.read()
            cv2.imshow('base', frame)
            '''try:
                pipeVal = self._comPipe.get_nowait()
                print("{}:{}".format(os.getpid(), pipeVal))
            except Exception as e:
                pass'''

            if self._comPipe.poll():
                pipeVal = self._comPipe.recv()
                print("Received {}:{}".format(os.getpid(), pipeVal))

            if cv2.getTrackbarPos('UseTemplate', 'Parameters') == 1:
                #self._currentTemplate = cv2.getTrackbarPos('Template', 'Parameters')
                #if str(self._currentTemplate) in self._templateImgs.keys():
                imgWithTp = cv2.addWeighted(frame, 1, self._templateImgs[str(self._currentTemplate)]['crt'], 1, 0)
                value = 0.0
                if self._templatePixel > 0:
                    value = self._templateFilledIn / self._templatePixel * 100.0
                cv2.putText(imgWithTp, 'Filled In : {:.2f}%'.format(value), (0, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                value = 0.0
                if self._templateFilledOut + self._templateFilledIn > 0:
                    value = self._templateFilledOut / (self._templateFilledOut + self._templateFilledIn) * 100.0
                #if self._totalPixel - self._templatePixel > 0:
                #    value = self._templateFilledOut / (self._totalPixel - self._templatePixel) * 100.0
                cv2.putText(imgWithTp, 'Filled Out : {:.2f}%'.format(value), (0, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                cv2.imshow('WebcamTemplate', imgWithTp)

            processedImg = self.CheckFunction(frame)
            finalImg = self.FindCountours(processedImg, frame.copy())
            cv2.imshow('Webcam', finalImg)

            key = cv2.waitKey(1)
            if key == 27:
                break
            elif key == 32:
                self.SaveFrameAsTemplate((frame.shape[0], frame.shape[1]))
            elif key == 13:
                if cv2.getTrackbarPos('UseTemplate', 'Parameters') == 1:
                    self.CompareInOutValues(0, frame.copy())

        self._capture.release()
        cv2.destroyAllWindows()

