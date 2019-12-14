import cv2
import numpy as np
import os
import time
import base64
import re
import json
from pathlib import Path
from AlternateControler.NetworkMessageType import NetworkMessageType
import params.Params as Params

class BlockDetector(object):
    def __init__(self, comPipe):
        self._initParametersWindow()
        #self._initWebcamParametersWindow()
        self._comPipe = comPipe
        self._capture = None
        self._lastContours = None
        self._lastRectangle = None
        self._lastCenter = None
        self._imgWithTemplate = None

        path = Path()
        self._templatePath = Path.joinpath(path.absolute(), Params.TEMPLATE_FOLDER)
        self._templateImgs = {}
        self._currentTemplate = -1
        self._nextTemplateNbr = None
        self.LoadAllTemplates()

        self._templateFilledIn = 0.0
        self._templateFilledOut = 0.0
        self._totalPixel = 0.0
        self._templatePixel = 0.0

    def _emptyCallBack(self, arg):
        pass

    def _cameraConfChanged(self, arg):
        if self._capture is not None:
            fps = 60 if cv2.getTrackbarPos('FrameRate', 'WebcamParam') == 0 else 50
            #self._capture.set(5, fps)
            self._capture.set(10, float(cv2.getTrackbarPos('Brightness', 'WebcamParam')) / 100.0)
            self._capture.set(11, float(cv2.getTrackbarPos('Contrast', 'WebcamParam')) / 100.0)
            self._capture.set(12, float(cv2.getTrackbarPos('Saturation', 'WebcamParam')) / 100.0)
            self._capture.set(14, float(cv2.getTrackbarPos('Gain', 'WebcamParam')) / 100.0)
            self._capture.set(15, float(cv2.getTrackbarPos('Exposition', 'WebcamParam')) / 100.0)


    def _initParametersWindow(self):
        cv2.namedWindow('Parameters')
        cv2.createTrackbar('Threshold', 'Parameters', 0, 255, self._emptyCallBack)
        cv2.createTrackbar('FillContours', 'Parameters', 0, 1, self._emptyCallBack)
        cv2.createTrackbar('UseTemplate', 'Parameters', 0, 1, self._emptyCallBack)
        cv2.createTrackbar('Template', 'Parameters', 0, 10, self._emptyCallBack)

    def _initWebcamParametersWindow(self):
        cv2.namedWindow('WebcamParam')
        cv2.createTrackbar('Brightness', 'WebcamParam', 0, 100, self._cameraConfChanged)
        cv2.createTrackbar('Contrast', 'WebcamParam', 0, 100, self._cameraConfChanged)
        cv2.createTrackbar('Saturation', 'WebcamParam', 0, 100, self._cameraConfChanged)
        cv2.createTrackbar('Gain', 'WebcamParam', 0, 100, self._cameraConfChanged)
        cv2.createTrackbar('Exposition', 'WebcamParam', 0, 100, self._cameraConfChanged)
        cv2.createTrackbar('FrameRate', 'WebcamParam', 0, 1, self._cameraConfChanged)


    def FindCountours(self, processedImg, base):
        # Find contours
        self._lastContours, h = cv2.findContours(processedImg.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in self._lastContours:
            if cv2.getTrackbarPos('FillContours', 'Parameters') == 1:
                cv2.fillPoly(base, [contour], (255, 0, 0))
            else:
                cv2.drawContours(base, [contour], -1, (255, 0, 0), 2)
            x, y, w, h = cv2.boundingRect(contour)
            self._lastRectangle = (x, y, w, h)
            self._lastCenter = (int((x + x + w) / 2), int((y + y + h) / 2))
            cv2.rectangle(base, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.circle(base, self._lastCenter, 2, (0, 255, 0), -1)
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
        #kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        #closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
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
        cv2.imwrite(str(Path.joinpath(self._templatePath, 'crt{}.png'.format(self._nextTemplateNbr))), crtImg)
        cv2.imwrite(str(Path.joinpath(self._templatePath, 'filled{}.png'.format(self._nextTemplateNbr))), filledImg)

        self._templateImgs[str(self._nextTemplateNbr)] = {}
        self._templateImgs[str(self._nextTemplateNbr)]['crt'] = cv2.imread(str(Path.joinpath(self._templatePath, 'crt{}.png'.format(self._nextTemplateNbr))))
        self._templateImgs[str(self._nextTemplateNbr)]['filled'] = cv2.imread(str(Path.joinpath(self._templatePath, 'filled{}.png'.format(self._nextTemplateNbr))))
        print('New template Saved and Added to {}'.format(self._nextTemplateNbr))
        self._nextTemplateNbr += 1

    def LoadAllTemplates(self):
        last = -1
        regex = r'(filled|crt)(\d+)\.png'
        tmpImg = {}
        for filename in os.listdir(self._templatePath):
            match = re.search(regex, filename)
            if match:
                name = match.group(1)
                number = match.group(2)
                if number not in tmpImg.keys():
                    tmpImg[number] = {}
                tmpImg[number][name] = cv2.imread(str(Path.joinpath(self._templatePath, filename)), cv2.IMREAD_UNCHANGED)
                if last < int(number):
                    last = int(number)

        self._templateImgs = {}
        for nbr in tmpImg.keys():
            current = tmpImg[nbr]
            if 'crt' in current.keys() and 'filled' in current.keys():
                self._templateImgs[nbr] = current
                # Template should have only one contour
                cnt, h = cv2.findContours(cv2.cvtColor(current['filled'], cv2.COLOR_BGR2GRAY), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                x, y, w, h = cv2.boundingRect(cnt[0])
                self._templateImgs[nbr]['boundingBox'] = (x, y, w, h)
                self._templateImgs[nbr]['boxCenter'] = (int((x + x + w) / 2), int((y + y + h) / 2))

                """tmp = current['filled'].copy()
                cv2.rectangle(tmp, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.circle(tmp, self._templateImgs[nbr]['boxCenter'], 2, (0, 255, 0), -1)
                cv2.imshow(nbr, tmp)"""
            else:
                print('Missing crt or filled for template {} : Removing'.format(nbr))

        self._nextTemplateNbr = last + 1
        print('Templates loaded ({})'.format(len(self._templateImgs)))

    def DisplayTemplate(self, base):
        crtTemplate = self._templateImgs[str(self._currentTemplate)]['crt']
        x, y, w, h = self._templateImgs[str(self._currentTemplate)]['boundingBox']
        cx, cy = self._templateImgs[str(self._currentTemplate)]['boxCenter']
        croppedTemplate = crtTemplate[y:y+h, x:x+w]
        baseOffset = (self._lastCenter[0] - cx, self._lastCenter[1] - cy)
        offset = (self._lastCenter[0] - cx + x, self._lastCenter[1] - cy + y)  # x ; y

        # it works now but should rename a bit
        templatedImg = base.copy()
        maxY = offset[1] + croppedTemplate.shape[0] if (offset[1] + croppedTemplate.shape[0]) < base.shape[0] else base.shape[0]
        maxX = offset[0] + croppedTemplate.shape[1] if (offset[0] + croppedTemplate.shape[1]) < base.shape[1] else base.shape[1]
        minY = offset[1] if offset[1] > 0 else 0
        minX = offset[0] if offset[0] > 0 else 0

        toRemoveW = (base.shape[1] - (offset[0] + croppedTemplate.shape[1])) if (base.shape[1] - (offset[0] + croppedTemplate.shape[1])) < 0 else 0
        toRemoveH = (base.shape[0] - (offset[1] + croppedTemplate.shape[0])) if (base.shape[0] - (offset[1] + croppedTemplate.shape[0])) < 0 else 0

        maxW = croppedTemplate.shape[1] + toRemoveW
        maxH = croppedTemplate.shape[0] + toRemoveH
        startW = -offset[0] if offset[0] < 0 else 0
        startH = -offset[1] if offset[1] < 0 else 0

        print(self._lastCenter)
        print(self._templateImgs[str(self._currentTemplate)]['boxCenter'])
        print(offset)
        print(croppedTemplate.shape)
        print(maxX)
        print(maxY)
        print(minX)
        print(minY)
        print(maxH)
        print(maxW)
        print('------------')

        alphaTemplate = croppedTemplate[startH:maxH, startW:maxW, 3] / 255.0
        inversedAlpha = 1 - alphaTemplate
        for c in range(0, 3):
            templatedImg[minY:maxY, minX:maxX, c] = (alphaTemplate * croppedTemplate[startH:maxH, startW:maxW, c] + inversedAlpha * templatedImg[minY:maxY, minX:maxX, c])

        # Draw both bounding and center : DEBUG
        x2, y2, w2, h2 = self._lastRectangle
        cv2.rectangle(templatedImg, (x2, y2), (x2 + w2, y2 + h2), (0, 255, 0), 2)
        cv2.circle(templatedImg, self._lastCenter, 2, (0, 255, 0), -1)
        cv2.rectangle(templatedImg, (x + baseOffset[0], y + baseOffset[1]), (x + w + baseOffset[0], y + h + baseOffset[1]), (255, 0, 0), 2)
        cv2.circle(templatedImg, (cx + baseOffset[0], cy + baseOffset[1]), 2, (255, 0, 0), -1)

        return templatedImg

    def CompareInOutValues2(self, templateNbr, processedImg):
        templateImg = self._templateImgs[str(templateNbr)]['filled']
        x, y, w, h = self._templateImgs[str(templateNbr)]['boundingBox']
        maxW = w
        maxH = h
        croppedTemplate = templateImg[y:y+h, x:x+w]

        for contour in self._lastContours:
            cv2.fillPoly(processedImg, [contour], (0, 0, 255))
        x, y, w, h = self._lastRectangle
        if w > maxW:
            maxW = w
        if h > maxH:
            maxH = h
        croppedBase = processedImg[y:y+h, x:x+w]

        emptyImg = np.zeros(shape=[maxH, maxW, 3])
        # base
        baseX, baseY = (maxW - croppedBase.shape[1])//2, (maxH - croppedBase.shape[0])//2
        baseResized = emptyImg.copy()
        baseResized[baseY:(baseY+croppedBase.shape[0]), baseX:(baseX+croppedBase.shape[1])] = croppedBase
        # template
        templateX, templateY = (maxW - croppedTemplate.shape[1])//2, (maxH - croppedTemplate.shape[0])//2
        templateResized = emptyImg.copy()
        templateResized[templateY:(templateY+croppedTemplate.shape[0]), templateX:(templateX+croppedTemplate.shape[1])] = croppedTemplate

        cv2.imshow('baseCropped', baseResized)
        cv2.imshow('templateCropped', templateResized)

        templateP = 0
        inP = 0
        outP = 0
        for y in range(0, maxH):
            for x in range(0, maxW):
                if np.all(templateResized[y, x] == (0, 0, 255)):
                    templateP += 1
                    if np.all(baseResized[y, x] == (0, 0, 255)):
                        inP += 1
                elif np.all(baseResized[y, x] == (0, 0, 255)):
                    outP += 1
        print('template : {}\nin : {}\nout : {}\n'.format(templateP, inP, outP))
        print('-------------------')
        self._templateFilledIn = float(inP)
        self._templateFilledOut = float(outP)
        self._templatePixel = float(templateP)

        retval, buffer = cv2.imencode('.png', croppedBase)

        toSend = {
            'filledIn': self._templateFilledIn / self._templatePixel * 100.0,
            'filledOut': self._templateFilledOut / (self._templateFilledOut + self._templateFilledIn) * 100.0,
            'croppedBase': base64.b64encode(buffer).decode('utf-8')
        }
        self._comPipe.send(json.dumps(toSend))
        """self._comPipe.send("{},{}".format(self._templateFilledIn / self._templatePixel * 100.0,
                                          self._templateFilledOut / (
                                                      self._templateFilledOut + self._templateFilledIn) * 100.0))"""


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
        print('-------------------')
        self._templateFilledIn = float(inP)
        self._templateFilledOut = float(outP)
        self._totalPixel = float(totalP)
        self._templatePixel = float(templateP)
        self._comPipe.send("{},{}".format(self._templateFilledIn / self._templatePixel * 100.0,
                            self._templateFilledOut / (self._templateFilledOut + self._templateFilledIn) * 100.0))

    def DisplayTextFilledPercent(self, imgWithTp):
        value = 0.0
        if self._templatePixel > 0:
            value = self._templateFilledIn / self._templatePixel * 100.0
        cv2.putText(imgWithTp, 'Filled In : {:.2f}%'.format(value), (0, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        value = 0.0
        if self._templateFilledOut + self._templateFilledIn > 0:
            value = self._templateFilledOut / (self._templateFilledOut + self._templateFilledIn) * 100.0
        # if self._totalPixel - self._templatePixel > 0:
        #    value = self._templateFilledOut / (self._totalPixel - self._templatePixel) * 100.0
        cv2.putText(imgWithTp, 'Filled Out : {:.2f}%'.format(value), (0, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        return imgWithTp

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
                print("Received from Com {}:{}".format(os.getpid(), pipeVal))
                # Can only receive template value atm
                if str(pipeVal) in self._templateImgs.keys():
                    self._currentTemplate = pipeVal
                    self._comPipe.send(NetworkMessageType.TEMPLATE_CHANGED.value[0])
                else:
                    self._comPipe.send(NetworkMessageType.TEMPLATE_UNKNOWN.value[0])

            processedImg = self.CheckFunction(frame)
            finalImg = self.FindCountours(processedImg, frame.copy())
            cv2.imshow('Webcam', finalImg)

            key = cv2.waitKey(1)
            if key == 27:
                break
            elif key == 32:
                self.SaveFrameAsTemplate((frame.shape[0], frame.shape[1]))
            elif key == 13:
                #if cv2.getTrackbarPos('UseTemplate', 'Parameters') == 1:
                if self._currentTemplate != -1:
                    currentTime = time.time()
                    self.CompareInOutValues2(self._currentTemplate, frame.copy())
                    if self._imgWithTemplate is None:
                        self._imgWithTemplate = self.DisplayTemplate(frame)
                    imgWithTp = self.DisplayTextFilledPercent(self._imgWithTemplate.copy())
                    cv2.imshow('WebcamTemplate', imgWithTp)
                    print('Time elipsed : {}s'.format(time.time() - currentTime))
            elif key == 8:  # no mvt
                # if cv2.getTrackbarPos('UseTemplate', 'Parameters') == 1:
                if self._currentTemplate != -1:
                    # self._currentTemplate = cv2.getTrackbarPos('Template', 'Parameters')
                    # if str(self._currentTemplate) in self._templateImgs.keys():
                    # imgWithTp = cv2.addWeighted(frame, 0.8, self._templateImgs[str(self._currentTemplate)]['crt'], 1, 0)
                    self._imgWithTemplate = self.DisplayTemplate(frame)
                    imgWithTp = self.DisplayTextFilledPercent(self._imgWithTemplate.copy())
                    cv2.imshow('WebcamTemplate', imgWithTp)

        self._capture.release()
        cv2.destroyAllWindows()

