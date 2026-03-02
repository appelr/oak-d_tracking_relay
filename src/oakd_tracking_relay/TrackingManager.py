import cv2

import numpy as np
import mediapipe as mp

from .TrackingDTO import *
from enum import Enum, auto

class TrackerState(Enum):
    SEARCHING = auto()
    TRACKING = auto()

class TrackerBase():
    def __init__(self, utils, config):
            self.utils = utils
            self.config = config
            self.model = None
            self.currentState = TrackerState.SEARCHING
            self.trackingData = TrackingData()
            self.prevFrameL = None
            self.prevFrameR = None
            self.trackingConfidence = 0
            self.frameCounter = 0
            self.detectionBuffer = []
            self.detectionBufferMaxSize = 2
            self.minTrackingPoints = 2
            self.confidenceInit = 15
            self.confidenceMin = 0

            # Thresholds
            self.maxJump = 35
            self.maxDispDelta = 10
            self.searchStabilityThreshold = 20
            self.recheckCorrectionThreshold = 3
            self.recheckInterval = 10
            self.opticalFlowParams = dict(winSize=(12, 12), maxLevel=5,
                                        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.03))
            

    def detect(self, frameL, frameR) -> TrackingData:
        raise NotImplementedError("Please Implement this method")
    

    def processFrame(self, frameL, frameR):
        self.frameCounter += 1

        if self.currentState == TrackerState.SEARCHING:
            self.search(frameL, frameR)
            
        elif self.currentState == TrackerState.TRACKING:
            self.track(frameL, frameR)


    def search(self, frameL, frameR):
        detectedData = self.detect(frameL, frameR)

        if detectedData.valid():
            self.detectionBuffer.append(detectedData)
            
            if len(self.detectionBuffer) == self.detectionBufferMaxSize:
                firstPoint = self.detectionBuffer[0].aggregated.left
                lastPoint = self.detectionBuffer[-1].aggregated.left
                
                # Distanz berechnen (hat sich das Auge bewegt?)
                dist = np.sqrt((firstPoint.x - lastPoint.x)**2 + (firstPoint.y - lastPoint.y)**2)
                
                # Wenn stabil (< 10 Pixel Bewegung) -> START TRACKING
                if dist < self.searchStabilityThreshold:
                    self.trackingData = self.detectionBuffer[-1]
                    self.prevFrameL, self.prevFrameR = frameL.copy(), frameR.copy()
                    self.trackingConfidence = self.confidenceInit
                    self.detectionBuffer = []
                    self.currentState = TrackerState.TRACKING
                else:
                    self.detectionBuffer.pop(0)
        else:
            self.detectionBuffer = []


    def track(self, frameL, frameR): 
        leftPointsLeftCam = [p.left.as_np() for p in self.trackingData.left.stereoPoints]
        leftPointsRightCam = [p.right.as_np() for p in self.trackingData.left.stereoPoints]
        rightPointsLeftCam = [p.left.as_np() for p in self.trackingData.right.stereoPoints]
        rightPointsRightCam = [p.right.as_np() for p in self.trackingData.right.stereoPoints]

        leftPointsLeftCam = np.array(leftPointsLeftCam, dtype=np.float32).reshape(-1, 1, 2)
        leftPointsRightCam = np.array(leftPointsRightCam, dtype=np.float32).reshape(-1, 1, 2)
        rightPointsLeftCam = np.array(rightPointsLeftCam, dtype=np.float32).reshape(-1, 1, 2)
        rightPointsRightCam = np.array(rightPointsRightCam, dtype=np.float32).reshape(-1, 1, 2)

        nextLeftPointsLeftCam, leftStatusLeftCam, _ = cv2.calcOpticalFlowPyrLK(self.prevFrameL, frameL, leftPointsLeftCam, None, **self.opticalFlowParams) # type: ignore
        nextLeftPointsRightCam, leftStatusRightCam, _ = cv2.calcOpticalFlowPyrLK(self.prevFrameR, frameR, leftPointsRightCam, None, **self.opticalFlowParams) # type: ignore
        nextRightPointsLeftCam, rightStatusLeftCam, _ = cv2.calcOpticalFlowPyrLK(self.prevFrameL, frameL, rightPointsLeftCam, None, **self.opticalFlowParams) # type: ignore
        nextRightPointsRightCam, rightStatusRightCam, _ = cv2.calcOpticalFlowPyrLK(self.prevFrameR, frameR, rightPointsRightCam, None, **self.opticalFlowParams) # type: ignore

        # Wenn ein Auge komplett verloren wurde, brich ab, bevor OpenCV crasht!
        if len(leftPointsLeftCam) < self.minTrackingPoints or len(rightPointsLeftCam) < self.minTrackingPoints:
            self._decrease_confidence(amount=2)
            return

        data = TrackingData()

        for i in range(len(leftPointsLeftCam)):
            if leftStatusLeftCam[i][0] == 1 and leftStatusRightCam[i][0] == 1:
                leftNextL = Point2D(float(nextLeftPointsLeftCam[i][0][0]), float(nextLeftPointsLeftCam[i][0][1]))
                leftNextR = Point2D(float(nextLeftPointsRightCam[i][0][0]), float(nextLeftPointsRightCam[i][0][1]))
                data.left.stereoPoints.append(StereoPoint(leftNextL, leftNextR))

        # Rechtes Auge separat auswerten
        for i in range(len(rightPointsLeftCam)):
            if rightStatusLeftCam[i][0] == 1 and rightStatusRightCam[i][0] == 1:
                rightNextL = Point2D(float(nextRightPointsLeftCam[i][0][0]), float(nextRightPointsLeftCam[i][0][1]))
                rightNextR = Point2D(float(nextRightPointsRightCam[i][0][0]), float(nextRightPointsRightCam[i][0][1]))
                data.right.stereoPoints.append(StereoPoint(rightNextL, rightNextR))

        if len(data.left.stereoPoints) < self.minTrackingPoints or len(data.right.stereoPoints) < self.minTrackingPoints:
            self._decrease_confidence(amount=2)
            return

        if data.valid():
            data.aggregate_median()

        oldCenter = self.trackingData.aggregated
        newCenter = data.aggregated

        distX = newCenter.left.x - oldCenter.left.x
        distY = newCenter.left.y - oldCenter.left.y
        prevDisp = oldCenter.left.x - oldCenter.right.x
        currentDisp = newCenter.left.x - newCenter.right.x

        if np.hypot(distX, distY) > self.maxJump or abs(currentDisp - prevDisp) > self.maxDispDelta:
            self._decrease_confidence()
            return
        
        self.trackingData = data
        self.prevFrameL, self.prevFrameR = frameL.copy(), frameR.copy()
        self.trackingConfidence = min(self.trackingConfidence + 1, self.confidenceInit)
    
        # RECHECK
        # if self.currentState == TrackerState.TRACKING and self.frameCounter % self.recheckInterval == 0:
        if self.frameCounter % self.recheckInterval == 0 or len(self.detectionBuffer) > 0:
            recheckData = self.detect(frameL, frameR)

            if recheckData.valid():
                self.detectionBuffer.append(recheckData)

                if len(self.detectionBuffer) == self.detectionBufferMaxSize:
                    firstPoint = self.detectionBuffer[0].aggregated.left
                    lastPoint = self.detectionBuffer[-1].aggregated.left

                    distMP = np.hypot(firstPoint.x - lastPoint.x, firstPoint.y - lastPoint.y)

                    if distMP < self.searchStabilityThreshold:
                        distMPOF = np.hypot(lastPoint.x - self.trackingData.aggregated.left.x, lastPoint.y - self.trackingData.aggregated.left.y) 
                        
                        if distMPOF > self.recheckCorrectionThreshold:
                            self.trackingData = recheckData
                            self.prevFrameL, self.prevFrameR = frameL.copy(), frameR.copy()
                            self.trackingConfidence = self.confidenceInit
                    
                    self.detectionBuffer = []
            else:
                self.detectionBuffer = []


    def _decrease_confidence(self, amount=1):
        self.trackingConfidence -= amount

        if self.trackingConfidence <= self.confidenceMin:
            self.trackingData = TrackingData()
            self.detectionBuffer = []
            self.currentState = TrackerState.SEARCHING

class EyeTracker(TrackerBase):
    def __init__(self, utils, config):
        super().__init__(utils, config)
        self.model = mp.solutions.face_mesh.FaceMesh( # type: ignore[attr-defined]
            max_num_faces=4, 
            refine_landmarks=True,
            min_detection_confidence=float(config.mp_min_detection_percent)/100, 
            min_tracking_confidence=float(config.mp_min_tracking_percent/100))

    def detect(self, frameL, frameR) -> TrackingData:
        lRGB = cv2.cvtColor(frameL, cv2.COLOR_GRAY2RGB)
        rRGB = cv2.cvtColor(frameR, cv2.COLOR_GRAY2RGB)
        
        resultsL = self.model.process(lRGB)
        resultsR = self.model.process(rRGB)

        data = TrackingData()

        if resultsL.multi_face_landmarks and resultsR.multi_face_landmarks:
            leftIrisIndices = [468, 469, 470, 471, 472]
            rightIrisIndices = [473, 474, 475, 476, 477]

            bestFaceId = self.utils.selectBestFace(resultsL, resultsR)

            for leftId in leftIrisIndices:
                camLIrisL = resultsL.multi_face_landmarks[bestFaceId].landmark[leftId]
                camRIrisL = resultsR.multi_face_landmarks[bestFaceId].landmark[leftId]
                leftIris = StereoPoint(Point2D(camLIrisL.x, camLIrisL.y), Point2D(camRIrisL.x, camRIrisL.y))
                leftIrisPx = self.utils.stereoLandmarkToPixelCoordinates(leftIris)
                data.left.stereoPoints.append(leftIrisPx)

            for rightId in rightIrisIndices:
                camLIrisR = resultsL.multi_face_landmarks[bestFaceId].landmark[rightId]
                camRIrisR = resultsR.multi_face_landmarks[bestFaceId].landmark[rightId]
                rightIris = StereoPoint(Point2D(camLIrisR.x, camLIrisR.y), Point2D(camRIrisR.x, camRIrisR.y))
                rightIrisPx = self.utils.stereoLandmarkToPixelCoordinates(rightIris)
                data.right.stereoPoints.append(rightIrisPx)

            if data.valid():
                data.aggregate_median()
        
        return data


class HandTracker(TrackerBase):
    def __init__(self, utils, config):
        super().__init__(utils, config)
        self.model = mp.solutions.hands.Hands( # type: ignore[attr-defined]
            max_num_hands=2,
            model_complexity=0,
            min_detection_confidence=float(config.mp_min_detection_percent/100),
            min_tracking_confidence=float(config.mp_min_tracking_percent/100))
        
    def detect(self, frameL, frameR) -> TrackingData:
        return TrackingData()