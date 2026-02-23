import cv2

import numpy as np
import mediapipe as mp

from .TrackingDTO import StereoPoint, Point2D
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
            self.stereoCoordinates = StereoPoint()
            self.prevFrameL = None
            self.prevFrameR = None
            self.trackingConfidence = 0
            self.frameCounter = 0
            self.detectionBuffer = []

            # Parameter
            self.maxJump = 20
            self.maxDispDelta = 3
            self.confidenceInit = 5
            self.confidenceMin = 0
            self.recheckInterval = 20
            self.opticalFlowParams = dict(winSize=(21, 21), maxLevel=3,
                                        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.03))
            

    def detect(self, frameL, frameR) -> StereoPoint:
        raise NotImplementedError("Please Implement this method")
    

    def processFrame(self, frameL, frameR):
        self.frameCounter += 1

        if self.currentState == TrackerState.SEARCHING:
            self.search(frameL, frameR)
            
        if self.currentState == TrackerState.TRACKING:
            self.track(frameL, frameR)


    def search(self, frameL, frameR):
        detectedPoint = self.detect(frameL, frameR)

        if detectedPoint.valid():
            self.detectionBuffer.append(detectedPoint)
            
            if len(self.detectionBuffer) >= 3:
                p1 = self.detectionBuffer[0].left
                p3 = self.detectionBuffer[-1].left
                
                # Distanz berechnen (hat sich das Auge bewegt?)
                dist = np.sqrt((p1.x - p3.x)**2 + (p1.y - p3.y)**2)
                
                # Wenn stabil (< 3 Pixel Bewegung) -> START TRACKING
                if dist < 3.0:
                    self.stereoCoordinates = self.detectionBuffer[-1]
                    self.prevFrameL, self.prevFrameR = frameL.copy(), frameR.copy()
                    self.trackingConfidence = self.confidenceInit
                    self.detectionBuffer = []
                    self.currentState = TrackerState.TRACKING
                else:
                    self.detectionBuffer.pop(0)
        else:
            self.detectionBuffer = []


    def track(self, frameL, frameR): 
        pointsL = self.stereoCoordinates.left.as_np().reshape(-1, 1, 2)
        pointsR = self.stereoCoordinates.right.as_np().reshape(-1, 1, 2)

        nextPointsL, statusL, _ = cv2.calcOpticalFlowPyrLK(self.prevFrameL, frameL, pointsL, None, **self.opticalFlowParams) # type: ignore
        nextPointsR, statusR, _ = cv2.calcOpticalFlowPyrLK(self.prevFrameR, frameR, pointsR, None, **self.opticalFlowParams) # type: ignore

        if statusL[0][0] == 1 and statusR[0][0] == 1:
            nextLx, nextLy = nextPointsL[0][0]
            nextRx, nextRy = nextPointsR[0][0]

            distX = nextLx - self.stereoCoordinates.left.x
            distY = nextLy - self.stereoCoordinates.left.y
            prevDisp = (self.stereoCoordinates.left.x - self.stereoCoordinates.right.x)
            currentDisp = nextLx - nextRx

            if np.hypot(distX, distY) > self.maxJump or abs(currentDisp - prevDisp) > self.maxDispDelta:
                self._decrease_confidence()
                return
            
            self.stereoCoordinates = StereoPoint(Point2D(nextLx, nextLy), Point2D(nextRx, nextRy))
            self.prevFrameL, self.prevFrameR = frameL.copy(), frameR.copy()
            self.trackingConfidence = min(self.trackingConfidence + 1, self.confidenceInit)
        else:
            self._decrease_confidence(amount=2)

        # if self.currentState == TrackerState.TRACKING and self.frameCounter % self.recheckInterval == 0:
        if self.frameCounter % self.recheckInterval == 0:
            recheck_point = self.detect(frameL, frameR)

            if recheck_point.valid():
                dist = np.hypot(recheck_point.left.x - self.stereoCoordinates.left.x, 
                                recheck_point.left.y - self.stereoCoordinates.left.y)

                if dist > 6:
                    self.stereoCoordinates = recheck_point
                    self.prevFrameL, self.prevFrameR = frameL.copy(), frameR.copy()
                    self.trackingConfidence = self.confidenceInit


    def _decrease_confidence(self, amount=1):
        self.trackingConfidence -= amount

        if self.trackingConfidence <= self.confidenceMin:
            self.stereoCoordinates = StereoPoint()
            self.detectionBuffer = []
            self.currentState = TrackerState.SEARCHING

class EyeTracker(TrackerBase):
    def __init__(self, utils, config):
        super().__init__(utils, config)
        self.model = mp.solutions.face_mesh.FaceMesh( # type: ignore[attr-defined]
            max_num_faces=2, 
            refine_landmarks=True,
            min_detection_confidence=float(config.mp_min_detection_percent)/100, 
            min_tracking_confidence=float(config.mp_min_tracking_percent/100))

    def detect(self, frameL, frameR) -> StereoPoint:
        lRGB = cv2.cvtColor(frameL, cv2.COLOR_GRAY2RGB)
        rRGB = cv2.cvtColor(frameR, cv2.COLOR_GRAY2RGB)
        
        resultsL = self.model.process(lRGB)
        resultsR = self.model.process(rRGB)

        if resultsL.multi_face_landmarks and resultsR.multi_face_landmarks:
            camLIrisL = resultsL.multi_face_landmarks[0].landmark[473]
            camRIrisL = resultsR.multi_face_landmarks[0].landmark[473]

            iris_stereo = StereoPoint(Point2D(camLIrisL.x, camLIrisL.y), Point2D(camRIrisL.x, camRIrisL.y))
            return self.utils.stereoLandmarkToPixelCoordinates(iris_stereo)
        
        return StereoPoint()


class HandTracker(TrackerBase):
    def __init__(self, utils, config):
        super().__init__(utils, config)
        self.model = mp.solutions.hands.Hands( # type: ignore[attr-defined]
            max_num_hands=2,
            model_complexity=0,
            min_detection_confidence=float(config.mp_min_detection_percent/100),
            min_tracking_confidence=float(config.mp_min_tracking_percent/100))
        
    def detect(self, frameL, frameR) -> StereoPoint:
        return StereoPoint()