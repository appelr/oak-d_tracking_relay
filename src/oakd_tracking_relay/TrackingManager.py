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
            self.trackedData = StereoPointCluster()
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
            

    def detect(self, frameL, frameR) -> StereoPointCluster:
        raise NotImplementedError("Please Implement this method")
    

    def processFrame(self, frameL, frameR):
        self.frameCounter += 1

        if self.currentState == TrackerState.SEARCHING:
            self.search(frameL, frameR)
            
        elif self.currentState == TrackerState.TRACKING:
            self.track(frameL, frameR)


    def search(self, frameL, frameR):
        detectedPointCluster = self.detect(frameL, frameR)

        if detectedPointCluster.valid():
            self.detectionBuffer.append(detectedPointCluster)
            
            if len(self.detectionBuffer) >= 3:
                p1 = self.detectionBuffer[0].aggregated.left
                p3 = self.detectionBuffer[-1].aggregated.left
                
                # Distanz berechnen (hat sich das Auge bewegt?)
                dist = np.sqrt((p1.x - p3.x)**2 + (p1.y - p3.y)**2)
                
                # Wenn stabil (< 3 Pixel Bewegung) -> START TRACKING
                if dist < 3.0:
                    self.trackedData = self.detectionBuffer[-1]
                    self.prevFrameL, self.prevFrameR = frameL.copy(), frameR.copy()
                    self.trackingConfidence = self.confidenceInit
                    self.detectionBuffer = []
                    self.currentState = TrackerState.TRACKING
                else:
                    self.detectionBuffer.pop(0)
        else:
            self.detectionBuffer = []


    def track(self, frameL, frameR): 
        pointsL = [p.left.as_np() for p in self.trackedData.stereoPoints]
        pointsR = [p.right.as_np() for p in self.trackedData.stereoPoints]

        pointsL = np.array(pointsL, dtype=np.float32).reshape(-1, 1, 2)
        pointsR = np.array(pointsR, dtype=np.float32).reshape(-1, 1, 2)

        # pointsL = self.trackedData.left.as_np().reshape(-1, 1, 2)
        # pointsR = self.trackedData.right.as_np().reshape(-1, 1, 2)

        nextPointsL, statusL, _ = cv2.calcOpticalFlowPyrLK(self.prevFrameL, frameL, pointsL, None, **self.opticalFlowParams) # type: ignore
        nextPointsR, statusR, _ = cv2.calcOpticalFlowPyrLK(self.prevFrameR, frameR, pointsR, None, **self.opticalFlowParams) # type: ignore

        cluster = StereoPointCluster()

        for i in range(len(pointsL)):
            if statusL[i][0] == 1 and statusR[i][0] == 1:
                nextL = Point2D(float(nextPointsL[i][0][0]), float(nextPointsL[i][0][1]))
                nextR = Point2D(float(nextPointsR[i][0][0]), float(nextPointsR[i][0][1]))
                cluster.stereoPoints.append(StereoPoint(nextL, nextR))

        if len(cluster.stereoPoints) < 2:
            self._decrease_confidence(amount=2)
            return

        cluster.aggregate_mean()

        oldCenter = self.trackedData.aggregated
        newCenter = cluster.aggregated

        distX = newCenter.left.x - oldCenter.left.x
        distY = newCenter.left.y - oldCenter.left.y
        prevDisp = oldCenter.left.x - oldCenter.right.x
        currentDisp = newCenter.left.x - newCenter.right.x

        if np.hypot(distX, distY) > self.maxJump or abs(currentDisp - prevDisp) > self.maxDispDelta:
            self._decrease_confidence()
            return
        
        self.trackedData = cluster
        self.prevFrameL, self.prevFrameR = frameL.copy(), frameR.copy()
        self.trackingConfidence = min(self.trackingConfidence + 1, self.confidenceInit)
    
        # RECHECK
        # if self.currentState == TrackerState.TRACKING and self.frameCounter % self.recheckInterval == 0:
        if self.frameCounter % self.recheckInterval == 0:
            recheckPointCluster = self.detect(frameL, frameR)

            if recheckPointCluster.valid():
                dist = np.hypot(recheckPointCluster.aggregated.left.x - self.trackedData.aggregated.left.x, 
                                recheckPointCluster.aggregated.left.y - self.trackedData.aggregated.left.y)

                if dist > 6:
                    self.trackedData = recheckPointCluster
                    self.prevFrameL, self.prevFrameR = frameL.copy(), frameR.copy()
                    self.trackingConfidence = self.confidenceInit


    def _decrease_confidence(self, amount=1):
        self.trackingConfidence -= amount

        if self.trackingConfidence <= self.confidenceMin:
            self.trackedData = StereoPointCluster()
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

    def detect(self, frameL, frameR) -> StereoPointCluster:
        lRGB = cv2.cvtColor(frameL, cv2.COLOR_GRAY2RGB)
        rRGB = cv2.cvtColor(frameR, cv2.COLOR_GRAY2RGB)
        
        resultsL = self.model.process(lRGB)
        resultsR = self.model.process(rRGB)

        cluster = StereoPointCluster()

        if resultsL.multi_face_landmarks and resultsR.multi_face_landmarks:
            iris_indices = [473, 474, 475, 476, 477]
            for id in iris_indices:
                camLIrisL = resultsL.multi_face_landmarks[0].landmark[id]
                camRIrisL = resultsR.multi_face_landmarks[0].landmark[id]

                iris_stereo = StereoPoint(Point2D(camLIrisL.x, camLIrisL.y), Point2D(camRIrisL.x, camRIrisL.y))
                iris_stereo_px = self.utils.stereoLandmarkToPixelCoordinates(iris_stereo)
                cluster.stereoPoints.append(iris_stereo_px)

            if cluster.valid():
                cluster.aggregate_mean()
        
        return cluster


class HandTracker(TrackerBase):
    def __init__(self, utils, config):
        super().__init__(utils, config)
        self.model = mp.solutions.hands.Hands( # type: ignore[attr-defined]
            max_num_hands=2,
            model_complexity=0,
            min_detection_confidence=float(config.mp_min_detection_percent/100),
            min_tracking_confidence=float(config.mp_min_tracking_percent/100))
        
    def detect(self, frameL, frameR) -> StereoPointCluster:
        return StereoPointCluster()