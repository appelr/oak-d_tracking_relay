import cv2
import numpy as np
import mediapipe as mp
from enum import Enum, auto

from oakd_tracking_relay.TrackingDTO import *

class TrackerState(Enum):
    SEARCHING = auto()
    TRACKING = auto()

# =========================================================================================
# EYE TRACKER (Alles in einer Klasse vereint: Logik, Optical Flow & MediaPipe)
# =========================================================================================
class EyeTracker:
    def __init__(self, utils, config):
        self.utils = utils
        self.config = config
        
        # MediaPipe FaceMesh
        self.model = mp.solutions.face_mesh.FaceMesh(
            max_num_faces=4, 
            refine_landmarks=True,
            min_detection_confidence=float(config.mp_min_detection_percent)/100, 
            min_tracking_confidence=float(config.mp_min_tracking_percent/100)
        )
        
        # State & Logik
        self.currentState = TrackerState.SEARCHING
        self.trackingData = TrackingData()
        self.trackingConfidence = 0
        self.frameCounter = 0
        self.detectionBuffer = []
        self.detectionBufferMaxSize = 2
        self.confidenceInit = 15
        self.confidenceMin = 0

        # Thresholds
        self.searchStabilityThreshold = 20
        self.recheckCorrectionThreshold = 1.5
        self.recheckInterval = 15
        self.maxJump = 35
        self.maxDispDelta = 10
        self.minTrackingPoints = 2
        
        # Optical Flow
        self.prevFrameL = None
        self.prevFrameR = None
        self.opticalFlowParams = dict(
            winSize=(8, 8), 
            maxLevel=5,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.03)
        )

    def processFrame(self, frameL, frameR):
        self.frameCounter += 1

        if self.currentState == TrackerState.SEARCHING:
            self._search(frameL, frameR)
        elif self.currentState == TrackerState.TRACKING:
            self._track(frameL, frameR)

    def _decrease_confidence(self, amount=1):
        self.trackingConfidence -= amount
        if self.trackingConfidence <= self.confidenceMin:
            self.trackingData = TrackingData()
            self.detectionBuffer = []
            self.currentState = TrackerState.SEARCHING

    def _search(self, frameL, frameR):
        detectedData = self.detect(frameL, frameR)

        if detectedData.valid():
            self.detectionBuffer.append(detectedData)
            
            if len(self.detectionBuffer) == self.detectionBufferMaxSize:
                firstPoint = self.detectionBuffer[0].aggregated.left
                lastPoint = self.detectionBuffer[-1].aggregated.left
                dist = np.hypot(firstPoint.x - lastPoint.x, firstPoint.y - lastPoint.y)
                
                if dist < self.searchStabilityThreshold:
                    self.trackingData = self.detectionBuffer[-1]
                    self.trackingConfidence = self.confidenceInit
                    self.detectionBuffer = []
                    self.currentState = TrackerState.TRACKING
                    
                    # WICHTIG: Bilder für den kommenden Optical Flow speichern!
                    self.prevFrameL, self.prevFrameR = frameL.copy(), frameR.copy()
                else:
                    self.detectionBuffer.pop(0)
        else:
            self.detectionBuffer = []

    def _track(self, frameL, frameR): 
        leftPointsLeftCam = np.array([p.left.as_np() for p in self.trackingData.left.stereoPoints], dtype=np.float32).reshape(-1, 1, 2)
        leftPointsRightCam = np.array([p.right.as_np() for p in self.trackingData.left.stereoPoints], dtype=np.float32).reshape(-1, 1, 2)
        rightPointsLeftCam = np.array([p.left.as_np() for p in self.trackingData.right.stereoPoints], dtype=np.float32).reshape(-1, 1, 2)
        rightPointsRightCam = np.array([p.right.as_np() for p in self.trackingData.right.stereoPoints], dtype=np.float32).reshape(-1, 1, 2)

        nextLeftPointsLeftCam, leftStatusLeftCam, _ = cv2.calcOpticalFlowPyrLK(self.prevFrameL, frameL, leftPointsLeftCam, None, **self.opticalFlowParams)
        nextLeftPointsRightCam, leftStatusRightCam, _ = cv2.calcOpticalFlowPyrLK(self.prevFrameR, frameR, leftPointsRightCam, None, **self.opticalFlowParams)
        nextRightPointsLeftCam, rightStatusLeftCam, _ = cv2.calcOpticalFlowPyrLK(self.prevFrameL, frameL, rightPointsLeftCam, None, **self.opticalFlowParams)
        nextRightPointsRightCam, rightStatusRightCam, _ = cv2.calcOpticalFlowPyrLK(self.prevFrameR, frameR, rightPointsRightCam, None, **self.opticalFlowParams)

        if len(leftPointsLeftCam) < self.minTrackingPoints or len(rightPointsLeftCam) < self.minTrackingPoints:
            self._decrease_confidence(amount=2)
            return

        data = TrackingData()

        for i in range(len(leftPointsLeftCam)):
            if leftStatusLeftCam[i][0] == 1 and leftStatusRightCam[i][0] == 1:
                leftNextL = Point2D(float(nextLeftPointsLeftCam[i][0][0]), float(nextLeftPointsLeftCam[i][0][1]))
                leftNextR = Point2D(float(nextLeftPointsRightCam[i][0][0]), float(nextLeftPointsRightCam[i][0][1]))
                data.left.stereoPoints.append(StereoPoint(leftNextL, leftNextR))

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
    
        # --- RECHECK ---
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

    def detect(self, frameL, frameR) -> TrackingData:
        centerL = Point2D()
        centerR = Point2D()

        if self.trackingData.valid():
            centerL = self.trackingData.aggregated.left
            centerR = self.trackingData.aggregated.right

        # --- ROI Debug Window ---
        debugFrameL = cv2.cvtColor(frameL.copy(), cv2.COLOR_GRAY2BGR)
        cropL, offxL, offyL = self.utils.cropFrame(frameL, center=centerL)
        h, w = cropL.shape[:2]
        cv2.rectangle(debugFrameL, (offxL, offyL), (offxL + w, offyL + h), (0,255,0), 2)
        if centerL is not None:
            cv2.circle(debugFrameL, (int(centerL.x), int(centerL.y)), 5, (0,0,255), -1)
        cv2.imshow("ROI Debug", debugFrameL)
        # ------------------------

        cropL, offxL, offyL = self.utils.cropFrame(frameL, centerL)
        cropR, offxR, offyR = self.utils.cropFrame(frameR, centerR)

        cropLRGB = cv2.cvtColor(cropL, cv2.COLOR_GRAY2RGB)
        cropRRGB = cv2.cvtColor(cropR, cv2.COLOR_GRAY2RGB)

        resultsL = self.model.process(cropLRGB)
        resultsR = self.model.process(cropRRGB)

        data = TrackingData()

        if resultsL.multi_face_landmarks and resultsR.multi_face_landmarks:
            leftIrisIndices = [468, 469, 470, 471, 472]
            rightIrisIndices = [473, 474, 475, 476, 477]

            bestFaceId = self.utils.getBestFace(resultsL, resultsR)
            landmarksL = resultsL.multi_face_landmarks[bestFaceId]
            landmarksR = resultsR.multi_face_landmarks[bestFaceId]

            for idx in leftIrisIndices:
                left  = Point2D(offxL + landmarksL.landmark[idx].x * cropL.shape[1], offyL + landmarksL.landmark[idx].y * cropL.shape[0])
                right = Point2D(offxR + landmarksR.landmark[idx].x * cropR.shape[1], offyR + landmarksR.landmark[idx].y * cropR.shape[0])
                data.left.stereoPoints.append(StereoPoint(left, right))

            for idx in rightIrisIndices:
                left  = Point2D(offxL + landmarksL.landmark[idx].x * cropL.shape[1], offyL + landmarksL.landmark[idx].y * cropL.shape[0])
                right = Point2D(offxR + landmarksR.landmark[idx].x * cropR.shape[1], offyR + landmarksR.landmark[idx].y * cropR.shape[0])
                data.right.stereoPoints.append(StereoPoint(left, right))

            if data.valid():
                data.aggregate_median()

        return data


# =========================================================================================
# HAND TRACKER (Unabhängige Quadranten-Logik)
# =========================================================================================
class HandTracker:
    def __init__(self, config):
        self.model = mp.solutions.hands.Hands( 
            max_num_hands=2,
            model_complexity=0,
            min_detection_confidence=float(config.mp_min_detection_percent/100),
            min_tracking_confidence=float(config.mp_min_tracking_percent/100)
        )
        self.config = config
        
        self.patience = 5
        self.missing_oben_links = self.patience
        self.missing_unten_links = self.patience
        self.missing_oben_rechts = self.patience
        self.missing_unten_rechts = self.patience

    def check_presence(self, frameL):
        frame_rgb = cv2.cvtColor(frameL, cv2.COLOR_GRAY2RGB)
        results = self.model.process(frame_rgb)
        
        found_ol = False
        found_ul = False
        found_or = False
        found_ur = False
        
        half_width = self.config.resolutionWidth / 2
        half_height = self.config.resolutionHeight / 2
        
        if results.multi_hand_landmarks:
            for landmarks in results.multi_hand_landmarks:
                x_pixel = landmarks.landmark[9].x * self.config.resolutionWidth
                y_pixel = landmarks.landmark[9].y * self.config.resolutionHeight
                
                if x_pixel < half_width and y_pixel < half_height:
                    found_ol = True
                elif x_pixel < half_width and y_pixel >= half_height:
                    found_ul = True
                elif x_pixel >= half_width and y_pixel < half_height:
                    found_or = True
                elif x_pixel >= half_width and y_pixel >= half_height:
                    found_ur = True

        if found_ol: self.missing_oben_links = 0
        else: self.missing_oben_links += 1
            
        if found_ul: self.missing_unten_links = 0
        else: self.missing_unten_links += 1

        if found_or: self.missing_oben_rechts = 0
        else: self.missing_oben_rechts += 1
            
        if found_ur: self.missing_unten_rechts = 0
        else: self.missing_unten_rechts += 1

        is_oben_links = self.missing_oben_links < self.patience
        is_unten_links = self.missing_unten_links < self.patience
        is_oben_rechts = self.missing_oben_rechts < self.patience
        is_unten_rechts = self.missing_unten_rechts < self.patience
        
        return is_oben_links, is_unten_links, is_oben_rechts, is_unten_rechts