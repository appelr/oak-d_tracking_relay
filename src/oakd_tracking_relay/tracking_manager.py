import cv2
import numpy as np
import mediapipe as mp
from enum import Enum, auto

from oakd_tracking_relay.tracking_dto import *

class TrackerState(Enum):
    SEARCHING = auto()
    TRACKING = auto()

class EyeTracker:
    def __init__(self, utils, config):
        self.utils = utils
        self.config = config
        self.current_state = TrackerState.SEARCHING
        self.tracking_data = TrackingData()
        
        # MediaPipe FaceMesh 
        self.model = mp.solutions.face_mesh.FaceMesh(
            max_num_faces=4, 
            refine_landmarks=True, # Benötigt für Iris-Koordinaten
            static_image_mode=True, # Tracking-Infos ignorieren, da wir Optical Flow nutzen
            min_detection_confidence=float(config.confidence_percent)/100
        )
        
        # Schwellwerte für Wechsel Tracking -> Searching
        self.tracking_confidence_counter = 0
        self.tracking_confidence_init = 15
        self.tracking_confidence_minimum = 0

        # Schwellwerte für Wechsel Searching -> Tracking
        self.detection_buffer = []
        self.detection_buffer_max_size = 2

        # Counter für Mediapipe Recheck
        self.frame_count = 0
        self.recheck_interval = 15

        # Optical Flow
        self.previous_frame_left = None
        self.previous_frame_right = None
        self.optical_flow_params = dict(
            winSize=(8, 8), 
            maxLevel=5,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.03)
        )

        # Schwellwerte für Sprünge zwischen Mediapipe und OpticalFlow
        self.search_stability_threshold = 20
        self.max_diff_opticalflow_mediapipe = 1.5
        self.max_jump_between_frames = 35
        self.max_disparity_between_frames = 10

        # Ab wievielen Punkten ist TrackingData valide
        self.min_tracking_points = 2

    def process_stereo_frame(self, frame_left, frame_right):
        self.frame_count += 1

        if self.current_state == TrackerState.SEARCHING:
            self._search(frame_left, frame_right)
        elif self.current_state == TrackerState.TRACKING:
            self._track(frame_left, frame_right)

    def _decrease_confidence(self, amount=1):
        self.tracking_confidence_counter -= amount
        if self.tracking_confidence_counter <= self.tracking_confidence_minimum:
            self.tracking_data = TrackingData()
            self.detection_buffer = []
            self.current_state = TrackerState.SEARCHING

    def _search(self, frame_left, frame_right):
        detectedData = self.detect(frame_left, frame_right)

        if detectedData.valid():
            self.detection_buffer.append(detectedData)
            
            if len(self.detection_buffer) == self.detection_buffer_max_size:
                firstPoint = self.detection_buffer[0].aggregated.left
                lastPoint = self.detection_buffer[-1].aggregated.left
                dist = np.hypot(firstPoint.x - lastPoint.x, firstPoint.y - lastPoint.y)
                
                if dist < self.search_stability_threshold:
                    self.tracking_data = self.detection_buffer[-1]
                    self.tracking_confidence_counter = self.tracking_confidence_init
                    self.detection_buffer = []
                    self.current_state = TrackerState.TRACKING
                    
                    # WICHTIG: Bilder für den kommenden Optical Flow speichern!
                    self.previous_frame_left, self.previous_frame_right = frame_left.copy(), frame_right.copy()
                else:
                    self.detection_buffer.pop(0)
        else:
            self.detection_buffer = []

    def _track(self, frameL, frameR): 
        leftPointsLeftCam = np.array([p.left.as_np() for p in self.tracking_data.left.stereo_points], dtype=np.float32).reshape(-1, 1, 2)
        leftPointsRightCam = np.array([p.right.as_np() for p in self.tracking_data.left.stereo_points], dtype=np.float32).reshape(-1, 1, 2)
        rightPointsLeftCam = np.array([p.left.as_np() for p in self.tracking_data.right.stereo_points], dtype=np.float32).reshape(-1, 1, 2)
        rightPointsRightCam = np.array([p.right.as_np() for p in self.tracking_data.right.stereo_points], dtype=np.float32).reshape(-1, 1, 2)

        nextLeftPointsLeftCam, leftStatusLeftCam, _ = cv2.calcOpticalFlowPyrLK(self.previous_frame_left, frameL, leftPointsLeftCam, None, **self.optical_flow_params)
        nextLeftPointsRightCam, leftStatusRightCam, _ = cv2.calcOpticalFlowPyrLK(self.previous_frame_right, frameR, leftPointsRightCam, None, **self.optical_flow_params)
        nextRightPointsLeftCam, rightStatusLeftCam, _ = cv2.calcOpticalFlowPyrLK(self.previous_frame_left, frameL, rightPointsLeftCam, None, **self.optical_flow_params)
        nextRightPointsRightCam, rightStatusRightCam, _ = cv2.calcOpticalFlowPyrLK(self.previous_frame_right, frameR, rightPointsRightCam, None, **self.optical_flow_params)

        if len(leftPointsLeftCam) < self.min_tracking_points or len(rightPointsLeftCam) < self.min_tracking_points:
            self._decrease_confidence(amount=2)
            return

        data = TrackingData()

        for i in range(len(leftPointsLeftCam)):
            if leftStatusLeftCam[i][0] == 1 and leftStatusRightCam[i][0] == 1:
                leftNextL = Point2D(float(nextLeftPointsLeftCam[i][0][0]), float(nextLeftPointsLeftCam[i][0][1]))
                leftNextR = Point2D(float(nextLeftPointsRightCam[i][0][0]), float(nextLeftPointsRightCam[i][0][1]))
                data.left.stereo_points.append(StereoPoint(leftNextL, leftNextR))

        for i in range(len(rightPointsLeftCam)):
            if rightStatusLeftCam[i][0] == 1 and rightStatusRightCam[i][0] == 1:
                rightNextL = Point2D(float(nextRightPointsLeftCam[i][0][0]), float(nextRightPointsLeftCam[i][0][1]))
                rightNextR = Point2D(float(nextRightPointsRightCam[i][0][0]), float(nextRightPointsRightCam[i][0][1]))
                data.right.stereo_points.append(StereoPoint(rightNextL, rightNextR))

        if len(data.left.stereo_points) < self.min_tracking_points or len(data.right.stereo_points) < self.min_tracking_points:
            self._decrease_confidence(amount=2)
            return

        if data.valid():
            data.aggregate_median()

        oldCenter = self.tracking_data.aggregated
        newCenter = data.aggregated
        distX = newCenter.left.x - oldCenter.left.x
        distY = newCenter.left.y - oldCenter.left.y
        prevDisp = oldCenter.left.x - oldCenter.right.x
        currentDisp = newCenter.left.x - newCenter.right.x

        if np.hypot(distX, distY) > self.max_jump_between_frames or abs(currentDisp - prevDisp) > self.max_disparity_between_frames:
            self._decrease_confidence()
            return
        
        self.tracking_data = data
        self.previous_frame_left, self.previous_frame_right = frameL.copy(), frameR.copy()
        self.tracking_confidence_counter = min(self.tracking_confidence_counter + 1, self.tracking_confidence_init)
    
        # --- RECHECK ---
        if self.frame_count % self.recheck_interval == 0 or len(self.detection_buffer) > 0:
            recheckData = self.detect(frameL, frameR)

            if recheckData.valid():
                self.detection_buffer.append(recheckData)
                if len(self.detection_buffer) == self.detection_buffer_max_size:
                    firstPoint = self.detection_buffer[0].aggregated.left
                    lastPoint = self.detection_buffer[-1].aggregated.left
                    distMP = np.hypot(firstPoint.x - lastPoint.x, firstPoint.y - lastPoint.y)

                    if distMP < self.search_stability_threshold:
                        distMPOF = np.hypot(lastPoint.x - self.tracking_data.aggregated.left.x, lastPoint.y - self.tracking_data.aggregated.left.y) 
                        if distMPOF > self.max_diff_opticalflow_mediapipe:
                            self.tracking_data = recheckData
                            self.previous_frame_left, self.previous_frame_right = frameL.copy(), frameR.copy()
                            self.tracking_confidence_counter = self.tracking_confidence_init
                    self.detection_buffer = []
            else:
                self.detection_buffer = []

    def detect(self, frameL, frameR) -> TrackingData:
        centerL = Point2D()
        centerR = Point2D()

        if self.tracking_data.valid():
            centerL = self.tracking_data.aggregated.left
            centerR = self.tracking_data.aggregated.right

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
                data.left.stereo_points.append(StereoPoint(left, right))

            for idx in rightIrisIndices:
                left  = Point2D(offxL + landmarksL.landmark[idx].x * cropL.shape[1], offyL + landmarksL.landmark[idx].y * cropL.shape[0])
                right = Point2D(offxR + landmarksR.landmark[idx].x * cropR.shape[1], offyR + landmarksR.landmark[idx].y * cropR.shape[0])
                data.right.stereo_points.append(StereoPoint(left, right))

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
            min_detection_confidence=float(config.confidence_percent/100),
            min_tracking_confidence=float(config.confidence_percent/100)
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
        
        half_width = self.config.resolution_width / 2
        half_height = self.config.resolution_height / 2
        
        if results.multi_hand_landmarks:
            for landmarks in results.multi_hand_landmarks:
                x_pixel = landmarks.landmark[9].x * self.config.resolution_width
                y_pixel = landmarks.landmark[9].y * self.config.resolution_height
                
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