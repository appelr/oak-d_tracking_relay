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
        self.model = mp.solutions.face_mesh.FaceMesh( # type: ignore
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
            self._detect(frame_left=frame_left, frame_right=frame_right)
        elif self.current_state == TrackerState.TRACKING:
            self._track(frame_left=frame_left, frame_right=frame_right)

    def _detect(self, frame_left, frame_right):
        detected_data = self._run_detection(frame_left, frame_right)

        if detected_data.valid():
            self.detection_buffer.append(detected_data)
            
            if len(self.detection_buffer) == self.detection_buffer_max_size:
                first_point = self.detection_buffer[0].aggregated.left
                last_point = self.detection_buffer[-1].aggregated.left
                distance = np.hypot(first_point.x - last_point.x, first_point.y - last_point.y)
                
                if distance < self.search_stability_threshold:
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

    def _track(self, frame_left, frame_right): 
        left_points_left_cam = np.array([p.left.as_np() for p in self.tracking_data.left.stereo_points], dtype=np.float32).reshape(-1, 1, 2)
        left_points_right_cam = np.array([p.right.as_np() for p in self.tracking_data.left.stereo_points], dtype=np.float32).reshape(-1, 1, 2)
        right_points_left_cam = np.array([p.left.as_np() for p in self.tracking_data.right.stereo_points], dtype=np.float32).reshape(-1, 1, 2)
        right_points_right_cam = np.array([p.right.as_np() for p in self.tracking_data.right.stereo_points], dtype=np.float32).reshape(-1, 1, 2)

        # Optical Flow auf Tracking Daten anwenden
        next_left_points_left_cam, left_status_left_cam, _ = cv2.calcOpticalFlowPyrLK(self.previous_frame_left, frame_left, left_points_left_cam, None, **self.optical_flow_params) # type: ignore

        next_left_points_right_cam, left_status_right_cam, _ = cv2.calcOpticalFlowPyrLK(self.previous_frame_right, frame_right, left_points_right_cam, None, **self.optical_flow_params) # type: ignore

        next_right_points_left_cam, right_status_left_cam, _ = cv2.calcOpticalFlowPyrLK(self.previous_frame_left, frame_left, right_points_left_cam, None, **self.optical_flow_params) # type: ignore

        next_right_points_right_cam, right_status_right_cam, _ = cv2.calcOpticalFlowPyrLK(self.previous_frame_right, frame_right, right_points_right_cam, None, **self.optical_flow_params) # type: ignore

        # Mindestanzahl Punkte muss über Schwellwert liegen um Tracking zu starten
        if len(left_points_left_cam) < self.min_tracking_points or len(right_points_left_cam) < self.min_tracking_points:
            self._decrease_confidence(amount=2)
            return

        data = TrackingData()

        # DTO erstellen
        for i in range(len(left_points_left_cam)):
            if left_status_left_cam[i][0] == 1 and left_status_right_cam[i][0] == 1:
                next_left_point_left_cam = Point2D(float(next_left_points_left_cam[i][0][0]), float(next_left_points_left_cam[i][0][1]))
                netx_left_point_right_cam = Point2D(float(next_left_points_right_cam[i][0][0]), float(next_left_points_right_cam[i][0][1]))
                data.left.stereo_points.append(StereoPoint(next_left_point_left_cam, netx_left_point_right_cam))

        for i in range(len(right_points_left_cam)):
            if right_status_left_cam[i][0] == 1 and right_status_right_cam[i][0] == 1:
                next_right_point_left_cam = Point2D(float(next_right_points_left_cam[i][0][0]), float(next_right_points_left_cam[i][0][1]))
                next_right_point_right_cam = Point2D(float(next_right_points_right_cam[i][0][0]), float(next_right_points_right_cam[i][0][1]))
                data.right.stereo_points.append(StereoPoint(next_right_point_left_cam, next_right_point_right_cam))

        # Mindestanzahl getrackter Punkte muss über Schwellwert liegen
        if len(data.left.stereo_points) < self.min_tracking_points or len(data.right.stereo_points) < self.min_tracking_points:
            self._decrease_confidence(amount=2)
            return

        # Zentrum der getrackten Punkte bestimmen (Iris)
        if data.valid():
            data.aggregate_median()

        old_center = self.tracking_data.aggregated
        new_center = data.aggregated
        distance_x = new_center.left.x - old_center.left.x
        distance_y = new_center.left.y - old_center.left.y
        previous_disparity = old_center.left.x - old_center.right.x
        current_disparity = new_center.left.x - new_center.right.x

        # Sprung zwischen Frames muss innerhalb des Schwellwertes liegen
        if np.hypot(distance_x, distance_y) > self.max_jump_between_frames or abs(current_disparity - previous_disparity) > self.max_disparity_between_frames:
            self._decrease_confidence()
            return
        
        self.tracking_data = data
        self.previous_frame_left, self.previous_frame_right = frame_left.copy(), frame_right.copy()
        self.tracking_confidence_counter = min(self.tracking_confidence_counter + 1, self.tracking_confidence_init)
    
        # Mediapipe Recheck, wenn Intervall erreicht wurde
        if self.frame_count % self.recheck_interval == 0 or len(self.detection_buffer) > 0:
            recheck_data = self._run_detection(frame_left=frame_left, frame_right=frame_right)

            if recheck_data.valid():
                self.detection_buffer.append(recheck_data)
                if len(self.detection_buffer) == self.detection_buffer_max_size:
                    first_bufffer_element = self.detection_buffer[0].aggregated.left
                    last_buffer_element = self.detection_buffer[-1].aggregated.left

                    distance = np.hypot(first_bufffer_element.x - last_buffer_element.x, first_bufffer_element.y - last_buffer_element.y)

                    # Abweichung innerhalb der letzten x Detection-Daten muss innerhalb des Schwellwertes sein
                    if distance < self.search_stability_threshold:
                        distance_tracking_to_detection_data = np.hypot(last_buffer_element.x - self.tracking_data.aggregated.left.x, last_buffer_element.y - self.tracking_data.aggregated.left.y) 

                        # Distanz zwischen Optical Flow und Mediapipe muss innerhalb des Schwellwertes sein
                        if distance_tracking_to_detection_data > self.max_diff_opticalflow_mediapipe:
                            self.tracking_data = recheck_data
                            self.previous_frame_left, self.previous_frame_right = frame_left.copy(), frame_right.copy()
                            self.tracking_confidence_counter = self.tracking_confidence_init
                    self.detection_buffer = []
            else:
                self.detection_buffer = []

    def _run_detection(self, frame_left, frame_right) -> TrackingData:
        center_left = Point2D()
        center_right = Point2D()

        # Bildzentrum für crop auf letzte valide TrackingData setzen
        if self.tracking_data.valid():
            center_left = self.tracking_data.aggregated.left
            center_right = self.tracking_data.aggregated.right

        # Debug Fenster für ROI Crop und Bewegung
        debug_frame = cv2.cvtColor(frame_left.copy(), cv2.COLOR_GRAY2BGR)
        crop_left, offset_x_left, offset_y_left = self.utils.crop_frame(frame_left, center=center_left)
        height, width = crop_left.shape[:2]
        cv2.rectangle(debug_frame, (offset_x_left, offset_y_left), (offset_x_left + width, offset_y_left + height), (0,255,0), 2)
        if center_left is not None:
            cv2.circle(debug_frame, (int(center_left.x), int(center_left.y)), 5, (0,0,255), -1)
        cv2.imshow("ROI Debug", debug_frame)

        # Frame Crop
        crop_left, offset_x_left, offset_y_left = self.utils.crop_frame(frame_left, center_left)
        crop_right, offset_x_right, offset_y_right = self.utils.crop_frame(frame_right, center_right)

        crop_left_rgb = cv2.cvtColor(crop_left, cv2.COLOR_GRAY2RGB)
        crop_right_rgb = cv2.cvtColor(crop_right, cv2.COLOR_GRAY2RGB)

        results_left = self.model.process(crop_left_rgb)
        results_right = self.model.process(crop_right_rgb)

        data = TrackingData()

        # Augen-Landmark Detection
        if results_left.multi_face_landmarks and results_right.multi_face_landmarks:
            LEFT_IRIS_INDICES = [468, 469, 470, 471, 472]
            RIGHT_IRIS_INDICES = [473, 474, 475, 476, 477]

            best_face_id = self.utils.get_best_face(results_left, results_right)
            landmarks_left = results_left.multi_face_landmarks[best_face_id]
            landmarks_right = results_right.multi_face_landmarks[best_face_id]

            for id in LEFT_IRIS_INDICES:
                left  = Point2D(offset_x_left + landmarks_left.landmark[id].x * crop_left.shape[1], offset_y_left + landmarks_left.landmark[id].y * crop_left.shape[0])
                right = Point2D(offset_x_right + landmarks_right.landmark[id].x * crop_right.shape[1], offset_y_right + landmarks_right.landmark[id].y * crop_right.shape[0])
                data.left.stereo_points.append(StereoPoint(left, right))

            for id in RIGHT_IRIS_INDICES:
                left  = Point2D(offset_x_left + landmarks_left.landmark[id].x * crop_left.shape[1], offset_y_left + landmarks_left.landmark[id].y * crop_left.shape[0])
                right = Point2D(offset_x_right + landmarks_right.landmark[id].x * crop_right.shape[1], offset_y_right + landmarks_right.landmark[id].y * crop_right.shape[0])
                data.right.stereo_points.append(StereoPoint(left, right))
            
            # Zentrum der gefundenen Punkte bestimmen (Iris)
            if data.valid():
                data.aggregate_median()

        return data
    
    def _decrease_confidence(self, amount=1):
        self.tracking_confidence_counter -= amount
        if self.tracking_confidence_counter <= self.tracking_confidence_minimum:
            self.tracking_data = TrackingData()
            self.detection_buffer = []
            self.current_state = TrackerState.SEARCHING

class HandTracker:
    def __init__(self, utils, config):
        self.model = mp.solutions.hands.Hands(  # type: ignore
            max_num_hands=2,
            model_complexity=0,
            min_detection_confidence=float(config.confidence_percent/100),
            min_tracking_confidence=float(config.confidence_percent/100)
        )
        self.config = config
        self.utils = utils
        
        # Toleranz für Fehlklassifizierung von fehlenden Handpositionen
        self.patience = 5
        self.missing_upper_left = self.patience
        self.missing_lower_left = self.patience
        self.missing_upper_right = self.patience
        self.missing_lower_right = self.patience

    def process(self, frame):
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
        results = self.model.process(frame_rgb)
        
        found_upper_left = False
        found_lower_left = False
        found_upper_right = False
        found_lower_right = False
        
        half_width = self.config.resolution_width / 2
        half_height = self.config.resolution_height / 2
        
        # Detection und Landmark-Verarbeitung
        if results.multi_hand_landmarks:
            for landmarks in results.multi_hand_landmarks:
                landmark = self.utils.point_to_pixel_coordinates(Point2D(landmarks.landmark[9].x, landmarks.landmark[9].y))
                
                if landmark.x < half_width and landmark.y < half_height:
                    found_upper_left = True
                elif landmark.x < half_width and landmark.y >= half_height:
                    found_lower_left = True
                elif landmark.x >= half_width and landmark.y < half_height:
                    found_upper_right = True
                elif landmark.x >= half_width and landmark.y >= half_height:
                    found_lower_right = True

        # Anpassen der Patience
        if found_upper_left: self.missing_upper_left = 0
        else: self.missing_upper_left += 1
            
        if found_lower_left: self.missing_lower_left = 0
        else: self.missing_lower_left += 1

        if found_upper_right: self.missing_upper_right = 0
        else: self.missing_upper_right += 1
            
        if found_lower_right: self.missing_lower_right = 0
        else: self.missing_lower_right += 1

        upper_left = self.missing_upper_left < self.patience
        lower_left = self.missing_lower_left < self.patience
        upper_right = self.missing_upper_right < self.patience
        lower_right = self.missing_lower_right < self.patience
        
        return upper_left, lower_left, upper_right, lower_right