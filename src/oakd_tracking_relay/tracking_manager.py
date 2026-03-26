import cv2
import numpy as np

import mediapipe as mp
from enum import Enum, auto

from oakd_tracking_relay.utils import ProcessingUtils
from oakd_tracking_relay.configuration_manager import Configuration
from oakd_tracking_relay.tracking_dto import *

class TrackerState(Enum):
    DETECTION = auto()
    TRACKING = auto()

class IrisTracker:
    def __init__(self, utils: ProcessingUtils, config: Configuration):
        self.utils = utils
        self.config = config
        self.current_state = TrackerState.DETECTION
        self.current_tracking_data = TrackingData()
        
        # MediaPipe FaceMesh 
        self.model = mp.solutions.face_mesh.FaceMesh( # type: ignore
            max_num_faces=4, 
            refine_landmarks=True, # Benötigt für Iris-Koordinaten
            static_image_mode=True, # Tracking-Infos ignorieren, da wir Optical Flow nutzen
            min_detection_confidence=float(config.confidence_percent)/100
        )
        
        # Schwellwerte für Wechsel Tracking -> Detection
        self.tracking_confidence_counter = 0
        self.tracking_confidence_init = 15
        self.tracking_confidence_minimum = 0

        # Schwellwerte für Wechsel Detection -> Tracking
        self.detection_buffer: list[TrackingData] = []        
        self.DETECTION_BUFFER_MAX_SIZE = 2

        # Counter für Mediapipe Recheck
        self.frame_count = 0
        self.RECHECK_INTERVAL = 15

        # Optical Flow
        self.previous_frame_left = None
        self.previous_frame_right = None
        self.OPTICAL_FLOW_PARAMS = dict(
            winSize=(8, 8), 
            maxLevel=5,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.03)
        )

        # Schwellwerte für Sprünge zwischen Mediapipe und OpticalFlow
        self.SEARCH_STABILITY_THRESHOLD = 20
        self.MAX_DIFF_OPTICALFLOW_MEDIAPIPE = 1.5
        self.MAX_EYE_JUMP_BETWEEN_FRAMES = 35
        self.MAX_DISPARITY_BETWEEN_FRAMES = 10
        self.MAX_EYE_DISTANCE_DIFFERENCE_BETWEEN_FRAMES = 20

        self.LEFT_IRIS_INDICES = [468, 469, 470, 471, 472]
        self.RIGHT_IRIS_INDICES = [473, 474, 475, 476, 477]

    def process_stereo_frame(self, frame_left, frame_right):
        self.frame_count += 1

        if self.current_state == TrackerState.DETECTION:
            self._detect(frame_left=frame_left, frame_right=frame_right)
        elif self.current_state == TrackerState.TRACKING:
            self._track(frame_left=frame_left, frame_right=frame_right)
    
    def _detect(self, frame_left, frame_right):
        detected_data = self._run_detection(frame_left=frame_left, frame_right=frame_right)

        if detected_data.valid():
            self.detection_buffer.append(detected_data)
            
            # Prüfen, ob genug Detection-Daten gesammelt
            if len(self.detection_buffer) == self.DETECTION_BUFFER_MAX_SIZE:
                # Abweichung innerhalb der letzten x Detection-Daten prüfen
                if self._is_detection_stable():
                    # Zustandswechsel zu TRACKING
                    self._switch_to_tracking(frame_left=frame_left, frame_right=frame_right)
                else:
                    self._reset_detection_buffer()
        else:
            self._reset_detection_buffer()

    def _run_detection(self, frame_left, frame_right) -> TrackingData:
        center_left = Point2D()
        center_right = Point2D()

        # Bildzentrum für crop auf letzte valide TrackingData setzen
        if self.current_tracking_data.valid():
            center_left = self.current_tracking_data.center_between_eyes.left_cam
            center_right = self.current_tracking_data.center_between_eyes.right_cam

        crop_left, offset_x_left, offset_y_left = self.utils.crop_frame(frame=frame_left, center=center_left)
        crop_right, offset_x_right, offset_y_right = self.utils.crop_frame(frame=frame_right, center=center_right)

        crop_left_rgb = cv2.cvtColor(crop_left, cv2.COLOR_GRAY2RGB)
        crop_right_rgb = cv2.cvtColor(crop_right, cv2.COLOR_GRAY2RGB)

        results_left = self.model.process(crop_left_rgb)
        results_right = self.model.process(crop_right_rgb)

        data = TrackingData()

        # Augen-Landmark Detection
        if results_left.multi_face_landmarks and results_right.multi_face_landmarks:
            best_face_id = self.utils.get_best_face(results_left=results_left, results_right=results_right)
            landmarks_left = results_left.multi_face_landmarks[best_face_id]
            landmarks_right = results_right.multi_face_landmarks[best_face_id]

            for id in self.LEFT_IRIS_INDICES:
                left  = Point2D(offset_x_left + landmarks_left.landmark[id].x * crop_left.shape[1], offset_y_left + landmarks_left.landmark[id].y * crop_left.shape[0])
                right = Point2D(offset_x_right + landmarks_right.landmark[id].x * crop_right.shape[1], offset_y_right + landmarks_right.landmark[id].y * crop_right.shape[0])
                data.left_eye.stereo_points.append(StereoPoint(left, right))

            for id in self.RIGHT_IRIS_INDICES:
                left  = Point2D(offset_x_left + landmarks_left.landmark[id].x * crop_left.shape[1], offset_y_left + landmarks_left.landmark[id].y * crop_left.shape[0])
                right = Point2D(offset_x_right + landmarks_right.landmark[id].x * crop_right.shape[1], offset_y_right + landmarks_right.landmark[id].y * crop_right.shape[0])
                data.right_eye.stereo_points.append(StereoPoint(left, right))
            
            # Zentrum der gefundenen Punkte bestimmen (Iris)
            if data.valid():
                data.aggregate_median()

        return data

    def _is_detection_stable(self):
        first_left = self.detection_buffer[0].center_between_eyes.left_cam
        last_left = self.detection_buffer[-1].center_between_eyes.left_cam
        first_right = self.detection_buffer[0].center_between_eyes.right_cam
        last_right = self.detection_buffer[-1].center_between_eyes.right_cam
        return np.hypot(first_left.x - last_left.x, first_left.y - last_left.y) < self.SEARCH_STABILITY_THRESHOLD and np.hypot(first_right.x - last_right.x, first_right.y - last_right.y) < self.SEARCH_STABILITY_THRESHOLD
    
    def _reset_detection_buffer(self):
        self.detection_buffer.clear()

    def _update_previous_frames(self, frame_left, frame_right):
        self.previous_frame_left = frame_left.copy()
        self.previous_frame_right = frame_right.copy()

    def _switch_to_tracking(self, frame_left, frame_right):
        self.current_tracking_data = self.detection_buffer[-1]
        self.tracking_confidence_counter = self.tracking_confidence_init
        self._reset_detection_buffer()
        self.current_state = TrackerState.TRACKING
        self._update_previous_frames(frame_left=frame_left, frame_right=frame_right)


    def _track(self, frame_left, frame_right): 
        # Optical Flow auf Tracking Daten anwenden
        left_points_left_cam = np.array([p.left_cam.as_np() for p in self.current_tracking_data.left_eye.stereo_points], dtype=np.float32).reshape(-1, 1, 2)
        left_points_right_cam = np.array([p.right_cam.as_np() for p in self.current_tracking_data.left_eye.stereo_points], dtype=np.float32).reshape(-1, 1, 2)
        right_points_left_cam = np.array([p.left_cam.as_np() for p in self.current_tracking_data.right_eye.stereo_points], dtype=np.float32).reshape(-1, 1, 2)
        right_points_right_cam = np.array([p.right_cam.as_np() for p in self.current_tracking_data.right_eye.stereo_points], dtype=np.float32).reshape(-1, 1, 2)

        next_left_points_left_cam, left_status_left_cam, _ = cv2.calcOpticalFlowPyrLK(self.previous_frame_left, frame_left, left_points_left_cam, None, **self.OPTICAL_FLOW_PARAMS) # type: ignore

        next_left_points_right_cam, left_status_right_cam, _ = cv2.calcOpticalFlowPyrLK(self.previous_frame_right, frame_right, left_points_right_cam, None, **self.OPTICAL_FLOW_PARAMS) # type: ignore

        next_right_points_left_cam, right_status_left_cam, _ = cv2.calcOpticalFlowPyrLK(self.previous_frame_left, frame_left, right_points_left_cam, None, **self.OPTICAL_FLOW_PARAMS) # type: ignore

        next_right_points_right_cam, right_status_right_cam, _ = cv2.calcOpticalFlowPyrLK(self.previous_frame_right, frame_right, right_points_right_cam, None, **self.OPTICAL_FLOW_PARAMS) # type: ignore

        data = TrackingData()

        # DTO erstellen und befüllen
        for i in range(len(left_points_left_cam)):
            if left_status_left_cam[i][0] == 1 and left_status_right_cam[i][0] == 1:
                next_left_point_left_cam = Point2D(float(next_left_points_left_cam[i][0][0]), float(next_left_points_left_cam[i][0][1]))
                netx_left_point_right_cam = Point2D(float(next_left_points_right_cam[i][0][0]), float(next_left_points_right_cam[i][0][1]))
                data.left_eye.stereo_points.append(StereoPoint(next_left_point_left_cam, netx_left_point_right_cam))

        for i in range(len(right_points_left_cam)):
            if right_status_left_cam[i][0] == 1 and right_status_right_cam[i][0] == 1:
                next_right_point_left_cam = Point2D(float(next_right_points_left_cam[i][0][0]), float(next_right_points_left_cam[i][0][1]))
                next_right_point_right_cam = Point2D(float(next_right_points_right_cam[i][0][0]), float(next_right_points_right_cam[i][0][1]))
                data.right_eye.stereo_points.append(StereoPoint(next_right_point_left_cam, next_right_point_right_cam))

        # Zentrum der getrackten Punkte bestimmen (Iris)
        if data.valid():
            data.aggregate_median()
        else:
            self._decrease_confidence(amount=2)
            return

        # Plausibilitätscheck zwischen 2 aufeinanderfolgenden Frames
        if not self._is_movement_plausible(self.current_tracking_data, data):
            self._decrease_confidence()
            return
        
        # Tracking Daten akzeptieren
        self._update_tracking_data(new_tracking_data=data, frame_left=frame_left, frame_right=frame_right)
    
        # Periodischer Mediapipe Recheck
        if self._should_recheck():
            self._recheck(frame_left=frame_left, frame_right=frame_right)

    def _recheck(self, frame_left, frame_right):
        recheck_data = self._run_detection(frame_left=frame_left, frame_right=frame_right)
        if recheck_data.valid():
                self.detection_buffer.append(recheck_data)

                # Prüfen, ob genug Detection-Daten gesammelt
                if len(self.detection_buffer) == self.DETECTION_BUFFER_MAX_SIZE:
                    # Abweichung innerhalb der letzten x Detection-Daten prüfen
                    if self._is_detection_stable():
                        # Abweichtung zwischen Optical Flow und Mediapipe prüfen
                        if self._is_tracking_deviating_from_detection(recheck_data):
                            # Tracking-Daten mit Detection-Daten überschreiben
                            self.current_tracking_data = recheck_data
                            self._update_previous_frames(frame_left=frame_left, frame_right=frame_right)
                   
                    self._reset_detection_buffer()
        else:
            self._reset_detection_buffer()

    def _is_tracking_deviating_from_detection(self, recheck_data: TrackingData):
        tracking_point_left = self.current_tracking_data.center_between_eyes.left_cam
        tracking_point_right = self.current_tracking_data.center_between_eyes.right_cam

        recheck_data_left = recheck_data.center_between_eyes.left_cam
        recheck_data_right = recheck_data.center_between_eyes.right_cam

        distance_left = np.hypot(
            recheck_data_left.x - tracking_point_left.x,
            recheck_data_left.y - tracking_point_left.y
        )
        distance_right = np.hypot(
            recheck_data_right.x - tracking_point_right.x,
            recheck_data_right.y - tracking_point_right.y
        )
        
        return distance_left > self.MAX_DIFF_OPTICALFLOW_MEDIAPIPE or distance_right > self.MAX_DIFF_OPTICALFLOW_MEDIAPIPE
    
    def _is_movement_plausible(self, previous_data, new_data):
        jump_l, jump_r = self._get_eye_jumps_between_frames(previous_data, new_data)
        disparity = self._get_disparity_between_frames(previous_data, new_data)
        eye_dist = self._get_eye_distance_between_frames(previous_data, new_data)

        return (
            jump_l <= self.MAX_EYE_JUMP_BETWEEN_FRAMES and
            jump_r <= self.MAX_EYE_JUMP_BETWEEN_FRAMES and
            disparity <= self.MAX_DISPARITY_BETWEEN_FRAMES and
            eye_dist <= self.MAX_EYE_DISTANCE_DIFFERENCE_BETWEEN_FRAMES
        )
    
    def _get_eye_jumps_between_frames(self, previous_data: TrackingData, new_data: TrackingData):
        distance_x_left_eye = new_data.left_eye.iris.left_cam.x - previous_data.left_eye.iris.left_cam.x
        distance_y_left_eye = new_data.left_eye.iris.left_cam.y - previous_data.left_eye.iris.left_cam.y
        distance_x_right_eye = new_data.right_eye.iris.left_cam.x - previous_data.right_eye.iris.left_cam.x
        distance_y_right_eye = new_data.right_eye.iris.left_cam.y - previous_data.right_eye.iris.left_cam.y

        return np.hypot(distance_x_left_eye, distance_y_left_eye), np.hypot(distance_x_right_eye, distance_y_right_eye)

    def _get_disparity_between_frames(self, previous_data: TrackingData, new_data: TrackingData):
        previous_disparity = previous_data.center_between_eyes.left_cam.x - previous_data.center_between_eyes.right_cam.x
        current_disparity = new_data.center_between_eyes.left_cam.x - new_data.center_between_eyes.right_cam.x

        return abs(current_disparity - previous_disparity)
    
    def _get_eye_distance_between_frames(self, previous_data: TrackingData, new_data: TrackingData):
        previous_eye_distance = np.hypot(previous_data.left_eye.iris.left_cam.x - previous_data.right_eye.iris.left_cam.x, previous_data.left_eye.iris.left_cam.y - previous_data.right_eye.iris.left_cam.y)

        current_eye_distance = np.hypot(new_data.left_eye.iris.left_cam.x - new_data.right_eye.iris.left_cam.x, new_data.left_eye.iris.left_cam.y - new_data.right_eye.iris.left_cam.y)

        return abs(previous_eye_distance - current_eye_distance)

    def _should_recheck(self):
        return (
            self.frame_count % self.RECHECK_INTERVAL == 0
            or self.detection_buffer
        )
    
    def _decrease_confidence(self, amount=1):
        self.tracking_confidence_counter -= amount
        if self.tracking_confidence_counter <= self.tracking_confidence_minimum:
            self.current_tracking_data = TrackingData()
            self._reset_detection_buffer()
            self.current_state = TrackerState.DETECTION
            
    def _update_tracking_data(self, new_tracking_data, frame_left, frame_right):
            self.current_tracking_data = new_tracking_data
            self._update_previous_frames(frame_left=frame_left, frame_right=frame_right)
            self.tracking_confidence_counter = min(self.tracking_confidence_counter + 1, self.tracking_confidence_init)

class HandTracker:
    def __init__(self, utils: ProcessingUtils, config: Configuration):
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
                landmark = self.utils.point_to_pixel_coordinates(point=Point2D(landmarks.landmark[9].x, landmarks.landmark[9].y))
                
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