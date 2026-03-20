import cv2
import time
import numpy as np
import depthai as dai

from oakd_tracking_relay.tracking_dto import *

class ProcessingUtils:
    def __init__(self, camera, config):
        self.camera = camera
        self.config = config

        calibration_data = self.camera.device.readCalibration()
        
        # Intrinische Daten abfragen
        self.intrinsics_left = np.array(calibration_data.getCameraIntrinsics(dai.CameraBoardSocket.CAM_B, self.config.resolution_width, self.config.resolution_height))
        self.intrinsics_right = np.array(calibration_data.getCameraIntrinsics(dai.CameraBoardSocket.CAM_C,self.config.resolution_width, self.config.resolution_height))

        # Verzerrung abfragen
        self.distortion_left = np.array(calibration_data.getDistortionCoefficients(dai.CameraBoardSocket.CAM_B))
        self.distortion_right = np.array(calibration_data.getDistortionCoefficients(dai.CameraBoardSocket.CAM_C))

        # Brennweite und Bildmitte
        self.focal_length_x = self.intrinsics_left[0,0]
        self.focal_length_y = self.intrinsics_left[1,1]
        self.center_x = self.intrinsics_left[0,2]
        self.center_y = self.intrinsics_left[1,2]

        # Extrinsische Daten der Kamera abfragen
        extrinsics = np.array(calibration_data.getCameraExtrinsics(dai.CameraBoardSocket.CAM_B, dai.CameraBoardSocket.CAM_C))
        
        self.rotation = extrinsics[:3, :3]
        self.translation = extrinsics[:3, 3].reshape(3,1) * 10

        self.baseline = calibration_data.getBaselineDistance() * 10

        image_size = (self.config.resolution_width, self.config.resolution_height)

        # Rektifizierungsmatritzen berechnen
        self.rotation_left, self.rotation_right, self.projection_left, self.projection_right, _, _, _ = cv2.stereoRectify(
            self.intrinsics_left, self.distortion_left,
            self.intrinsics_right, self.distortion_right,
            imageSize=image_size,
            R=self.rotation,
            T=self.translation,
            flags=cv2.CALIB_ZERO_DISPARITY,
            alpha=0
        )

        self.map_left_x, self.map_left_y = cv2.initUndistortRectifyMap(
            self.intrinsics_left, self.distortion_left, 
            self.rotation_left, self.projection_left,
            image_size, 
            cv2.CV_16SC2
        )
        self.map_right_x, self.map_right_y = cv2.initUndistortRectifyMap(
            self.intrinsics_right, self.distortion_right, 
            self.rotation_right, self.projection_right,
            image_size, 
            cv2.CV_16SC2
        )

        self.clahe = cv2.createCLAHE(
                    clipLimit=2.0,
                    tileGridSize=(8, 8)
                )

        # Data rate calculation
        self.prev_frame_time = 0.0
        self.smoothed_dt = 0.0

        self.CROP_SCALE_NO_FACE = 0.6
        self.CROP_SCALE_FACE = 0.4

    def stereo_point_to_pixel_coordinates(self, stereo_point: StereoPoint) -> StereoPoint: 
        left = self.point_to_pixel_coordinates(point=stereo_point.left)
        right = self.point_to_pixel_coordinates(point=stereo_point.right)

        return StereoPoint(left, right)

    def point_to_pixel_coordinates(self, point: Point2D) -> Point2D:
        x = float(point.x * self.config.resolution_width)
        y = float(point.y * self.config.resolution_height)

        return Point2D(x, y)

    def rectify_stereo_frame(self, frame_left, frame_right):
        left = cv2.remap(frame_left, self.map_left_x, self.map_left_y, cv2.INTER_LINEAR)
        right = cv2.remap(frame_right, self.map_right_x, self.map_right_y, cv2.INTER_LINEAR)
        
        return left, right
    
    def triangulate_stereo_point(self, stereo_point: StereoPoint) -> Point3D:
        left = stereo_point.left.as_np()
        right = stereo_point.right.as_np()

        points_4D = cv2.triangulatePoints(self.projection_left, self.projection_right, left, right)
        points_3D = points_4D[:3] / points_4D[3:]

        x = points_3D[0].item()
        y = points_3D[1].item()
        z = points_3D[2].item()

        return Point3D(x, y, z)
    
    def get_data_rate(self):
        current_frame_time = time.perf_counter()
        if self.prev_frame_time != 0:
            dt = current_frame_time - self.prev_frame_time # Verstrichene Zeit für diesen Frame
            
            if self.smoothed_dt == 0.0:
                self.smoothed_dt = dt
            else:
                self.smoothed_dt = (self.smoothed_dt * 0.90) + (dt * 0.10)
            
            data_rate = 1.0 / self.smoothed_dt 
        else:
            data_rate = 0.0
            
        self.prev_frame_time = current_frame_time
        return data_rate
    
    def get_best_face(self, results_left, results_right):
        scores = []

        for i, (face_left, face_right) in enumerate(zip(results_left.multi_face_landmarks,
                                        results_right.multi_face_landmarks)):
            score_left = self._calculate_face_score(face_landmarks=face_left)
            score_right = self._calculate_face_score(face_landmarks=face_right)

            # Stereo-Score = Mittelwert
            scores.append((i, (score_left + score_right) / 2))

        best_face = max(scores, key=lambda x: x[1])[0]
        return best_face

    def _calculate_face_score(self, face_landmarks, area_weight=0.5, center_weight=0.5):
        coordinates_x = np.array([lm.x for lm in face_landmarks.landmark])
        coordinates_y = np.array([lm.y for lm in face_landmarks.landmark])

        min_x, max_x = coordinates_x.min(), coordinates_x.max()
        min_y, max_y = coordinates_y.min(), coordinates_y.max()

        # Bounding-Box Größe
        area = (max_x - min_x) * (max_y - min_y)

        # Distanz zur Bildmitte
        center_distance_x = (min_x + max_x) / 2
        center_distance_y = (min_y + max_y) / 2

        center_dist = np.hypot(center_distance_x - 0.5, center_distance_y - 0.5)

        # Kombinierter Score (mit Gewichten)
        score = area_weight * area - center_weight * center_dist
        return score
    
    def crop_frame(self, frame, center: Point2D):
        height, width = frame.shape[:2]

        # Weniger Crop, wenn kein Gesicht gefunden wurde
        if not center.valid():
            scale = self.CROP_SCALE_NO_FACE
            center_x = width // 2
            center_y = height // 2
        else:
            scale = self.CROP_SCALE_FACE
            center_x = int(center.x)
            center_y = int(center.y)

        crop_height, crop_width = int(height * scale), int(width * scale)

        # Ursprung links oben
        origin_x = center_x - crop_width // 2
        origin_y = center_y - crop_height // 2

        # Clamping
        origin_x = max(0, min(origin_x, width - crop_width))
        origin_y = max(0, min(origin_y, height - crop_height))

        return frame[origin_y:origin_y+crop_height, origin_x:origin_x+crop_width], origin_x, origin_y
    
    def apply_clahe_to_stereo_frame(self, frame_left, frame_right):
        frame_left = self.clahe.apply(frame_left)
        frame_right = self.clahe.apply(frame_right)

        return frame_left, frame_right
    