import cv2

from oakd_tracking_relay.configuration_manager import Configuration
from oakd_tracking_relay.camera_manager import OakDPro
from oakd_tracking_relay.tracking_dto import *


class ConfigurationUI:
    def __init__(self, camera: OakDPro, config: Configuration):
        self.config = config
        self.camera = camera
        self.display_frame = None

        self.WINDOW_NAME = "Preview"
        self.TRACKBAR_APPLY_CLAHE = "Apply CLAHE"
        self.TRACKBAR_ISO = "ISO"
        self.TRACKBAR_EXPOSURE = "Exposure"
        self.TRACKBAR_IR_LASER = "IR Laser"
        self.TRACKBAR_CONFIDENCE = "Min. Confidence"
        
        self._create_ui_elements()

       

    # Benötigtes callback für OpenCV Trackbar
    def _nothing(self, x):
        pass

    def _create_ui_elements(self):

        cv2.namedWindow(self.WINDOW_NAME, cv2.WINDOW_NORMAL)

        cv2.createTrackbar(self.TRACKBAR_APPLY_CLAHE, self.WINDOW_NAME, self.config.apply_clahe, 1, self._nothing)
        cv2.setTrackbarMin(self.TRACKBAR_APPLY_CLAHE, self.WINDOW_NAME, 0)
        cv2.setTrackbarMax(self.TRACKBAR_APPLY_CLAHE, self.WINDOW_NAME, 1)

        cv2.createTrackbar(self.TRACKBAR_ISO, self.WINDOW_NAME, self.config.iso, 1000, self._nothing)
        cv2.setTrackbarMin(self.TRACKBAR_ISO, self.WINDOW_NAME, 100)
        cv2.setTrackbarMax(self.TRACKBAR_ISO, self.WINDOW_NAME, 1000)

        cv2.createTrackbar(self.TRACKBAR_EXPOSURE, self.WINDOW_NAME, self.config.exposure_us, 4000, self._nothing)
        cv2.setTrackbarMin(self.TRACKBAR_EXPOSURE, self.WINDOW_NAME, 50)
        cv2.setTrackbarMax(self.TRACKBAR_EXPOSURE, self.WINDOW_NAME, 9500)

        # Über 90% kann unerwünschtes Verhalten auftreten
        cv2.createTrackbar(self.TRACKBAR_IR_LASER, self.WINDOW_NAME, self.config.ir_intensity_percent, 0, self._nothing)
        cv2.setTrackbarMin(self.TRACKBAR_IR_LASER, self.WINDOW_NAME, 0)
        cv2.setTrackbarMax(self.TRACKBAR_IR_LASER, self.WINDOW_NAME, 90)

        cv2.createTrackbar(self.TRACKBAR_CONFIDENCE, self.WINDOW_NAME, self.config.confidence_percent, 75, self._nothing)
        cv2.setTrackbarMin(self.TRACKBAR_CONFIDENCE, self.WINDOW_NAME, 20)
        cv2.setTrackbarMax(self.TRACKBAR_CONFIDENCE, self.WINDOW_NAME, 100)

    def check_for_close_key(self):
        return cv2.waitKey(1) & 0xFF == ord("q")

    def exit(self):
        cv2.destroyAllWindows()
        cv2.waitKey(1)  

    def change_display_frame(self, display_frame):
        self.display_frame = cv2.cvtColor(display_frame, cv2.COLOR_GRAY2BGR)

    def show(self):
        if self.display_frame is not None:
            cv2.imshow(self.WINDOW_NAME, self.display_frame)

    def update_config_if_changed(self):
        new_values = {
            "iso": cv2.getTrackbarPos(self.TRACKBAR_ISO, self.WINDOW_NAME),
            "exposure_us": cv2.getTrackbarPos(self.TRACKBAR_EXPOSURE, self.WINDOW_NAME),
            "apply_clahe": cv2.getTrackbarPos(self.TRACKBAR_APPLY_CLAHE, self.WINDOW_NAME),
            "ir_laser_intensity_percent": cv2.getTrackbarPos(self.TRACKBAR_IR_LASER, self.WINDOW_NAME),
            "confidence_percent": cv2.getTrackbarPos(self.TRACKBAR_CONFIDENCE, self.WINDOW_NAME),
        }

        changed = False

        for k, v in new_values.items():
            if getattr(self.config, k) != v:
                setattr(self.config, k, v)
                changed = True

        if changed:
            self.camera.update_settings()
            self.config.save_to_file()

    def draw_hand_quadrants(self, upper_left, lower_left, upper_right, lower_right):
        
        if self.display_frame is not None:
            overlay = self.display_frame.copy()

            half_w, half_h = int(self.config.resolution_width / 2), int(self.config.resolution_height / 2)

            if upper_left: 
                cv2.rectangle(overlay, (0, 0), (half_w, half_h), (0, 255, 0), -1)   # Grün
            if lower_left: 
                cv2.rectangle(overlay, (0, half_h), (half_w, self.config.resolution_height), (255, 0, 0), -1)   # Blau
            if upper_right: 
                cv2.rectangle(overlay, (half_w, 0), (self.config.resolution_width, half_h), (0, 0, 255), -1)   # Rot
            if lower_right: 
                cv2.rectangle(overlay, (half_w, half_h), (self.config.resolution_width, self.config.resolution_height), (0, 255, 255), -1) # Gelb

            cv2.addWeighted(overlay, 0.3, self.display_frame, 0.7, 0, self.display_frame)

    def draw_eye_landmarks(self, tracking_data: TrackingData):
        if tracking_data.valid():
            left_x, left_y = int(tracking_data.left_eye.iris.left_cam.x), int(tracking_data.left_eye.iris.left_cam.y)
            right_x, right_y = int(tracking_data.right_eye.iris.left_cam.x), int(tracking_data.right_eye.iris.left_cam.y)
            if self.display_frame is not None:
                cv2.circle(self.display_frame, (left_x, left_y), 5, (0, 255, 255), -1)
                cv2.circle(self.display_frame, (right_x, right_y), 5, (0, 255, 0), -1)

    def draw_info(self):
        if self.display_frame is not None:
            cv2.putText(self.display_frame, f"'Q' druecken, um Highspeed Uebertragung zu starten", (10, 380), cv2.FONT_HERSHEY_DUPLEX, 0.7, (0, 127, 255), 2)
