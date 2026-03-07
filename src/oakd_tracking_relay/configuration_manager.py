import json
import cv2
import os
from dataclasses import dataclass, asdict

from oakd_tracking_relay.camera_manager import OakDPro
from oakd_tracking_relay.tracking_dto import *

@dataclass
class Configuration:
    # Netzwerk
    udp_ip: str = "127.0.0.1"
    udp_port_eyes: int = 5005
    udp_port_hands: int = 5006
    
    # Kamera
    fps: int = 100
    iso: int = 1000
    exposure_us: int = 4000
    apply_clahe: int = 1
    ir_laser_intensity_percent: int = 100
    resolution_width: int = 640
    resolution_height: int = 400
    
    # Tracking
    confidence_percent: int = 75

    def save_to_file(self, filename="config.json"):
        data = asdict(self)
            
        try:
            with open(filename, 'w') as f:
                json.dump(data, f, indent=4)
                print(f"Config in Datei {filename} gespeichert: {data}", flush=True)
        except Exception as e:
            print(f"Konnte Datei {filename} nicht speichern: {e}", flush=True)

    @classmethod
    def load_from_file(cls, filename="config.json"):
        config = cls()
        
        if not os.path.exists(filename):
            print(f"Datei nicht gefunden: {filename}", flush=True)
            raise FileNotFoundError(filename)

        try:
            with open(filename, 'r') as f:
                data = json.load(f)
                
            for key, value in data.items():
                if hasattr(config, key):
                    setattr(config, key, value)
            
            print(f"Config aus Datei {filename} geladen: {data}", flush=True)
            
        except Exception as e:
            print(f"Konnte Datei {filename} nicht laden: {e}", flush=True)
            raise e
            
        return config
    
class ConfigurationUI:
    def __init__(self, camera: OakDPro, config: Configuration):
        self.config = config
        self.camera = camera
        self.window_name = "Preview"
        self.display_frame = None
        self._create_ui_elements()

    # Benötigtes callback für OpenCV Trackbar
    def _nothing(self, x):
        pass

    def _create_ui_elements(self):
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)

        cv2.createTrackbar("Apply CLAHE", self.window_name, self.config.apply_clahe, 1, self._nothing)
        cv2.setTrackbarMin("Apply CLAHE", self.window_name, 0)
        cv2.setTrackbarMax("Apply CLAHE", self.window_name, 1)

        cv2.createTrackbar("ISO", self.window_name, self.config.iso, 1000, self._nothing)
        cv2.setTrackbarMin("ISO", self.window_name, 100)
        cv2.setTrackbarMax("ISO", self.window_name, 1000)

        cv2.createTrackbar("Exposure", self.window_name, self.config.exposure_us, 4000, self._nothing)
        cv2.setTrackbarMin("Exposure", self.window_name, 50)
        cv2.setTrackbarMax("Exposure", self.window_name, 9500)

        # Über 90% kann unerwünschtes Verhalten auftreten
        cv2.createTrackbar("IR Laser", self.window_name, self.config.ir_laser_intensity_percent, 0, self._nothing)
        cv2.setTrackbarMin("IR Laser", self.window_name, 0)
        cv2.setTrackbarMax("IR Laser", self.window_name, 90)

        cv2.createTrackbar("Min. Confidence", self.window_name, self.config.confidence_percent, 75, self._nothing)
        cv2.setTrackbarMin("Min. Confidence", self.window_name, 20)
        cv2.setTrackbarMax("Min. Confidence", self.window_name, 100)

    def check_for_close_key(self):
        return cv2.waitKey(1) & 0xFF == ord("q")

    def exit(self):
        cv2.destroyAllWindows()
        cv2.waitKey(1)  

    def change_display_frame(self, display_frame):
        self.display_frame = cv2.cvtColor(display_frame, cv2.COLOR_GRAY2BGR)

    def show(self):
        if self.display_frame is not None:
            cv2.imshow(self.window_name, self.display_frame)

    def update_config_if_changed(self):
        new_values = {
            "iso": cv2.getTrackbarPos("ISO", self.window_name),
            "exposure_us": cv2.getTrackbarPos("Exposure", self.window_name),
            "apply_clahe": cv2.getTrackbarPos("Apply CLAHE", self.window_name),
            "ir_laser_intensity_percent": cv2.getTrackbarPos("IR Laser", self.window_name),
            "confidence_percent": cv2.getTrackbarPos("Min. Confidence", self.window_name),
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
            left_x, left_y = int(tracking_data.left.aggregated.left.x), int(tracking_data.left.aggregated.left.y)
            right_x, right_y = int(tracking_data.right.aggregated.left.x), int(tracking_data.right.aggregated.left.y)
            if self.display_frame is not None:
                cv2.circle(self.display_frame, (left_x, left_y), 5, (0, 255, 255), -1)
                cv2.circle(self.display_frame, (right_x, right_y), 5, (0, 255, 0), -1)

    def draw_data_rate(self, data_rate: float):
        if self.display_frame is not None:
            cv2.putText(self.display_frame, f"FPS: {int(data_rate)}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
