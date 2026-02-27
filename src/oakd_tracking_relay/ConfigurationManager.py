from dataclasses import dataclass, asdict
import depthai as dai
import json
import cv2
import os
import sys

from oakd_tracking_relay.TrackingDTO import *


@dataclass
class RuntimeState:
    update_trigger: bool = True
    showPreview: bool = True

@dataclass
class Configuration:
    # Netzwerk
    udp_ip: str = "127.0.0.1"
    udp_port_head: int = 5005
    udp_port_hands: int = 5006
    
    # Kamera
    fps: int = 100
    exposure_us: int = 600
    iso: int = 200
    ir_laser_intensity_percent: int = 100
    resolutionWidth: int = 640
    resolutionHeight: int = 400
    
    # Tracking
    mp_min_detection_percent: int = 40
    mp_min_tracking_percent: int = 20

    def save(self, filename="config.json"):
        data = asdict(self)
            
        try:
            with open(filename, 'w') as f:
                json.dump(data, f, indent=4)
                print(f"Config gespeichert: {data}", flush=True)
        except Exception as e:
            print(f"Fehler beim Speichern: {e}", flush=True)

    @classmethod
    def load(cls, filename="config.json"):
        config = cls()
        
        if not os.path.exists(filename):
            print(f"FEHLER: {filename} fehlt! Bitte Datei erstellen.", flush=True)
            raise FileNotFoundError(f"{filename} fehlt.")

        try:
            with open(filename, 'r') as f:
                data = json.load(f)
                
            for key, value in data.items():
                if hasattr(config, key):
                    setattr(config, key, value)
            
            print(f"Config geladen: {data}", flush=True)
            
        except Exception as e:
            print(f"KRITISCHER FEHLER: Konnte Config nicht lesen: {e}", flush=True)
            raise e
            
        return config
    
class ConfigurationUI:
    def __init__(self, camera, config: Configuration, state: RuntimeState):
        self.config = config
        self.state = state
        self.camera = camera
        self.window_name = "Preview"
        self.displayFrame = None
        self.enabled = True
        self._create()

    # Needed by OpenCV
    def _nothing(self, x):
        pass

    def _create(self):
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)

        cv2.createTrackbar("ISO", self.window_name, self.config.iso, 1000, self._nothing)
        cv2.setTrackbarMin("ISO", self.window_name, 400)

        cv2.createTrackbar("Exposure", self.window_name, self.config.exposure_us, 100, self._nothing)
        cv2.setTrackbarMin("Exposure", self.window_name, 10)

        cv2.createTrackbar("IR Laser", self.window_name, self.config.ir_laser_intensity_percent, 100, self._nothing)
        cv2.createTrackbar("Min. Detection", self.window_name, self.config.mp_min_detection_percent, 100, self._nothing)
        cv2.createTrackbar("Min. Tracking", self.window_name, self.config.mp_min_tracking_percent, 100, self._nothing)

    def shouldExit(self):
        return cv2.waitKey(1) & 0xFF == ord("q")

    def exit(self):
        cv2.destroyAllWindows()
        cv2.waitKey(1)  

    def setDisplayFrame(self, frame):
        self.displayFrame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

    def show(self):
        if self.displayFrame is not None:
            cv2.imshow(self.window_name, self.displayFrame)

    def updateConfigIfChanged(self):
        if not self.enabled:
            return

        new_values = {
            "iso": cv2.getTrackbarPos("ISO", self.window_name),
            "exposure_us": cv2.getTrackbarPos("Exposure", self.window_name),
            "ir_laser_intensity_percent": cv2.getTrackbarPos("IR Laser", self.window_name),
            "mp_min_detection_percent": cv2.getTrackbarPos("Min. Detection", self.window_name),
            "mp_min_tracking_percent": cv2.getTrackbarPos("Min. Tracking", self.window_name),
        }

        changed = False
        for k, v in new_values.items():
            if getattr(self.config, k) != v:
                setattr(self.config, k, v)
                changed = True

        if changed:
            self.state.update_trigger = True
            self.camera._updateSettings()
            self.state.update_trigger = False
            self.config.save()

    def drawTrackingData(self, trackingData: TrackingData, dataRate):
        if trackingData.valid():
            leftPointX, leftPointY = int(trackingData.left.aggregated.left.x), int(trackingData.left.aggregated.left.y)
            rightPointX, rightPointY = int(trackingData.right.aggregated.left.x), int(trackingData.right.aggregated.left.y)
            if self.displayFrame is not None:
                cv2.circle(self.displayFrame, (leftPointX, leftPointY), 5, (0, 255, 255), -1)
                cv2.circle(self.displayFrame, (rightPointX, rightPointY), 5, (0, 255, 0), -1)
                cv2.putText(self.displayFrame, f"FPS: {int(dataRate)}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        