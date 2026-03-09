import json
import os
from dataclasses import dataclass, asdict

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
    