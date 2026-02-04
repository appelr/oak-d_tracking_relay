from dataclasses import dataclass, asdict
import depthai as dai
import json
import os
import sys

@dataclass
class Configuration:
    # Netzwerk
    udp_ip: str = "127.0.0.1"
    udp_port_head: int = 5005
    udp_port_hands: int = 5006
    
    # Kamera
    fps: int = 100
    exposure_us: int = 800
    iso: int = 200
    ir_laser_intensity: float = 1.0
    
    # Tracking Filter
    mp_min_detection: float = 0.4
    mp_min_tracking: float = 0.2

    @classmethod
    def save(cls, filename="config.json"):
        config = cls()
        data = asdict(config)
            
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