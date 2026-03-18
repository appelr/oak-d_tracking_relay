import socket
import orjson
from typing import Dict

from oakd_tracking_relay.configuration_manager import Configuration
from oakd_tracking_relay.tracking_dto import *

class UDPSender:
    def __init__(self, config: Configuration):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.config = config
    
    def send_eyes(self, iris_left: Point3D, iris_right: Point3D, timestamp: float):
        if iris_left.valid() and iris_right.valid(): 
            data = {
                "ts": timestamp,
                "eyes": {
                    "left_iris": {
                        "x": iris_left.x,
                        "y": iris_left.y,
                        "z": iris_left.z
                    },
                    "right_iris": {
                        "x": iris_right.x,
                        "y": iris_right.y,
                        "z": iris_right.z
                    }
                }
            }
            self._send_data(data=data, port=self.config.udp_port_eyes)

    def send_hands(self, upper_left: bool, lower_left: bool, upper_right: bool, lower_right: bool, timestamp: float):
            
            # Vertauschen von links und rechts, um Unity Welt zu entsprechen 
            data = {"ts": timestamp, "hands": {}}
            data["hands"]["Left Up"] = upper_right
            data["hands"]["Left Down"] = lower_right
            data["hands"]["Right Up"] = upper_left
            data["hands"]["Right Down"] = lower_left
            
            self._send_data(data=data, port=self.config.udp_port_hands)

    def _send_data(self, data: Dict, port: int):
        try:
            json_bytes = orjson.dumps(data, option=orjson.OPT_SERIALIZE_NUMPY)
            self.sock.sendto(json_bytes, (self.config.udp_ip, port))
        except Exception as e:
            print(f"UDP Error (Target {port}): {e}")

    def close(self):
        self.sock.close()