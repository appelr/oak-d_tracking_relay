import socket
import orjson

from typing import Dict

from .ConfigurationManager import Configuration
from .TrackingDTO import *

class UDP:
    def __init__(self, config: Configuration):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.config = config
    
    def send(self, irisL: Point3D, irisR: Point3D, handL: Point3D, handR: Point3D, timeStamp: float):
        if irisL.valid() and irisR.valid(): 
            head = {
                "ts": timeStamp,
                "head": {
                    "left_iris": {
                        "x": irisL.x,
                        "y": irisL.y,
                        "z": irisL.z
                    },
                    "right_iris": {
                        "x": irisR.x,
                        "y": irisR.y,
                        "z": irisR.z
                    }
                }
            }
            self.sendInternal(head, self.config.udp_port_head)

        if handL.valid() or handR.valid():
            hands = {"ts": timeStamp, "hands": {}}

            if handL.valid():
                hands["hands"]["left_hand"] = {
                    "x": handL.x,
                    "y": handL.y,
                    "z": handL.z
                }
                
            if handR.valid():
                hands["hands"]["right_hand"] = {
                    "x": handR.x,
                    "y": handR.y,
                    "z": handR.z
                }

            self.sendInternal(hands, self.config.udp_port_hands)

    def sendInternal(self, payload: Dict, port: int):
        try:
            json_bytes = orjson.dumps(payload, option=orjson.OPT_SERIALIZE_NUMPY)
            self.sock.sendto(json_bytes, (self.config.udp_ip, port))
            print(json_bytes, flush=True)
        except Exception as e:
            print(f"UDP Error (Target {port}): {e}")

    def close(self):
        self.sock.close()