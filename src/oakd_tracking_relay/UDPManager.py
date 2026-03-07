import socket
import orjson
from typing import Dict

from oakd_tracking_relay.ConfigurationManager import Configuration
from oakd_tracking_relay.TrackingDTO import *

class UDP:
    def __init__(self, config: Configuration):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.config = config
    
    def sendEyes(self, irisL: Point3D, irisR: Point3D, timeStamp: float):
        if irisL.valid() and irisR.valid(): 
            eyes = {
                "ts": timeStamp,
                "eyes": {
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
            self.sendInternal(eyes, self.config.udp_port_eyes)

    def sendHands(self, hand_ol: bool, hand_ul: bool, hand_or: bool, hand_ur: bool, timeStamp: float):
            hands = {"ts": timeStamp, "hands": {}}
            hands["hands"]["Left Up"] = hand_ol
            hands["hands"]["Left Down"] = hand_ul
            hands["hands"]["Right Up"] = hand_or
            hands["hands"]["Right Down"] = hand_ur
            
            print(hands)
            self.sendInternal(hands, self.config.udp_port_hands)

    def sendInternal(self, payload: Dict, port: int):
        try:
            json_bytes = orjson.dumps(payload, option=orjson.OPT_SERIALIZE_NUMPY)
            self.sock.sendto(json_bytes, (self.config.udp_ip, port))
        except Exception as e:
            print(f"UDP Error (Target {port}): {e}")

    def close(self):
        self.sock.close()