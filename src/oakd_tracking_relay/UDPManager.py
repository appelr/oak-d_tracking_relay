import socket
import orjson

from typing import Dict

from .ConfigurationManager import Configuration

class UDP:
    def __init__(self, config: Configuration):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.config = config
    
    def send(self, headData: Dict, handData: Dict, timeStamp: float):
        if headData: 
            head_packet = {
                "ts": timeStamp,
                "head": headData
            }
            self.sendInternal(head_packet, self.config.udp_port_head)

        if handData:
            hands_packet = {"ts": timeStamp}
            if "left_hand" in handData:
                hands_packet["left_hand"] = handData["left_hand"]
                
            if "right_hand" in handData:
                hands_packet["right_hand"] = handData["right_hand"]

            self.sendInternal(hands_packet, self.config.udp_port_hands)
        

    def sendInternal(self, payload: Dict, port: int):
        try:
            json_bytes = orjson.dumps(payload, option=orjson.OPT_SERIALIZE_NUMPY)
            self.sock.sendto(json_bytes, (self.config.udp_ip, port))
            print(json_bytes, flush=True)
        except Exception as e:
            print(f"UDP Error (Target {port}): {e}")

    def close(self):
        self.sock.close()