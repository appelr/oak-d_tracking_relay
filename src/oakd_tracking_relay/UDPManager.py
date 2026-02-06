import socket
import orjson

from typing import Dict

from .ConfigurationManager import Configuration

class UDP:
    def __init__(self, config: Configuration):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.config = config
    
    def send(self, irisL: Dict, irisR: Dict, handL: Dict, handR: Dict, timeStamp: float):
        if irisL and irisR: 
            head = {
                "ts": timeStamp,
                "head": {
                    "irisL": irisL,
                    "irisR": irisR
                }
            }
            self.sendInternal(head, self.config.udp_port_head)

        if handL or handR:
            hands = {"ts": timeStamp, "hands": {}}
            if handL:
                hands["hands"]["left_hand"] = handL
                
            if handR:
                hands["hands"]["right_hand"] = handR

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