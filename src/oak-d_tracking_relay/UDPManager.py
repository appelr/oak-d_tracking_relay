import socket
import orjson

from typing import Dict

from .ConfigurationManager import ConfigurationManager

class UDPManager:
    def __init__(self, config: ConfigurationManager):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.config = config
    
    def send(self, trackingData: Dict):
        # Zeitstempel extrahieren, damit beide Pakete synchronisierbar bleiben
        ts = trackingData.get("ts", 0)

        head = trackingData.get("head", {})
        hands = trackingData.get("hands", {})

        if head: 
            head_packet = {
                "ts": ts,
                "head": trackingData["head"]
            }
            self.sendInternal(head_packet, self.config.udp_port_head)

        if hands:
            hands_packet = {"ts": ts}
            if "left_hand" in trackingData:
                hands_packet["left_hand"] = trackingData["left_hand"]
                
            if "right_hand" in trackingData:
                hands_packet["right_hand"] = trackingData["right_hand"]

            self.sendInternal(hands_packet, self.config.udp_port_hands)
        

    def sendInternal(self, payload: Dict, port: int):
        try:
            json_bytes = orjson.dumps(payload, option=orjson.OPT_SERIALIZE_NUMPY)
            self.sock.sendto(json_bytes, (self.config.udp_ip, port))
        except Exception as e:
            print(f"UDP Error (Target {port}): {e}")

    def close(self):
        self.sock.close()