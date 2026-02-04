import cv2
import numpy as np
import mediapipe as mp

from typing import Dict, Tuple

from .ConfigurationManager import Configuration

class TrackingEngine:
    def __init__(self, config: Configuration):
        self.config = config
        self.headTracker = HeadTracker(config)
        self.handTracker = HandTracker(config)

    def processFrame(self, frameL: np.ndarray, frameR: np.ndarray) -> Tuple[Dict, Dict]:
        lRGB = cv2.cvtColor(frameL, cv2.COLOR_GRAY2RGB)
        rRGB = cv2.cvtColor(frameR, cv2.COLOR_GRAY2RGB)
        
        headData = self.headTracker.process(lRGB, rRGB)
        handData = self.handTracker.process(lRGB)

        return headData, handData

class HeadTracker:
    def __init__(self, config: Configuration):
        self.config = config
        self.model = mp.solutions.face_mesh.FaceMesh( 
            max_num_faces=2, 
            refine_landmarks=True,
            min_detection_confidence=config.mp_min_detection, 
            min_tracking_confidence=config.mp_min_tracking)
        

    def process(self, lRGB: np.ndarray, rRGB: np.ndarray) -> Dict:
        headData = {}
        
        lResult = self.model.process(lRGB)
        rResult = self.model.process(rRGB)

        if lResult.multi_face_landmarks:
            lLandmarks = lResult.multi_face_landmarks[0].landmark
            
            lLandmarks_iris_l, lLandmarks_iris_r = lLandmarks[468], lLandmarks[473]
            
            if rResult.multi_face_landmarks:
                rLandmarks = rResult.multi_face_landmarks[0].landmark
                rLandmarks_iris_l, rLandmarks_iris_r = rLandmarks[468], rLandmarks[473]
                
            headData["head"] = {
                "CAM L": {
                    "Iris L": {
                        "x": float(lLandmarks_iris_l.x * -1.0),
                        "y": float(lLandmarks_iris_l.y * -1.0),
                        "z": 0.0
                    },
                    "Iris R": {
                        "x": float(lLandmarks_iris_r.x * -1.0),
                        "y": float(lLandmarks_iris_r.y * -1.0),
                        "z": 0.0
                    }
                },
                "CAM R": {
                    "Iris L": {
                        "x": float(rLandmarks_iris_l.x * -1.0),
                        "y": float(rLandmarks_iris_l.y * -1.0),
                        "z": 0.0
                    },
                    "Iris R": {
                        "x": float(rLandmarks_iris_r.x * -1.0),
                        "y": float(rLandmarks_iris_r.y * -1.0),
                        "z": 0.0
                    }
                }
            }

        return headData

class HandTracker:
    def __init__(self, config: Configuration):
        self.config = config
        self.model = mp.solutions.hands.Hands(
            max_num_hands=2,
            model_complexity=0,
            min_detection_confidence=config.mp_min_detection,
            min_tracking_confidence=config.mp_min_tracking
        )

    def process(self, lRGB: np.ndarray) -> Dict:
        handData = {}

        lResult = self.model.process(lRGB)
        
        if lResult.multi_hand_landmarks:
            for lLandmarks, handedness in zip(lResult.multi_hand_landmarks, lResult.multi_handedness):
                handCenter = lLandmarks.landmark[9]

                # Hände tauschen um Ich-Perspektive zu erzeugen
                handLabel = handedness.classification[0].label
                handLabel = "right_hand" if handLabel == "Left" else "left_hand"

                handData[handLabel] = {
                    "x": float(handCenter.x * 1.0),
                    "y": float(handCenter.y * -1.0),
                    "z": 0.0
                }
        
        return handData