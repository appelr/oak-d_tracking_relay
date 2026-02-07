import cv2
import numpy as np
import mediapipe as mp

from typing import Dict, Tuple, Optional

from .ConfigurationManager import Configuration

class TrackingEngine:
    def __init__(self, config: Configuration):
        self.config = config
        self.headTracker = HeadTracker(config)
        self.handTracker = HandTracker(config)

    def processStereoFrame(self, frameL: np.ndarray, frameR: np.ndarray) -> Tuple[Dict, Dict, Dict, Dict]:
        lRGB = cv2.cvtColor(frameL, cv2.COLOR_GRAY2RGB)
        rRGB = cv2.cvtColor(frameR, cv2.COLOR_GRAY2RGB)
        
        irisL, irisR = self.headTracker.process(lRGB, rRGB)
        handL, handR = self.handTracker.process(lRGB, rRGB)

        return irisL, irisR, handL, handR

class HeadTracker:
    def __init__(self, config: Configuration):
        self.config = config
        self.model = mp.solutions.face_mesh.FaceMesh(
            max_num_faces=2, 
            refine_landmarks=True,
            min_detection_confidence=config.mp_min_detection, 
            min_tracking_confidence=config.mp_min_tracking)
        

    def process(self, lRGB: np.ndarray, rRGB: np.ndarray) -> Tuple[Dict, Dict]:
        irisL = {}
        irisR = {}

        # Execute model processing
        lResult = self.model.process(lRGB)
        rResult = self.model.process(rRGB)

        if lResult.multi_face_landmarks and rResult.multi_face_landmarks:
            lLandmarks = lResult.multi_face_landmarks[0].landmark
            rLandmarks = rResult.multi_face_landmarks[0].landmark

            # Left iris
            lLandmarksIrisL = lLandmarks[468]
            rLandmarksIrisL = rLandmarks[468]

            # Right iris
            lLandmarksIrisR = lLandmarks[473]
            rLandmarksIrisR = rLandmarks[473]

            irisL = {
                "left_cam":  {
                    "x": lLandmarksIrisL.x,
                    "y": lLandmarksIrisL.y
                },
                "right_cam": {
                    "x": rLandmarksIrisL.x,
                    "y": rLandmarksIrisL.y
                }
            }
            
            irisR = {
                "left_cam":  {
                    "x": lLandmarksIrisR.x,
                    "y": lLandmarksIrisR.y
                },
                "right_cam": {
                    "x": rLandmarksIrisR.x,
                    "y": rLandmarksIrisR.y
                }
            }

        return irisL, irisR

class HandTracker:
    def __init__(self, config: Configuration):
        self.config = config
        self.model = mp.solutions.hands.Hands(
            max_num_hands=2,
            model_complexity=0,
            min_detection_confidence=config.mp_min_detection,
            min_tracking_confidence=config.mp_min_tracking
        )

    def process(self, lRGB: np.ndarray, rRGB: np.ndarray) -> Tuple[Dict, Dict]:
        handL = {}
        handR = {}

        lResult = self.model.process(lRGB)
        rResult = self.model.process(rRGB)

        if lResult.multi_hand_landmarks and rResult.multi_hand_landmarks:
            handsCamR = {}
            for rLandmakrs, handedness in zip(rResult.multi_hand_landmarks, rResult.multi_handedness):
                handLabel = handedness.classification[0].label
                handCenter = rLandmakrs.landmark[9]
                handsCamR[handLabel] = handCenter

            for lLandmarks, handedness in zip(lResult.multi_hand_landmarks, lResult.multi_handedness):
                handLabel = handedness.classification[0].label
                
                if handLabel in handsCamR:
                    handCenterCamL = lLandmarks.landmark[9]
                    handCenterCamR = handsCamR[handLabel]

                    # Swap hands for ego perspective
                    if handLabel == "Right":
                        handL = {
                            "left_cam": {
                                "x": handCenterCamL.x ,
                                "y": handCenterCamL.y
                            },
                            "right_cam": {
                                "x": handCenterCamR.x,
                                "y": handCenterCamR.y
                            }
                        }
                    if handLabel == "Left":
                        handR = {
                            "left_cam": {
                                "x": handCenterCamL.x ,
                                "y": handCenterCamL.y
                            },
                            "right_cam": {
                                "x": handCenterCamR.x,
                                "y": handCenterCamR.y
                            }
                        }
        
        return handL, handR