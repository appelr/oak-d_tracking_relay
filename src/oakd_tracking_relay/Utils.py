import cv2

import mediapipe as mp
import numpy as np
import depthai as dai

from typing import Tuple, Dict

class Utils: 
    def __init__(self, config):
        self.config = config
    
class ProcessingUtils:
    def __init__(self, camera, config):
        self.camera = camera
        self.config = config

        calibData = self.camera.device.readCalibration()
        
        # Intrinsics
        self.K1 = np.array(calibData.getCameraIntrinsics(dai.CameraBoardSocket.CAM_B, self.config.resolutionWidth, self.config.resolutionHeight))
        self.D1 = np.array(calibData.getDistortionCoefficients(dai.CameraBoardSocket.CAM_B))

        self.K2 = np.array(calibData.getCameraIntrinsics(dai.CameraBoardSocket.CAM_C,self.config.resolutionWidth, self.config.resolutionHeight))
        self.D2 = np.array(calibData.getDistortionCoefficients(dai.CameraBoardSocket.CAM_C))

        self.fx = self.K1[0,0]
        self.fy = self.K1[1,1]
        self.cx = self.K1[0,2]
        self.cy = self.K1[1,2]

        # Extrinsics
        extrinsics = np.array(calibData.getCameraExtrinsics(dai.CameraBoardSocket.CAM_B, dai.CameraBoardSocket.CAM_C))
        
        self.R = extrinsics[:3, :3]
        # cm -> mm
        self.T = extrinsics[:3, 3].reshape(3,1) * 10

        # cm -> mm
        self.baseline = calibData.getBaselineDistance() * 10

        # Calculate Rectification Matrix
        imageSize = (self.config.resolutionWidth, self.config.resolutionHeight)

        self.R1, self.R2, self.P1, self.P2, self.Q, _, _ = cv2.stereoRectify(
            self.K1, self.D1,
            self.K2, self.D2,
            imageSize=imageSize,
            R=self.R,
            T=self.T,
            flags=cv2.CALIB_ZERO_DISPARITY,
            alpha=0
        )

        # Calculate Remapping Tables
        self.map1x, self.map1y = cv2.initUndistortRectifyMap(
            self.K1, self.D1, 
            self.R1, self.P1,
            imageSize, 
            cv2.CV_16SC2
        )
        self.map2x, self.map2y = cv2.initUndistortRectifyMap(
            self.K2, self.D2, 
            self.R2, self.P2,
            imageSize, 
            cv2.CV_16SC2
        )

    def stereoLandmarkToPixelCoordinates(self, landmarks: Dict) -> Dict: 
        leftCam = landmarks["left_cam"]
        rightCam = landmarks["right_cam"]

        leftX, leftY = self._landmarkToPixelCoordinates(leftCam["x"], leftCam["y"])
        rightX, rightY = self._landmarkToPixelCoordinates(rightCam["x"], rightCam["y"])

        return {
                "left_cam":  {
                    "x": leftX,
                    "y": leftY
                },
                "right_cam": {
                    "x": rightX,
                    "y": rightY
                }
            }

    def _landmarkToPixelCoordinates(self, landmarkX: float, landmarkY: float) -> Tuple[float, float]:
        x = float(landmarkX * self.config.resolutionWidth)
        y = float(landmarkY * self.config.resolutionHeight)

        return x, y

    def rectifyStereoFrame(self, frameL, frameR):
        rectL = cv2.remap(frameL, self.map1x, self.map1y, cv2.INTER_LINEAR)
        rectR = cv2.remap(frameR, self.map2x, self.map2y, cv2.INTER_LINEAR)
        
        return rectL, rectR
    
    # def triangulatePoints(self, coordinates: Dict) -> Dict:
    #     disparity = coordinates["left_cam"]["x"] - coordinates["right_cam"]["x"]
    #     z = self.fx * self.baseline / disparity

    #     return {
    #         "x": (coordinates["left_cam"]["x"]  - self.cx) * z / self.fx,
    #         "y": (coordinates["left_cam"]["y"]  - self.cy) * z / self.fy,
    #         "z": z
    #     }
    
    def createLandmakrDict(self, lx: float, ly: float, rx: float, ry: float) -> Dict:
        return {
            "left_cam": {
                "x": lx,
                "y": ly
            },
            "right_cam": {
                "x": rx,
                "y": ry
            }
        }
    
    def triangulatePoints_CV(self, coordinates: Dict) -> Dict:
        camL = coordinates["left_cam"]
        camR = coordinates["right_cam"] 
        # Formatieren für OpenCV (Shape: 2x1)
        pointsL = np.array([[camL["x"]], [camL["y"]]], dtype=np.float64)
        pointsR = np.array([[camR["x"]], [camR["y"]]], dtype=np.float64)

        # OpenCV Triangulation
        points4D = cv2.triangulatePoints(self.P1, self.P2, pointsL, pointsR)

        # 4D in 3D umwandeln (X, Y, Z durch W teilen)
        points3D = points4D[:3] / points4D[3:]

        # .item() extrahiert die reine Python-Zahl (Float) aus dem NumPy-Array
        x = points3D[0].item()
        y = points3D[1].item()
        z = points3D[2].item()
        print(z)

        return {
            "x": x,
            "y": y,
            "z": z
        }