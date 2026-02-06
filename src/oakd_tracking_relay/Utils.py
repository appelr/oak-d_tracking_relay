import cv2

import mediapipe as mp
import numpy as np
import depthai as dai

class Utils: 
    @staticmethod
    def normalizeCoordinate(landmark: mp.tasks.components.containers.NormalizedLandmark):
        offset = 0.5
        flipValue = - 1.0
        
        return [float((landmark.x * flipValue + offset)),  float((landmark.y * flipValue + offset))]
    
class RectificationUtils:
    def __init__(self, camera, config):
        self.camera = camera
        self.config = config

        calibData = self.camera.device.readCalibration()
        
        # Intrinsics
        self.K1 = np.array(calibData.getCameraIntrinsics(dai.CameraBoardSocket.CAM_B, self.config.resolutionWidth, self.config.resolutionHeight))
        self.D1 = np.array(calibData.getDistortionCoefficients(dai.CameraBoardSocket.CAM_B))

        self.K2 = np.array(calibData.getCameraIntrinsics(dai.CameraBoardSocket.CAM_C,self.config.resolutionWidth, self.config.resolutionHeight))
        self.D2 = np.array(calibData.getDistortionCoefficients(dai.CameraBoardSocket.CAM_C))

        # Extrinsics
        extrinsics = np.array(calibData.getCameraExtrinsics(dai.CameraBoardSocket.CAM_B, dai.CameraBoardSocket.CAM_C))
        
        self.R = extrinsics[:3, :3]
        # cm -> mm
        self.T = extrinsics[:3, 3].reshape(3,1) * 10

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

    def rectifyFrames(self, frameL, frameR):
        rectL = cv2.remap(frameL, self.map1x, self.map1y, cv2.INTER_LINEAR)
        rectR = cv2.remap(frameR, self.map2x, self.map2y, cv2.INTER_LINEAR)
        return rectL, rectR
    