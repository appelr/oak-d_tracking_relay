import time
import cv2
import mediapipe as mp
import numpy as np
from oakd_tracking_relay.ConfigurationManager import Configuration
from oakd_tracking_relay.CameraManager import OakD
from oakd_tracking_relay.Utils import ProcessingUtils
from oakd_tracking_relay.TrackingDTO import *

def main():
    config = Configuration.load("config.json")
    model = mp.solutions.face_mesh.FaceMesh(
            max_num_faces=2, 
            refine_landmarks=True,
            min_detection_confidence=float(config.mp_min_detection_percent)/100, 
            min_tracking_confidence=float(config.mp_min_tracking_percent/100))

    detectionBuffer = []
    maxJump = 20
    maxDispDelta = 3
    confidenceInit = 5
    confidenceMin = 0
    recheckInterval = 20

    opticalFlowParams = dict(winSize=(21, 21), maxLevel=3,
                     criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.03))

    cv2.namedWindow("Preview")

    def nothing(x): 
        pass
    
    cv2.createTrackbar("ISO", "Preview", config.iso, config.iso, nothing)
    cv2.setTrackbarMin("ISO", "Preview", 100)
    cv2.setTrackbarMax("ISO", "Preview", 600)

    cv2.createTrackbar("Exposure", "Preview", config.exposure_us, config.exposure_us, nothing)
    cv2.setTrackbarMin("Exposure", "Preview", 100)
    cv2.setTrackbarMax("Exposure", "Preview", 800)

    cv2.createTrackbar("IR Laser", "Preview", config.ir_laser_intensity_percent, config.ir_laser_intensity_percent, nothing)
    cv2.setTrackbarMin("IR Laser", "Preview", 0)
    cv2.setTrackbarMax("IR Laser", "Preview", 100)

    cv2.createTrackbar("Min. Detection", "Preview", config.mp_min_detection_percent, config.mp_min_detection_percent, nothing)
    cv2.setTrackbarMin("Min. Detection", "Preview", 0)
    cv2.setTrackbarMax("Min. Detection", "Preview", 100)

    cv2.createTrackbar("Min. Tracking", "Preview", config.mp_min_tracking_percent, config.mp_min_tracking_percent, nothing)
    cv2.setTrackbarMin("Min. Tracking", "Preview", 0)
    cv2.setTrackbarMax("Min. Tracking", "Preview", 100)
    # ---------------------------

    with OakD(config) as camera:
        utils = ProcessingUtils(camera=camera, config=config)
        prevFrameL, prevFrameR = None, None
        stereoCoordinates = StereoPoint()
        trackingConfidence = 0
        frameCounter = 0

        while True:
            # Config Änderungen
            new_iso = cv2.getTrackbarPos("ISO", "Preview")
            new_exp = cv2.getTrackbarPos("Exposure", "Preview")
            new_ir = cv2.getTrackbarPos("IR Laser", "Preview")
            new_det = cv2.getTrackbarPos("Min. Detection", "Preview")
            new_tra = cv2.getTrackbarPos("Min. Tracking", "Preview")

            if new_iso != config.iso or new_exp != config.exposure_us or new_ir != config.ir_laser_intensity_percent or new_det != config.mp_min_detection_percent or new_tra != config.mp_min_tracking_percent:
                config.iso = new_iso
                config.exposure_us = new_exp
                config.ir_laser_intensity_percent = new_ir
                config.mp_min_detection_percent = new_det
                config.mp_min_tracking_percent = new_tra
                config.update_trigger = True 
                camera._updateSettings()
                config.update_trigger = False
                config.save()

            # Verarbeitung
            frameL, frameR, _ = camera.get_frames()

            if frameL is None or frameR is None:
                time.sleep(0.002)
                print("Frame skipped")
                continue

            # frameL = cv2.rotate(frameL, cv2.ROTATE_180)
            # frameR = cv2.rotate(frameR, cv2.ROTATE_180)
            # frameL, frameR = frameR, frameL
            
            frameL, frameR = utils.rectifyStereoFrame(frameL, frameR)
            displayFrame = cv2.cvtColor(frameL, cv2.COLOR_GRAY2BGR)

            if not stereoCoordinates.valid():
                # Mediapipe braucht RGB
                lRGB = cv2.cvtColor(frameL, cv2.COLOR_GRAY2RGB)
                rRGB = cv2.cvtColor(frameR, cv2.COLOR_GRAY2RGB)
                # Verarbeitung Mediapipe
                resultsL = model.process(lRGB)
                resultsR = model.process(rRGB)

                if resultsL.multi_face_landmarks and resultsR.multi_face_landmarks:
                    camLIrisL = resultsL.multi_face_landmarks[0].landmark[473]
                    camRIrisL = resultsR.multi_face_landmarks[0].landmark[473]

                    irisL = StereoPoint(Point2D(camLIrisL.x, camLIrisL.y), Point2D(camRIrisL.x, camRIrisL.y))
                    detectionBuffer.append(utils.stereoLandmarkToPixelCoordinates(irisL))

                    if len(detectionBuffer) >= 3:
                        p1 = detectionBuffer[0].left
                        p3 = detectionBuffer[-1].left
                        
                        # Distanz berechnen (hat sich das Auge bewegt?)
                        dist = np.sqrt((p1.x - p3.x)**2 + (p1.y - p3.y)**2)
                        
                        # Wenn stabil (< 3 Pixel Bewegung) -> START TRACKING
                        if dist < 3.0:
                            stereoCoordinates = detectionBuffer[-1]
                            prevFrameL, prevFrameR = frameL.copy(), frameR.copy()
                            trackingConfidence = confidenceInit
                            detectionBuffer = []
                        else:
                            detectionBuffer.pop(0)
                else:
                    detectionBuffer = []
            else:
                pointsL = stereoCoordinates.left.as_np().reshape(-1, 1, 2)
                pointsR = stereoCoordinates.right.as_np().reshape(-1, 1, 2)

                nextPointsL, statusL, _ = cv2.calcOpticalFlowPyrLK(
                    prevFrameL, 
                    frameL, 
                    pointsL, 
                    None, 
                    **opticalFlowParams)
                
                nextPointsR, statusR, _ = cv2.calcOpticalFlowPyrLK(
                    prevFrameR, 
                    frameR, 
                    pointsR, 
                    None, 
                    **opticalFlowParams)

                if statusL[0][0] == 1 and statusR[0][0] == 1:
                    nextLx, nextLy = nextPointsL[0][0]
                    nextRx, nextRy = nextPointsR[0][0]

                    distX = nextLx - stereoCoordinates.left.x
                    distY = nextLy - stereoCoordinates.left.y

                    prevDisp = (stereoCoordinates.left.x - stereoCoordinates.right.x)
                    currentDisp = nextLx - nextRx

                    if np.hypot(distX, distY) > maxJump or abs(currentDisp - prevDisp) > maxDispDelta:
                        trackingConfidence -= 1

                        if trackingConfidence <= confidenceMin:
                            stereoCoordinates = StereoPoint()
                            detectionBuffer = []

                        continue
                    
                    stereoCoordinates = StereoPoint(Point2D(nextLx, nextLy), Point2D(nextRx, nextRy))
                    
                    prevFrameL, prevFrameR = frameL.copy(), frameR.copy()

                    trackingConfidence = min(trackingConfidence + 1, confidenceInit)
                else:
                    trackingConfidence -= 2
                    if trackingConfidence <= confidenceMin:
                        stereoCoordinates = StereoPoint()
                        detectionBuffer = []

            frameCounter += 1

            if stereoCoordinates.valid() and frameCounter % recheckInterval == 0:
                # Mediapipe braucht RGB
                lRGB = cv2.cvtColor(frameL, cv2.COLOR_GRAY2RGB)
                rRGB = cv2.cvtColor(frameR, cv2.COLOR_GRAY2RGB)
                # Verarbeitung Mediapipe
                resultsL = model.process(lRGB)
                resultsR = model.process(rRGB)

                if resultsL.multi_face_landmarks and resultsR.multi_face_landmarks:
                    camLIrisL = resultsL.multi_face_landmarks[0].landmark[473]
                    camRIrisL = resultsR.multi_face_landmarks[0].landmark[473]

                    stereoPoint = utils.stereoLandmarkToPixelCoordinates(StereoPoint(Point2D(camLIrisL.x, camLIrisL.y), Point2D(camRIrisL.x, camRIrisL.y)))
                   
                    dist = np.hypot(
                        stereoPoint.left.x - stereoCoordinates.left.x,
                        stereoPoint.left.y - stereoCoordinates.left.y
                    )

                    if dist > 6:
                        stereoCoordinates = stereoPoint
                        prevFrameL, prevFrameR = frameL.copy(), frameR.copy()
                        trackingConfidence = confidenceInit

            if stereoCoordinates.valid():
                # iris3D = utils.triangulatePoints(stereoCoordinates)
                iris3D = utils.triangulatePoints_CV(stereoCoordinates)

                pointX, pointY = int(stereoCoordinates.left.x), int(stereoCoordinates.left.y)
                cv2.circle(displayFrame, (pointX, pointY), 5, (0, 255, 0), -1)
                cv2.putText(displayFrame, f"X:{iris3D.x:.0f} Y:{iris3D.y:.0f} Z:{iris3D.z:.0f}mm",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                cv2.imshow("Preview", displayFrame)
            else:
                cv2.imshow("Preview", frameL)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()