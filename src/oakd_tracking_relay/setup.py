import time
import cv2
import mediapipe as mp
import numpy as np
from oakd_tracking_relay.ConfigurationManager import Configuration
from oakd_tracking_relay.CameraManager import OakD
from oakd_tracking_relay.Utils import ProcessingUtils

def main():
    config = Configuration.load("config.json")
    model = mp.solutions.face_mesh.FaceMesh(
            max_num_faces=2, 
            refine_landmarks=True,
            min_detection_confidence=float(config.mp_min_detection_percent)/100, 
            min_tracking_confidence=float(config.mp_min_tracking_percent/100))

    updateFrame = 0
    updateInterval = 15

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
        stereoCoordinates = None

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
            
            frameL, frameR = utils.rectifyStereoFrame(frameL, frameR)

            if updateFrame % updateInterval == 0 or stereoCoordinates is None:
                # Mediapipe braucht RGB
                lRGB = cv2.cvtColor(frameL, cv2.COLOR_GRAY2RGB)
                rRGB = cv2.cvtColor(frameR, cv2.COLOR_GRAY2RGB)
                # Verarbeitung Mediapipe
                resultsL = model.process(lRGB)
                resultsR = model.process(rRGB)

                if resultsL.multi_face_landmarks and resultsR.multi_face_landmarks:
                    camLIrisL = resultsL.multi_face_landmarks[0].landmark[468]
                    camRIrisL = resultsR.multi_face_landmarks[0].landmark[468]

                    landmarksIrisL = utils.createLandmakrDict(camLIrisL.x, camLIrisL.y, camRIrisL.x, camRIrisL.y)
                    stereoCoordinates = utils.stereoLandmarkToPixelCoordinates(landmarksIrisL)
                    
                    prevFrameL, prevFrameR = frameL.copy(), frameR.copy()
            else:
                pointsL = np.array(
                    [[stereoCoordinates["left_cam"]["x"], 
                      stereoCoordinates["left_cam"]["y"]]], 
                      dtype=np.float32).reshape(-1, 1, 2)
                pointsR = np.array(
                    [[stereoCoordinates["right_cam"]["x"], 
                      stereoCoordinates["right_cam"]["y"]]], 
                      dtype=np.float32).reshape(-1, 1, 2)

                nextPointsL, statusL, _ = cv2.calcOpticalFlowPyrLK(
                    prevFrameL, 
                    frameL, 
                    pointsL, 
                    None, 
                    **opticalFlowParams)#
                
                nextPointsR, statusR, _ = cv2.calcOpticalFlowPyrLK(
                    prevFrameR, 
                    frameR, 
                    pointsR, 
                    None, 
                    **opticalFlowParams)

                if statusL[0][0] == 1 and statusR[0][0] == 1:
                    stereoCoordinates = utils.createLandmakrDict(
                        nextPointsL[0][0][0], 
                        nextPointsL[0][0][1], 
                        nextPointsR[0][0][0], 
                        nextPointsR[0][0][1])
                    prevFrameL, prevFrameR = frameL.copy(), frameR.copy()
                else:
                    stereoCoordinates = None

            if stereoCoordinates is not None:
                displayFrame = cv2.cvtColor(frameL, cv2.COLOR_GRAY2BGR)
                iris3D = utils.triangulatePoints(stereoCoordinates)
                pointX, pointY = int(stereoCoordinates["left_cam"]["x"]), int(stereoCoordinates["left_cam"]["y"])
                cv2.circle(displayFrame, (pointX, pointY), 5, (0, 255, 0), -1)
                cv2.putText(displayFrame, f"X:{iris3D['x']:.0f} Y:{iris3D['y']:.0f} Z:{iris3D['z']:.0f}mm",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                cv2.imshow("Preview", displayFrame)
            else:
                cv2.imshow("Preview", frameL)
            updateFrame += 1

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cv2.destroyAllWindows()

                


if __name__ == "__main__":
    main()