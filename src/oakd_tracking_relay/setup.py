import time
import cv2

from oakd_tracking_relay.ConfigurationManager import Configuration
from oakd_tracking_relay.CameraManager import OakD
from oakd_tracking_relay.Utils import ProcessingUtils
from oakd_tracking_relay.TrackingManager import EyeTracker, TrackerState
from oakd_tracking_relay.TrackingDTO import *

def main():
    config = Configuration.load("config.json")
    cv2.namedWindow("Preview")

    def nothing(x): 
        pass
    
    cv2.createTrackbar("ISO", "Preview", config.iso, config.iso, nothing)
    cv2.setTrackbarMin("ISO", "Preview", 400)
    cv2.setTrackbarMax("ISO", "Preview", 1000)

    cv2.createTrackbar("Exposure", "Preview", config.exposure_us, config.exposure_us, nothing)
    cv2.setTrackbarMin("Exposure", "Preview", 10)
    cv2.setTrackbarMax("Exposure", "Preview", 100)

    cv2.createTrackbar("IR Laser", "Preview", config.ir_laser_intensity_percent, config.ir_laser_intensity_percent, nothing)
    cv2.setTrackbarMin("IR Laser", "Preview", 0)
    cv2.setTrackbarMax("IR Laser", "Preview", 100)

    cv2.createTrackbar("Min. Detection", "Preview", config.mp_min_detection_percent, config.mp_min_detection_percent, nothing)
    cv2.setTrackbarMin("Min. Detection", "Preview", 0)
    cv2.setTrackbarMax("Min. Detection", "Preview", 100)

    cv2.createTrackbar("Min. Tracking", "Preview", config.mp_min_tracking_percent, config.mp_min_tracking_percent, nothing)
    cv2.setTrackbarMin("Min. Tracking", "Preview", 0)
    cv2.setTrackbarMax("Min. Tracking", "Preview", 100)

    with OakD(config) as camera:
        utils = ProcessingUtils(camera=camera, config=config)
        
        eyeTracker = EyeTracker(utils, config)
        prev_frame_time = 0.0
        smoothed_dt = 0.0
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


            # ==========================================
            # FPS KORREKT GEGLÄTTET BERECHNEN
            # ==========================================
            current_frame_time = time.perf_counter()
            if prev_frame_time != 0:
                dt = current_frame_time - prev_frame_time # Verstrichene Zeit für diesen Frame
                
                if smoothed_dt == 0.0:
                    smoothed_dt = dt
                else:
                    # Wir glätten die Sekunden! 90% alter Zeitwert, 10% neuer Zeitwert
                    smoothed_dt = (smoothed_dt * 0.90) + (dt * 0.10)
                
                # FPS ist 1 / durchschnittliche Dauer
                fps = 1.0 / smoothed_dt 
            else:
                fps = 0.0
                
            prev_frame_time = current_frame_time
            # ==========================================
        

            eyeTracker.processFrame(frameL, frameR)

            if eyeTracker.currentState == TrackerState.TRACKING and eyeTracker.trackingData.valid():
                print(fps)
                irisLeft = utils.triangulatePoints_CV(eyeTracker.trackingData.left.aggregated)
                irisRight = utils.triangulatePoints_CV(eyeTracker.trackingData.right.aggregated)

                leftPointX, leftPointY = int(eyeTracker.trackingData.left.aggregated.left.x), int(eyeTracker.trackingData.left.aggregated.left.y)
                rightPointX, rightPointY = int(eyeTracker.trackingData.right.aggregated.left.x), int(eyeTracker.trackingData.right.aggregated.left.y)
                cv2.circle(displayFrame, (leftPointX, leftPointY), 5, (0, 255, 255), -1)
                cv2.circle(displayFrame, (rightPointX, rightPointY), 5, (0, 255, 0), -1)

                cv2.putText(displayFrame, f"Left: X:{irisLeft.x:.0f} Y:{irisLeft.y:.0f} Z:{irisLeft.z:.0f}mm",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                cv2.putText(displayFrame, f"Right: X:{irisRight.x:.0f} Y:{irisRight.y:.0f} Z:{irisRight.z:.0f}mm",
                            (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(displayFrame, f"FPS: {int(fps)}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.imshow("Preview", displayFrame)
            else:
                cv2.putText(displayFrame, f"FPS: {int(fps)}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.imshow("Preview", frameL)

            if cv2.waitKey(1) & 0xFF == ord('q'):
            # if 0xFF == ord('q'):
                break

        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()