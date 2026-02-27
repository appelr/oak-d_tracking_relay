import time
import cv2

from oakd_tracking_relay.ConfigurationManager import Configuration, ConfigurationUI, RuntimeState
from oakd_tracking_relay.CameraManager import OakD
from oakd_tracking_relay.Utils import ProcessingUtils
from oakd_tracking_relay.TrackingManager import EyeTracker, TrackerState
from oakd_tracking_relay.TrackingDTO import *
from oakd_tracking_relay.UDPManager import UDP

def main():
    config = Configuration.load("config.json")
    udpManager = UDP(config)
    state = RuntimeState()

    with OakD(config, state) as camera:
        utils = ProcessingUtils(camera=camera, config=config)
        ui = ConfigurationUI(camera, config, state)

        eyeTracker = EyeTracker(utils, config)

        while True:
            # Verarbeitung
            frameL, frameR, timeStamp = camera.get_frames()

            if frameL is None or frameR is None:
                time.sleep(0.002)
                print("Frame skipped")
                continue

            # frameL = cv2.rotate(frameL, cv2.ROTATE_180)
            # frameR = cv2.rotate(frameR, cv2.ROTATE_180)
            # frameL, frameR = frameR, frameL
            
            frameL, frameR = utils.rectifyStereoFrame(frameL, frameR)
            
            eyeTracker.processFrame(frameL, frameR)

            if state.showPreview:
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q") or ui.window_closed():
                    ui.shutdown()
                    cv2.destroyAllWindows()
                    cv2.waitKey(1)
                    state.showPreview = False
                    print(
                        "Preview geschlossen – läuft headless weiter")

                ui.update_config_if_changed()
                displayFrame = cv2.cvtColor(frameL, cv2.COLOR_GRAY2BGR)

            if eyeTracker.currentState == TrackerState.TRACKING and eyeTracker.trackingData.valid():
                dataRate = utils.getDataRate()
                print(dataRate)
                
                irisLeft = utils.triangulatePoints_CV(eyeTracker.trackingData.left.aggregated)
                irisRight = utils.triangulatePoints_CV(eyeTracker.trackingData.right.aggregated)
                udpManager.send(irisLeft, irisRight, Point3D(), Point3D(), timeStamp)
                
                if state.showPreview: 
                    leftPointX, leftPointY = int(eyeTracker.trackingData.left.aggregated.left.x), int(eyeTracker.trackingData.left.aggregated.left.y)
                    rightPointX, rightPointY = int(eyeTracker.trackingData.right.aggregated.left.x), int(eyeTracker.trackingData.right.aggregated.left.y)
                    cv2.circle(displayFrame, (leftPointX, leftPointY), 5, (0, 255, 255), -1)
                    cv2.circle(displayFrame, (rightPointX, rightPointY), 5, (0, 255, 0), -1)

            if state.showPreview:
                cv2.imshow(ui.window_name, displayFrame)
            
if __name__ == "__main__":
    main()