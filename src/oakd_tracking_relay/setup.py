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
            dataRate = utils.getDataRate()

            if eyeTracker.currentState == TrackerState.TRACKING and eyeTracker.trackingData.valid():
                print(dataRate)
                irisLeft = utils.triangulatePoints_CV(eyeTracker.trackingData.left.aggregated)
                irisRight = utils.triangulatePoints_CV(eyeTracker.trackingData.right.aggregated)
                udpManager.send(irisLeft, irisRight, Point3D(), Point3D(), timeStamp)

            if state.showPreview:
                if ui.shouldExit():
                    ui.exit()
                    state.showPreview = False
                else:
                    ui.updateConfigIfChanged()
                    ui.setDisplayFrame(frameL)
                    ui.drawTrackingData(eyeTracker.trackingData, dataRate)
                    ui.show()
            
if __name__ == "__main__":
    main()