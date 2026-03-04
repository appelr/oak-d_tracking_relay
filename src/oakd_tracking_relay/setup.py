import time
import cv2
import concurrent.futures

from oakd_tracking_relay.ConfigurationManager import Configuration, ConfigurationUI, RuntimeState
from oakd_tracking_relay.CameraManager import OakD
from oakd_tracking_relay.Utils import ProcessingUtils
from oakd_tracking_relay.TrackingManager import *
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
        handTracker = HandTracker(config)

        executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        hand_task = None

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

            if config.clahe == 1:
                frameL = utils.clahe.apply(frameL)
                frameR = utils.clahe.apply(frameR)


            frameL, frameR = utils.rectifyStereoFrame(frameL, frameR)
            
            eyeTracker.processFrame(frameL, frameR)

            if hand_task is None or hand_task.done():
                hand_task = executor.submit(handTracker.detect, frameL.copy(), frameR.copy())

            dataRate = utils.getDataRate()

            if eyeTracker.currentState == TrackerState.TRACKING and eyeTracker.trackingData.valid():
                print(dataRate)
                irisLeft = utils.triangulatePoints_CV(eyeTracker.trackingData.left.aggregated)
                irisRight = utils.triangulatePoints_CV(eyeTracker.trackingData.right.aggregated)

                # Check if distance is too far apart for eyes
                dist = np.abs(irisLeft.z - irisRight.z)
                if dist < 80:
                    udpManager.send(irisLeft, irisRight, Point3D(), Point3D(), timeStamp)

            if handTracker.trackingData.valid():
                handLeft = utils.triangulatePoints_CV(handTracker.trackingData.left.aggregated)
                handRight = utils.triangulatePoints_CV(handTracker.trackingData.right.aggregated)

                udpManager.send(Point3D(), Point3D(), handLeft, handRight, timeStamp)

            if state.showPreview:
                if ui.shouldExit():
                    ui.exit()
                    state.showPreview = False
                else:
                    ui.updateConfigIfChanged()
                    ui.setDisplayFrame(frameL)
                    ui.drawTrackingData(eyeTracker.trackingData)
                    ui.drawTrackingData(handTracker.trackingData)
                    ui.drawDataRate(dataRate)
                    ui.show()
            
if __name__ == "__main__":
    main()