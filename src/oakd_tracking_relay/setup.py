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
        handTracker = HandTracker(utils, config)

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
                hand_task = executor.submit(handTracker.processFrame, frameL.copy(), frameR.copy())

            dataRate = utils.getDataRate()

            if eyeTracker.currentState == TrackerState.TRACKING and eyeTracker.trackingData.valid():
                print(dataRate)
                irisLeftStereo = eyeTracker.trackingData.left.aggregated
                irisRightStereo = eyeTracker.trackingData.right.aggregated
                irisLeft3D = utils.triangulatePoints_CV(irisLeftStereo)
                irisRight3D = utils.triangulatePoints_CV(irisRightStereo)

                # Check if distance is too far apart for eyes
                dist = np.abs(irisLeft3D.z - irisRight3D.z)
                if dist < 80:
                    udpManager.sendEyes(irisLeft3D, irisRight3D, timeStamp)

            if handTracker.trackingData.valid():
                handLeftStereo = handTracker.trackingData.left.aggregated
                handRightStereo = handTracker.trackingData.right.aggregated
                handLeft3D = utils.triangulatePoints_CV(handLeftStereo)
                handRight3D = utils.triangulatePoints_CV(handRightStereo)

                udpManager.sendHands(handLeft3D, handRight3D, timeStamp)

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