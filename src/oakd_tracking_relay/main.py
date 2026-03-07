import time
import concurrent.futures

from oakd_tracking_relay.configuration_manager import Configuration, ConfigurationUI, RuntimeState
from oakd_tracking_relay.camera_manager import OakD
from oakd_tracking_relay.Utils import ProcessingUtils
from oakd_tracking_relay.tracking_manager import *
from oakd_tracking_relay.tracking_dto import *
from oakd_tracking_relay.udp_manager import UDP

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
        handTrackingTask = None
        hand_ol, hand_ul, hand_or, hand_ur = False, False, False, False

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

            if config.apply_clahe == 1:
                frameL = utils.clahe.apply(frameL)
                frameR = utils.clahe.apply(frameR)


            frameL, frameR = utils.rectifyStereoFrame(frameL, frameR)
            
            eyeTracker.processFrame(frameL, frameR)

            if handTrackingTask is None or handTrackingTask.done():
                
                if handTrackingTask is not None:
                    try:
                        hand_ol, hand_ul, hand_or, hand_ur = handTrackingTask.result()
                    except Exception as e:
                        print(f"Fehler im Hand-Thread: {e}")

                handTrackingTask = executor.submit(handTracker.check_presence, frameL.copy())

            dataRate = utils.getDataRate()

            if eyeTracker.currentState == TrackerState.TRACKING and eyeTracker.trackingData.valid():
                #print(dataRate)
                irisLeftStereo = eyeTracker.trackingData.left.aggregated
                irisRightStereo = eyeTracker.trackingData.right.aggregated
                irisLeft3D = utils.triangulatePoints_CV(irisLeftStereo)
                irisRight3D = utils.triangulatePoints_CV(irisRightStereo)

                # Check if distance is too far apart for eyes
                dist = np.abs(irisLeft3D.z - irisRight3D.z)
                if dist < 80:
                    udpManager.sendEyes(irisLeft3D, irisRight3D, timeStamp)

            if hand_ol or hand_ul or hand_or or hand_ur:
                udpManager.sendHands(hand_ol, hand_ul, hand_or, hand_ur, timeStamp)
            
            if state.showPreview:
                if ui.shouldExit():
                    ui.exit()
                    state.showPreview = False
                else:
                    ui.updateConfigIfChanged()
                    ui.setDisplayFrame(frameL)
                    ui.drawTrackingData(eyeTracker.trackingData)
                    ui.drawHandQuadrants(hand_ol, hand_ul, hand_or, hand_ur)
                    ui.drawDataRate(dataRate)
                    ui.show()
            
if __name__ == "__main__":
    main()