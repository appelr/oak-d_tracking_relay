import time

from .ConfigurationManager import Configuration
from .TrackingManager import TrackingEngine
from .UDPManager import UDP
from .CameraManager import OakD
from .Utils import ProcessingUtils

def main():
    config = Configuration.load("config.json")
    engine = TrackingEngine(config)
    udpManager = UDP(config)

    try:
        with OakD(config) as camera:
            processingUtils = ProcessingUtils(camera, config)
            while True:
                frameL, frameR, timeStamp = camera.get_frames()

                if frameL is None or frameR is None:
                        time.sleep(0.002)
                        print("Frame skipped")
                        continue
                
                # Rectify
                frameL, frameR = processingUtils.rectifyStereoFrame(frameL, frameR)

                # Process
                stereoLandmarksIrisL, stereoLandmarksIrisR, landmarksHandL, landmarksHandR = engine.processStereoFrame(frameL, frameR)

                # Initialization
                irisL3D = {}
                irisR3D = {}
                handL = {}
                handR = {}

                if stereoLandmarksIrisL and stereoLandmarksIrisR:
                    # Normalized Landmarks to Pixel
                    irisL2D = processingUtils.stereoLandmarkToPixelCoordinates(stereoLandmarksIrisL)
                    irisR2D = processingUtils.stereoLandmarkToPixelCoordinates(stereoLandmarksIrisR)            

                    # Triangulate to 3D coordinates
                    irisL3D = processingUtils.triangulatePoints(irisL2D)
                    irisR3D = processingUtils.triangulatePoints(irisR2D)

                if landmarksHandL:
                    handL= processingUtils.landmarkToPixelCoordinates(landmarksHandL["x"], landmarksHandL["y"])

                if landmarksHandR:
                    handR = processingUtils.landmarkToPixelCoordinates(landmarksHandR["x"], landmarksHandR["y"])

                udpManager.send(irisL3D, irisR3D, handL, handR, timeStamp)

    except Exception as e:
        print(f"Error in Worker: {e}", flush=True)
        import traceback
        traceback.print_exc()
    finally:
        udpManager.close()
        print("\nWorker stopped.", flush=True)

    
if __name__ == "__main__":
    main()