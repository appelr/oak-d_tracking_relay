import time

from .ConfigurationManager import Configuration
from .TrackingManager import TrackingEngine
from .UDPManager import UDP
from .CameraManager import OakD
from .Utils import ProcessingUtils
from .TrackingDTO import Point3D

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
                
                # frameL = cv2.rotate(frameL, cv2.ROTATE_180)
                # frameR = cv2.rotate(frameR, cv2.ROTATE_180)
                # frameL, frameR = frameR, frameL
                
                # Rectify
                frameL, frameR = processingUtils.rectifyStereoFrame(frameL, frameR)

                # Process
                stereoPointsIrisL, stereoPointsIrisR, stereoPointsHandL, stereoPointsHandR = engine.processStereoFrame(frameL, frameR)

                # Initialization
                irisL3D = Point3D()
                irisR3D = Point3D()
                handL3D = Point3D()
                handR3D = Point3D()

                if stereoPointsIrisL.valid() and stereoPointsIrisR.valid():
                    # Normalized Landmarks to Pixel
                    irisL2D = processingUtils.stereoLandmarkToPixelCoordinates(stereoPointsIrisL)
                    irisR2D = processingUtils.stereoLandmarkToPixelCoordinates(stereoPointsIrisR)            

                    # Triangulate to 3D coordinates
                    irisL3D = processingUtils.triangulatePoints_CV(irisL2D)
                    irisR3D = processingUtils.triangulatePoints_CV(irisR2D)

                if stereoPointsHandL.valid():
                    handL= processingUtils.stereoLandmarkToPixelCoordinates(stereoPointsHandL)
                    handL3D = processingUtils.triangulatePoints_CV(handL)

                if stereoPointsHandR.valid():
                    handR = processingUtils.stereoLandmarkToPixelCoordinates(stereoPointsHandR)
                    handR3D = processingUtils.triangulatePoints_CV(handR)

                udpManager.send(irisL3D, irisR3D, handL3D, handR3D, timeStamp)

    except Exception as e:
        print(f"Error in Worker: {e}", flush=True)
        import traceback
        traceback.print_exc()
    finally:
        udpManager.close()
        print("\nWorker stopped.", flush=True)

    
if __name__ == "__main__":
    main()