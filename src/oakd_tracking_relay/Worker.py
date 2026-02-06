import time
from .ConfigurationManager import Configuration
from .TrackingManager import TrackingEngine
from .UDPManager import UDP
from .CameraManager import OakD

def main():
    Configuration.save()
    config = Configuration.load("config.json")
    engine = TrackingEngine(config)
    udpManager = UDP(config)

    try:
        with OakD(config) as camera:
            while True:
                frameL, frameR, timeStamp = camera.get_frames()

                if frameL is None or frameR is None:
                        time.sleep(0.002)
                        print("Frame skipped")
                        continue
                
                headData, handData = engine.processFrame(frameL, frameR)
                
                if headData or handData:
                    print("Hand/Head found") 
                    udpManager.send(headData, handData, timeStamp)

    except Exception as e:
        print(f"Error in Worker: {e}", flush=True)
        import traceback
        traceback.print_exc()
    finally:
        udpManager.close()
        print("\nWorker stopped.", flush=True)

    
if __name__ == "__main__":
    main()