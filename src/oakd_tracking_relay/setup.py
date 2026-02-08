import time
import cv2
import mediapipe as mp
from oakd_tracking_relay.ConfigurationManager import Configuration
from oakd_tracking_relay.CameraManager import OakD
from oakd_tracking_relay.Utils import ProcessingUtils

def main():
    config = Configuration.load("config.json")
    model = mp.solutions.face_mesh.FaceMesh(
            max_num_faces=2, 
            refine_landmarks=True,
            min_detection_confidence=config.mp_min_detection, 
            min_tracking_confidence=config.mp_min_tracking)
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    

    # --- SETUP SCHIEBEREGLER ---
    cv2.namedWindow("Oak-D Live Stream")

    def nothing(x): 
        pass
    
    cv2.createTrackbar("ISO", "Oak-D Live Stream", config.iso, 200, nothing)
    cv2.setTrackbarMin("ISO", "Oak-D Live Stream", 100)
    cv2.setTrackbarMax("ISO", "Oak-D Live Stream", 600)

    cv2.createTrackbar("Exposure", "Oak-D Live Stream", config.exposure_us, 300, nothing)
    cv2.setTrackbarMin("Exposure", "Oak-D Live Stream", 100)
    cv2.setTrackbarMax("Exposure", "Oak-D Live Stream", 800)

    cv2.createTrackbar("IR Laser", "Oak-D Live Stream", config.ir_laser_intensity, 100, nothing)
    cv2.setTrackbarMin("IR Laser", "Oak-D Live Stream", 0)
    cv2.setTrackbarMax("IR Laser", "Oak-D Live Stream", 100)
    # ---------------------------

    with OakD(config) as camera:
        utils = ProcessingUtils(camera=camera, config=config)
        while True:
                
                # 1. Werte von Trackbars abgreifen
                new_iso = cv2.getTrackbarPos("ISO", "Oak-D Live Stream")
                new_exp = cv2.getTrackbarPos("Exposure", "Oak-D Live Stream")
                new_ir = cv2.getTrackbarPos("IR Laser", "Oak-D Live Stream")

                # 2. Prüfen, ob sich etwas geändert hat
                if new_iso != config.iso or new_exp != config.exposure_us or new_ir != config.ir_laser_intensity:
                    config.iso = new_iso
                    config.exposure_us = new_exp
                    config.ir_laser_intensity = new_ir
                    config.update_trigger = True 
                    camera._updateSettings()
                    config.update_trigger = False
                    config.save()

                frameL, frameR, _ = camera.get_frames()

                if frameL is None or frameR is None:
                        time.sleep(0.002)
                        print("Frame skipped")
                        continue
                
                frameL, frameR = utils.rectifyStereoFrame(frameL, frameR)

                lRGB = cv2.cvtColor(frameL, cv2.COLOR_GRAY2RGB)
                rRGB = cv2.cvtColor(frameR, cv2.COLOR_GRAY2RGB)

                resultsL = model.process(lRGB)
                resultsR = model.process(rRGB)

                if resultsL.multi_face_landmarks and resultsR.multi_face_landmarks:
                    for face_landmarks in resultsL.multi_face_landmarks:
                        mp_drawing.draw_landmarks(
                            image=lRGB,
                            landmark_list=face_landmarks,
                            connections=mp.solutions.face_mesh.FACEMESH_TESSELATION,
                            landmark_drawing_spec=None,
                            connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style()
                        )
                        # Zeichne die Konturen (Augen, Lippen, Gesichtsumriss)
                        mp_drawing.draw_landmarks(
                            image=lRGB,
                            landmark_list=face_landmarks,
                            connections=mp.solutions.face_mesh.FACEMESH_CONTOURS,
                            landmark_drawing_spec=None,
                            connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_contours_style()
                        )
                display_frame = cv2.cvtColor(lRGB, cv2.COLOR_RGB2BGR)
                cv2.imshow("Oak-D Live Stream", display_frame)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        cv2.destroyAllWindows()

                


if __name__ == "__main__":
    main()