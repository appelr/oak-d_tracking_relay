import time
import concurrent.futures

from oakd_tracking_relay.configuration_manager import Configuration
from oakd_tracking_relay.ui_manager import ConfigurationUI
from oakd_tracking_relay.camera_manager import OakDPro
from oakd_tracking_relay.Utils import ProcessingUtils
from oakd_tracking_relay.tracking_manager import *
from oakd_tracking_relay.tracking_dto import *
from oakd_tracking_relay.udp_manager import UDPSender

def main():
    config = Configuration.load_from_file("config.json")
    udp_manager = UDPSender(config=config)
    show_configuration_ui = True

    try:
        with OakDPro(config=config) as camera:
            utils = ProcessingUtils(camera=camera, config=config)
            ui = ConfigurationUI(camera=camera, config=config)

            eye_tracker = EyeTracker(utils=utils, config=config)
            hand_tracker = HandTracker(utils=utils, config=config)

            # Hände asynchron, da nicht relevant für real time tracking
            async_executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
            hand_tracking_task = None
            hand_upper_left, hand_lower_left, hand_upper_right, hand_lower_right = False, False, False, False


            while True:
                frame_left, frame_right, timestamp = camera.get_stereo_frame()

                if frame_left is None or frame_right is None:
                    time.sleep(0.002)
                    print("Unvollständiges Stereo-Frame Paar - Skip")
                    continue
                
                # Frames rotieren und tauschen, da Kamera falschherum montiert ist
                # frame_left = cv2.rotate(frame_left, cv2.ROTATE_180)
                # frame_right = cv2.rotate(frame_right, cv2.ROTATE_180)
                # frame_left, frame_right = frame_right, frame_left

                frame_left, frame_right = utils.rectify_stereo_frame(frame_left=frame_left, frame_right=frame_right)
                
                if config.apply_clahe == 1:
                    frame_left, frame_right = utils.apply_clahe_to_stereo_frame(frame_left=frame_left, frame_right=frame_right)
                
                eye_tracker.process_stereo_frame(frame_left=frame_left, frame_right=frame_right)

                if hand_tracking_task is None or hand_tracking_task.done():
                    
                    if hand_tracking_task is not None:
                        try:
                            hand_upper_left, hand_lower_left, hand_upper_right, hand_lower_right = hand_tracking_task.result()
                        except Exception as e:
                            print(f"Fehler im asynchronen Hand-Thread: {e}")

                    hand_tracking_task = async_executor.submit(hand_tracker.process, frame_left.copy())

                data_rate = utils.get_data_rate()

                if eye_tracker.current_state == TrackerState.TRACKING and eye_tracker.tracking_data.valid():
                    iris_left_stereo = eye_tracker.tracking_data.left.aggregated
                    iris_right_stereo = eye_tracker.tracking_data.right.aggregated
                    iris_left_3D = utils.triangulate_stereo_point(stereo_point=iris_left_stereo)
                    iris_right_3D = utils.triangulate_stereo_point(stereo_point=iris_right_stereo)

                    # Letzter Guard - Prüft, ob Augen realistische Distanz haben
                    if np.abs(iris_left_3D.z - iris_right_3D.z) < 80:
                        udp_manager.send_eyes(iris_left=iris_left_3D, iris_right=iris_right_3D, timestamp=timestamp)

                if hand_upper_left or hand_lower_left or hand_upper_right or hand_lower_right:
                    udp_manager.send_hands(upper_left=hand_upper_left, lower_left=hand_lower_left, upper_right=hand_upper_right, lower_right=hand_lower_right, timestamp=timestamp)
                
                if show_configuration_ui:
                    if ui.check_for_close_key():
                        ui.exit()
                        show_configuration_ui = False
                    else:
                        ui.update_config_if_changed()
                        ui.change_display_frame(display_frame=frame_left)
                        ui.draw_eye_landmarks(tracking_data=eye_tracker.tracking_data)
                        ui.draw_hand_quadrants(upper_left=hand_upper_left, lower_left=hand_lower_left, upper_right=hand_upper_right, lower_right=hand_lower_right)
                        ui.draw_data_rate(data_rate=data_rate)
                        ui.show()
    finally:
        async_executor.shutdown(wait=False)
        udp_manager.close()
            
if __name__ == "__main__":
    main()