import time
import depthai as dai
from datetime import timedelta

from oakd_tracking_relay.configuration_manager import Configuration

class OakDPro:
    def __init__(self, config: Configuration):
        self.config = config
        self.pipeline = self._create_Pipeline()
        self.device = None
        self.sync_queue = None
        self.control_queue = None
        self.reference_time = None

    def __enter__(self):
        self.device = dai.Device(self.pipeline, maxUsbSpeed=dai.UsbSpeed.SUPER_PLUS)
        try:
            # Dot Matrix führt zu schlechteren Detection Ergebnissen
            self.device.setIrLaserDotProjectorIntensity(0.0)
            self.device.setIrFloodLightIntensity(float(self.config.ir_laser_intensity_percent/100))
        except: 
            pass
        
        # Epoche bis PC-Start
        self.reference_time = (time.time() - dai.Clock.now().total_seconds()) * 1000.0 # type: ignore
        self.sync_queue = self.device.getOutputQueue(name="sync_out", maxSize=1, blocking=False) # type: ignore
        self.control_queue = self.device.getInputQueue(name="control")
        self.device.setTimesync(timedelta(seconds=2.5), 20, True)
        return self

    def __exit__(self, exc_type, exc, tb):
        if self.device: 
            self.device.close()
        
    def _create_Pipeline(self) -> dai.Pipeline:
        pipeline = dai.Pipeline()
        pipeline.setXLinkChunkSize(0)

        control_input_node = pipeline.create(dai.node.XLinkIn)
        control_input_node.setStreamName('control')

        # Links
        mono_node_left = pipeline.create(dai.node.MonoCamera)
        mono_node_left.setBoardSocket(dai.CameraBoardSocket.CAM_B)
        mono_node_left.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P) 
        mono_node_left.setFps(self.config.fps)
        mono_node_left.initialControl.setManualExposure(self.config.exposure_us, self.config.iso)

        # Rechts
        mono_node_right = pipeline.create(dai.node.MonoCamera)
        mono_node_right.setBoardSocket(dai.CameraBoardSocket.CAM_C)
        mono_node_right.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P) 
        mono_node_right.setFps(self.config.fps)
        mono_node_right.initialControl.setManualExposure(self.config.exposure_us, self.config.iso)

        control_input_node.out.link(mono_node_left.inputControl)
        control_input_node.out.link(mono_node_right.inputControl)

        sync_node = pipeline.create(dai.node.Sync)

        # Outputs
        mono_node_left.out.link(sync_node.inputs["left"])
        mono_node_right.out.link(sync_node.inputs["right"])

        sync_node_out = pipeline.create(dai.node.XLinkOut)
        sync_node_out.setStreamName("sync_out")
        
        sync_node.out.link(sync_node_out.input)   

        return pipeline

    def update_settings(self):
        if self.control_queue is None: 
            return
        
        camera_control = dai.CameraControl()
        camera_control.setManualExposure(self.config.exposure_us, self.config.iso)
        self.control_queue.send(camera_control)
        
        try:
            # Dot Matrix führt zu schlechteren Detection Ergebnissen
            if self.device is not None:
                self.device.setIrLaserDotProjectorIntensity(0.0)
                self.device.setIrFloodLightIntensity(float(self.config.ir_laser_intensity_percent)/100)
        except: 
            pass

    def get_stereo_frame(self):
        if self.sync_queue is not None:
            frame_group = self.sync_queue.get()

        if frame_group is not None:
            frame_left = frame_group["left"]
            frame_right = frame_group["right"]
            
            # Timestamps der Kamera sind relativ zum PC-Start, Umrechnung auf Zeit seit Epoche (1. Januar, 1970)
            timeSinceEpoch = frame_left.getTimestamp().total_seconds() * 1000.0 + self.reference_time
            
            return frame_left.getCvFrame(), frame_right.getCvFrame(), timeSinceEpoch
        
        return None, None, 0.0