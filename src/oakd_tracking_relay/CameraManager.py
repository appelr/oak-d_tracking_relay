import depthai as dai

from .ConfigurationManager import Configuration

class OakD:
    def __init__(self, config: Configuration):
        self.config = config
        self.pipeline = self._createPipeline()
        self.device = None
        self.qLeft = None
        self.qRight = None
        self.qControl = None

    def __enter__(self):
        self.device = dai.Device(self.pipeline, maxUsbSpeed=dai.UsbSpeed.SUPER_PLUS)
        try:
            self.device.setIrLaserDotProjectorIntensity(0.0)
            self.device.setIrFloodLightIntensity(self.config.ir_laser_intensity)
        except: 
            pass
        
        self.qLeft = self.device.getOutputQueue(name="left", maxSize=1, blocking=False)
        self.qRight = self.device.getOutputQueue(name="right", maxSize=1, blocking=False)
        self.qControl = self.device.getInputQueue(name="control")
        
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.device: 
            self.device.close()

    def _createPipeline(self) -> dai.Pipeline:
        pipeline = dai.Pipeline()
        pipeline.setXLinkChunkSize(0)

        controlIn = pipeline.create(dai.node.XLinkIn)
        controlIn.setStreamName('control')

        # Links
        monoLeft = pipeline.create(dai.node.MonoCamera)
        monoLeft.setBoardSocket(dai.CameraBoardSocket.CAM_B)
        monoLeft.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P) 
        monoLeft.setFps(self.config.fps)
        monoLeft.initialControl.setManualExposure(self.config.exposure_us, self.config.iso)

        # Rechts
        monoRight = pipeline.create(dai.node.MonoCamera)
        monoRight.setBoardSocket(dai.CameraBoardSocket.CAM_C)
        monoRight.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P) 
        monoRight.setFps(self.config.fps)
        monoRight.initialControl.setManualExposure(self.config.exposure_us, self.config.iso)

        controlIn.out.link(monoLeft.inputControl)
        controlIn.out.link(monoRight.inputControl)

        # Outputs
        xoutLeft = pipeline.create(dai.node.XLinkOut)
        xoutLeft.setStreamName("left")
        
        xoutRight = pipeline.create(dai.node.XLinkOut)
        xoutRight.setStreamName("right")

        monoLeft.out.link(xoutLeft.input)
        monoRight.out.link(xoutRight.input)
        
        return pipeline

    def _updateSettings(self):
        if self.qControl is None: 
            return
        
        if not getattr(self.config, 'update_trigger', False): 
            return
        
        cameraControl = dai.CameraControl()
        cameraControl.setManualExposure(self.config.exposure_us, self.config.iso)
        self.qControl.send(cameraControl)
        
        try:
            self.device.setIrLaserDotProjectorIntensity(0.0)
            self.device.setIrFloodLightIntensity(self.config.ir_laser_intensity)
        except: 
            pass

    def get_frames(self):
        inLeft = self.qLeft.tryGet()
        inRight = self.qRight.tryGet()

        while self.qLeft.has(): 
            inLeft = self.qLeft.get()

        while self.qRight.has(): 
            inRight = self.qRight.get()

        if inLeft is not None and inRight is not None:
            timeStamp = inLeft.getTimestamp().total_seconds() * 1000
            return inLeft.getCvFrame(), inRight.getCvFrame(), timeStamp
        
        return None, None, 0.0