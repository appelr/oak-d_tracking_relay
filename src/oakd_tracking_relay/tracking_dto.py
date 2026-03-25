import numpy as np
from typing import List
from enum import Enum, auto
from dataclasses import dataclass, field

@dataclass
class Point2D:
    x: float = 0.0
    y: float = 0.0

    def valid(self) -> bool:
        return not (self.x == 0.0 and self.y == 0.0)

    def as_np(self) -> np.ndarray:
        return np.array([self.x, self.y], dtype=np.float32)
    
@dataclass
class Point3D:
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0

    def valid(self) -> bool:
        return not (self.x == 0.0 and self.y == 0.0 and self.z == 0.0)

    def as_np(self) -> np.ndarray:
        return np.array([self.x, self.y, self.z], dtype=np.float32)
    
@dataclass
class StereoPoint:
    left_cam: Point2D = field(default_factory=Point2D)
    right_cam: Point2D = field(default_factory=Point2D)

    def valid(self) -> bool:
        return self.left_cam.valid() and self.right_cam.valid()
    
@dataclass
class EyeStereoPointCluster:
    MIN_TRACKING_POINTS = 2
    stereo_points: List[StereoPoint] = field(default_factory=list)
    iris: StereoPoint = field(default_factory=StereoPoint)

    def valid(self) -> bool:
        return len(self.stereo_points) > self.MIN_TRACKING_POINTS

    def aggregate_median(self):
        left_points = [p.left_cam.as_np() for p in self.stereo_points if p.left_cam.valid()]
        right_points = [p.right_cam.as_np() for p in self.stereo_points if p.right_cam.valid()]

        if left_points:
            median = np.median(left_points, axis=0)
            self.iris.left_cam = Point2D(*median)

        if right_points:
            median = np.median(right_points, axis=0)
            self.iris.right_cam = Point2D(*median)

@dataclass
class TrackingData:
    left_eye: EyeStereoPointCluster = field(default_factory=EyeStereoPointCluster)
    right_eye: EyeStereoPointCluster = field(default_factory=EyeStereoPointCluster)
    center_between_eyes: StereoPoint = field(default_factory=StereoPoint)

    def valid(self) -> bool: 
        return self.left_eye.valid() and self.right_eye.valid()
    
    def aggregate_median(self):
        left_cam_points = []
        right_cam_points = []

        if self.left_eye.valid():
            self.left_eye.aggregate_median()
            left_cam_points.append(self.left_eye.iris.left_cam.as_np())
            right_cam_points.append(self.left_eye.iris.right_cam.as_np())

        if self.right_eye.valid():
            self.right_eye.aggregate_median()
            left_cam_points.append(self.right_eye.iris.left_cam.as_np())
            right_cam_points.append(self.right_eye.iris.right_cam.as_np())

        if left_cam_points:
            median_left = np.median(left_cam_points, axis=0)
            self.center_between_eyes.left_cam = Point2D(*median_left)

        if right_cam_points:
            median_right = np.median(right_cam_points, axis=0)
            self.center_between_eyes.right_cam = Point2D(*median_right)
