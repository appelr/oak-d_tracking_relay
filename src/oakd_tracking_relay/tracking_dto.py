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
    left: Point2D = field(default_factory=Point2D)
    right: Point2D = field(default_factory=Point2D)

    def valid(self) -> bool:
        return self.left.valid() and self.right.valid()
    
@dataclass
class StereoPointCluster:
    MIN_TRACKING_POINTS = 2
    stereo_points: List[StereoPoint] = field(default_factory=list)
    aggregated: StereoPoint = field(default_factory=StereoPoint)

    def valid(self) -> bool:
        return len(self.stereo_points) > self.MIN_TRACKING_POINTS

    def aggregate_median(self):
        left_points = [p.left.as_np() for p in self.stereo_points if p.left.valid()]
        right_points = [p.right.as_np() for p in self.stereo_points if p.right.valid()]

        if left_points:
            median = np.median(left_points, axis=0)
            self.aggregated.left = Point2D(*median)

        if right_points:
            median = np.median(right_points, axis=0)
            self.aggregated.right = Point2D(*median)

@dataclass
class TrackingData:
    left: StereoPointCluster = field(default_factory=StereoPointCluster)
    right: StereoPointCluster = field(default_factory=StereoPointCluster)
    aggregated: StereoPoint = field(default_factory=StereoPoint)

    def valid(self) -> bool: 
        return self.left.valid() and self.right.valid()
    
    def aggregate_median(self):
        left_cam_points = []
        right_cam_points = []

        if self.left.valid():
            self.left.aggregate_median()
            left_cam_points.append(self.left.aggregated.left.as_np())
            right_cam_points.append(self.left.aggregated.right.as_np())

        if self.right.valid():
            self.right.aggregate_median()
            left_cam_points.append(self.right.aggregated.left.as_np())
            right_cam_points.append(self.right.aggregated.right.as_np())

        if left_cam_points:
            median_left = np.median(left_cam_points, axis=0)
            self.aggregated.left = Point2D(*median_left)

        if right_cam_points:
            median_right = np.median(right_cam_points, axis=0)
            self.aggregated.right = Point2D(*median_right)
