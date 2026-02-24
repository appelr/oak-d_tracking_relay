import numpy as np

from dataclasses import dataclass, field
from typing import List
from enum import Enum, auto

class TargetType(Enum):
    HEAD = auto()
    HANDS = auto()

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
    #position_3d: Point3D = field(default_factory=Point3D)

    def valid_2d(self) -> bool:
        return self.left.valid() and self.right.valid()

    # def valid_3d(self) -> bool:
    #     return self.position_3d.valid()

    def valid(self) -> bool:
        return self.valid_2d() #or self.valid_3d()
    
@dataclass
class StereoPointCluster:
    stereoPoints: List[StereoPoint] = field(default_factory=list)
    aggregated: StereoPoint = field(default_factory=StereoPoint)

    def valid(self) -> bool:
        return len(self.stereoPoints) > 0

    def aggregate_median(self):
        left_pts = [p.left.as_np() for p in self.stereoPoints if p.left.valid()]
        right_pts = [p.right.as_np() for p in self.stereoPoints if p.right.valid()]

        if left_pts:
            median = np.median(left_pts, axis=0)
            self.aggregated.left = Point2D(*median)

        if right_pts:
            median = np.median(right_pts, axis=0)
            self.aggregated.right = Point2D(*median)

@dataclass
class TrackingData:
    left: StereoPointCluster = field(default_factory=StereoPointCluster)
    right: StereoPointCluster = field(default_factory=StereoPointCluster)
    aggregated: StereoPoint = field(default_factory=StereoPoint)
    targetType: TargetType = TargetType.HEAD
    def valid(self) -> bool: 
            if self.targetType == TargetType.HEAD:
                return self.left.valid() and self.right.valid()
            elif self.targetType == TargetType.HANDS:
                return self.left.valid() or self.right.valid()
            
            return False
    
    def aggregate_median(self):
        left_cam_pts = []
        right_cam_pts = []

        if self.left.valid():
            self.left.aggregate_median()
            left_cam_pts.append(self.left.aggregated.left.as_np())
            right_cam_pts.append(self.left.aggregated.right.as_np())

        if self.right.valid():
            self.right.aggregate_median()
            left_cam_pts.append(self.right.aggregated.left.as_np())
            right_cam_pts.append(self.right.aggregated.right.as_np())

        # 3. Mittelpunkt für die LINKE Kamera berechnen
        if left_cam_pts:
            medianL = np.median(left_cam_pts, axis=0)
            self.aggregated.left = Point2D(*medianL)

        # 4. Mittelpunkt für die RECHTE Kamera berechnen
        if right_cam_pts:
            medianR = np.median(right_cam_pts, axis=0)
            self.aggregated.right = Point2D(*medianR)
