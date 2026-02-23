from dataclasses import dataclass, field
from typing import List, Optional
import numpy as np

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

    def aggregate_mean(self):
        left_pts = [p.left.as_np() for p in self.stereoPoints if p.left.valid()]
        right_pts = [p.right.as_np() for p in self.stereoPoints if p.right.valid()]

        if left_pts:
            mean = np.mean(left_pts, axis=0)
            self.aggregated.left = Point2D(*mean)

        if right_pts:
            mean = np.mean(right_pts, axis=0)
            self.aggregated.right = Point2D(*mean)

