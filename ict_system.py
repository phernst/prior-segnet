from dataclasses import dataclass
from enum import IntEnum


class DetectorBinning(IntEnum):
    BINNING1x1 = 1
    BINNING2x2 = 2
    BINNING4x4 = 4


class ArtisQSystem:
    def __init__(self, detector_binning: DetectorBinning):
        self.nb_pixels = (2480//detector_binning, 1920//detector_binning)
        self.pixel_dims = (0.154*detector_binning, 0.154*detector_binning)
        self.carm_span = 1200.0  # mm


@dataclass
class LungsGESystem:
    voltage: float = 120.0  # kVp
    source_to_detector: float = 949.075012  # mm
    source_to_isocenter: float = 541.0  # mm
    focal_spot_size: float = 1.2  # mm
    tube_current: float = 420  # mA
    exposure: float = 210  # mAs, full scan
    exposure_time: float = 17100  # ms, full scan
    detector_rows: int = 16
    detector_columns: int = 888
    pixel_size: float = 0.625  # mm
