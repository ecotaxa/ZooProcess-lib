# ZooProcess v10
# Entry point for image processing. No clever storage, with naming convention and so on.
# Just pixels and files.
from typing import Optional

from .BgCombiner import BackgroundCombiner
from .BgRemover import BackgroundRemover
from .Converter import Converter
from .Extractor import Extractor
from .Features import FeaturesCalculator
from .LegacyConfig import Lut, ZooscanConfig
from .Segmenter import Segmenter


# toto


class Processor:
    """ZooProcess v10 image processor"""

    def __init__(
        self, config: Optional[ZooscanConfig] = None, lut: Optional[Lut] = None
    ):
        """Provide all needed configuration."""
        self.config = config
        self.lut = lut

    @classmethod
    def from_legacy_config(
        cls, config: Optional[ZooscanConfig], lut: Optional[Lut]
    ) -> "Processor":
        """Initialize from ZooProcess v8 config"""
        return cls(config, lut)

    @property
    def converter(self) -> Converter:
        return Converter(self.lut)

    @property
    def bg_combiner(self) -> BackgroundCombiner:
        return BackgroundCombiner()

    @property
    def bg_remover(self) -> BackgroundRemover:
        return BackgroundRemover(self.config.background_process)

    @property
    def segmenter(self) -> Segmenter:
        return Segmenter(
            self.config.minsizeesd_mm, self.config.maxsizeesd_mm, self.config.upper
        )

    @property
    def extractor(self) -> Extractor:
        return Extractor(self.config.longline_mm, self.config.upper)

    @property
    def calculator(self) -> FeaturesCalculator:
        return FeaturesCalculator(self.config.upper)
