# ZooProcess v10
# Entry point for image processing. No clever storage, with naming convention and so on.
# Just pixels and files.
from typing import Optional

from .BgCombiner import BackgroundCombiner
from .BgRemover import BackgroundRemover
from .Converter import Converter
from .LegacyConfig import Lut, ZooscanConfig


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
        return BackgroundRemover()