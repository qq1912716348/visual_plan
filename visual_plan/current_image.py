"""软件里同时只有一张当前图像。"""

from pathlib import Path

import numpy as np


class CurrentImage:
    def __init__(self) -> None:
        self.array: np.ndarray | None = None
        self.source: Path | None = None

    def replace(self, array: np.ndarray, source: Path) -> None:
        """读入成功后，当前图像换成这一张。"""
        self.array = array.copy()
        self.source = Path(source)

    def adopt(self, array: np.ndarray) -> None:
        """当前图像换成这一张，来源文件不变。"""
        self.array = array.copy()
