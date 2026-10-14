"""把图像文件读成图像，或把图像写回文件。"""

from pathlib import Path

import cv2
import numpy as np


def read_image(path: Path) -> np.ndarray | None:
    """把图像文件读成图像，保持这份文件自身的颜色和尺寸。

    灰度文件仍是二维图像。这一课不把图像改成灰度，也不缩小。
    读不了时返回 None。
    """
    file_path = Path(path)
    if not file_path.is_file():
        return None
    try:
        # 路径交给 Python 读字节，再解码。这样中文路径也能读入。
        data = np.fromfile(file_path, dtype=np.uint8)
    except OSError:
        return None
    if data.size == 0:
        return None
    # IMREAD_UNCHANGED：通道和位数按文件本身来，尺寸也不缩小。
    image = cv2.imdecode(data, cv2.IMREAD_UNCHANGED)
    if image is None or image.size == 0:
        return None
    return image


_SAVE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


def save_image(array: np.ndarray, path: Path) -> bool:
    """把图像写成文件。路径交给 Python 写字节，这样中文路径也能保存。

    没有可用的后缀时写成 PNG。写不了时返回 False。
    """
    if array.size == 0:
        return False
    file_path = Path(path)
    suffix = file_path.suffix.lower()
    if suffix not in _SAVE_SUFFIXES:
        file_path = file_path.with_suffix(".png")
        suffix = ".png"
    try:
        ok, encoded = cv2.imencode(suffix, array)
    except cv2.error:
        return False
    if not ok:
        return False
    try:
        encoded.tofile(file_path)
    except OSError:
        return False
    return True


def image_facts(array: np.ndarray) -> tuple[int, int, int]:
    """图像的高度、宽度、通道数。二维图像的通道数是 1。"""
    height = int(array.shape[0])
    width = int(array.shape[1])
    if array.ndim == 2:
        channels = 1
    else:
        channels = int(array.shape[2])
    return height, width, channels
