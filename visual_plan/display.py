"""把当前图像画成画面。画面可以缩放，当前图像不变。"""

import numpy as np
from PyQt6.QtGui import QImage


def to_qimage(array: np.ndarray) -> QImage:
    """复制一份给窗口看的图像。内存里的当前图像仍保持读入时的通道顺序。"""
    shown = _as_uint8(array)
    if shown.ndim == 3 and shown.shape[2] == 1:
        shown = shown[:, :, 0]
    if shown.ndim == 3 and shown.shape[2] == 3:
        shown = np.ascontiguousarray(shown[:, :, ::-1])
        image_format = QImage.Format.Format_RGB888
    elif shown.ndim == 3 and shown.shape[2] == 4:
        shown = np.ascontiguousarray(shown[:, :, [2, 1, 0, 3]])
        image_format = QImage.Format.Format_RGBA8888
    else:
        if shown.ndim == 3:
            shown = shown[:, :, 0]
        shown = np.ascontiguousarray(shown)
        height, width = shown.shape
        image = QImage(shown.data, width, height, shown.strides[0], QImage.Format.Format_Grayscale8)
        return image.copy()

    height, width = shown.shape[:2]
    image = QImage(shown.data, width, height, shown.strides[0], image_format)
    return image.copy()


def _as_uint8(array: np.ndarray) -> np.ndarray:
    if array.dtype == np.uint8:
        return array
    values = array.astype(np.float32)
    low = float(values.min())
    high = float(values.max())
    if high > low:
        values = (values - low) * (255.0 / (high - low))
    else:
        values = np.zeros_like(values)
    return values.astype(np.uint8)
