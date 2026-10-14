"""当前图像那一块画面。划定 ROI 时在图上拖出框，并拖边、拖角微调。打开灰度读数后，悬停显示该像素的灰度。"""

from PyQt6.QtCore import QPoint, QRect, Qt, pyqtSignal
from PyQt6.QtGui import QColor, QPainter, QPen
from PyQt6.QtWidgets import QLabel

from visual_plan.operations import gray_at
from visual_plan.roi import (
    hit_handle,
    image_to_view,
    move_box,
    normalize_box,
    resize_box,
    view_to_image,
    view_to_image_clamped,
)

_INK = "#1C211F"
_PAPER = "#F1F2EF"
_HANDLE = 5

_CURSORS = {
    "nw": Qt.CursorShape.SizeFDiagCursor,
    "se": Qt.CursorShape.SizeFDiagCursor,
    "ne": Qt.CursorShape.SizeBDiagCursor,
    "sw": Qt.CursorShape.SizeBDiagCursor,
    "n": Qt.CursorShape.SizeVerCursor,
    "s": Qt.CursorShape.SizeVerCursor,
    "e": Qt.CursorShape.SizeHorCursor,
    "w": Qt.CursorShape.SizeHorCursor,
    "move": Qt.CursorShape.SizeAllCursor,
}


class ImageView(QLabel):
    roi_changed = pyqtSignal(int, int, int, int)

    def __init__(self) -> None:
        super().__init__()
        self._image_width = 0
        self._image_height = 0
        self._roi: tuple[int, int, int, int] | None = None
        self._roi_active = False
        self._mode = ""
        self._anchor = (0, 0)
        self._press_box: tuple[int, int, int, int] | None = None
        self._pixels = None
        self._gray_readout = False
        self._gray_text = ""
        self._gray_anchor = QPoint()
        self.setMouseTracking(True)

    def set_pixels(self, array) -> None:
        if array is self._pixels:
            return
        self._pixels = array
        self._gray_text = ""
        self.update()

    def set_gray_readout(self, enabled: bool) -> None:
        self._gray_readout = enabled
        if not enabled:
            self._gray_text = ""
            self.update()

    def gray_readout(self) -> bool:
        return self._gray_readout

    def gray_text(self) -> str:
        return self._gray_text

    def set_image_size(self, width: int, height: int) -> None:
        self._image_width = width
        self._image_height = height

    def set_roi_active(self, active: bool) -> None:
        self._roi_active = active
        self._mode = ""
        self.setCursor(Qt.CursorShape.CrossCursor if active else Qt.CursorShape.ArrowCursor)
        self.update()

    def set_roi(self, box: tuple[int, int, int, int] | None) -> None:
        self._roi = box
        self.update()

    def roi(self) -> tuple[int, int, int, int] | None:
        return self._roi

    def paintEvent(self, event) -> None:
        super().paintEvent(event)
        painter = QPainter(self)
        rect = self._roi_view_rect()
        if rect is not None:
            painter.setPen(QPen(QColor(_INK), 1))
            painter.drawRect(rect.adjusted(-1, -1, 1, 1))
            painter.setPen(QPen(QColor(_PAPER), 1))
            painter.drawRect(rect)
            if self._roi_active:
                painter.setBrush(QColor(_PAPER))
                for point in self._handle_points(rect):
                    painter.drawRect(point.x() - _HANDLE // 2, point.y() - _HANDLE // 2, _HANDLE, _HANDLE)
                painter.setBrush(Qt.BrushStyle.NoBrush)
        self._paint_gray(painter)
        painter.end()

    def _paint_gray(self, painter: QPainter) -> None:
        if not self._gray_text:
            return
        painter.setFont(self.font())
        metrics = painter.fontMetrics()
        pad_x, pad_y = 6, 3
        box_width = metrics.horizontalAdvance(self._gray_text) + pad_x * 2
        box_height = metrics.height() + pad_y * 2
        x = self._gray_anchor.x() + 14
        y = self._gray_anchor.y() - box_height - 8
        if x + box_width > self.width() - 2:
            x = max(2, self._gray_anchor.x() - box_width - 8)
        if y < 2:
            y = min(max(2, self.height() - box_height - 2), self._gray_anchor.y() + 16)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.fillRect(x, y, box_width, box_height, QColor(_PAPER))
        painter.setPen(QPen(QColor(_INK), 1))
        painter.drawRect(x, y, box_width - 1, box_height - 1)
        painter.drawText(x + pad_x, y + pad_y + metrics.ascent(), self._gray_text)

    def mousePressEvent(self, event) -> None:
        if not self._roi_active or event.button() != Qt.MouseButton.LeftButton:
            super().mousePressEvent(event)
            return
        point = event.position().toPoint()
        if self._image_point(point) is None:
            return
        located = self._image_point_clamped(point)
        if located is None:
            return
        handle = self._handle_at(point)
        if handle == "move" and self._roi is not None:
            self._mode = "move"
            self._anchor = located
            self._press_box = self._roi
        elif handle and self._roi is not None:
            self._mode = handle
            self._press_box = self._roi
        else:
            self._mode = "draw"
            self._anchor = located
            self._set_user_roi(normalize_box(*located, *located, self._image_width, self._image_height))
        event.accept()

    def leaveEvent(self, event) -> None:
        self._gray_text = ""
        self.update()
        super().leaveEvent(event)

    def mouseMoveEvent(self, event) -> None:
        point = event.position().toPoint()
        self._track_gray(point)
        if self._roi_active and self._mode and event.buttons() & Qt.MouseButton.LeftButton:
            located = self._image_point_clamped(point)
            if located is not None:
                self._update_drag(*located)
            event.accept()
            return
        if self._roi_active:
            handle = self._handle_at(point)
            self.setCursor(_CURSORS.get(handle, Qt.CursorShape.CrossCursor))
        super().mouseMoveEvent(event)

    def _track_gray(self, point: QPoint) -> None:
        text = ""
        if self._gray_readout and self._pixels is not None:
            located = self._image_point(point)
            if located is not None:
                value = gray_at(self._pixels, located[0], located[1])
                if value is not None:
                    text = f"灰度 {value}"
        if text == self._gray_text and (not text or point == self._gray_anchor):
            return
        self._gray_text = text
        self._gray_anchor = point
        self.update()

    def mouseReleaseEvent(self, event) -> None:
        if self._mode and event.button() == Qt.MouseButton.LeftButton:
            self._mode = ""
            event.accept()
            return
        super().mouseReleaseEvent(event)

    def _update_drag(self, x: int, y: int) -> None:
        if self._mode == "draw":
            box = normalize_box(*self._anchor, x, y, self._image_width, self._image_height)
        elif self._mode == "move" and self._press_box is not None:
            dx = x - self._anchor[0]
            dy = y - self._anchor[1]
            box = move_box(*self._press_box, dx, dy, self._image_width, self._image_height)
        elif self._press_box is not None:
            box = resize_box(*self._press_box, self._mode, x, y, self._image_width, self._image_height)
        else:
            return
        self._set_user_roi(box)

    def _set_user_roi(self, box: tuple[int, int, int, int]) -> None:
        self._roi = box
        self.update()
        self.roi_changed.emit(*box)

    def _pixmap_rect(self) -> QRect | None:
        pixmap = self.pixmap()
        if pixmap is None or pixmap.isNull() or self._image_width < 1 or self._image_height < 1:
            return None
        area = self.contentsRect()
        x = area.x() + max(0, (area.width() - pixmap.width()) // 2)
        y = area.y() + max(0, (area.height() - pixmap.height()) // 2)
        return QRect(x, y, pixmap.width(), pixmap.height())

    def _image_point(self, point: QPoint) -> tuple[int, int] | None:
        rect = self._pixmap_rect()
        if rect is None:
            return None
        return view_to_image(
            point.x(),
            point.y(),
            rect.x(),
            rect.y(),
            rect.width(),
            rect.height(),
            self._image_width,
            self._image_height,
        )

    def _image_point_clamped(self, point: QPoint) -> tuple[int, int] | None:
        rect = self._pixmap_rect()
        if rect is None:
            return None
        return view_to_image_clamped(
            point.x(),
            point.y(),
            rect.x(),
            rect.y(),
            rect.width(),
            rect.height(),
            self._image_width,
            self._image_height,
        )

    def _roi_view_rect(self) -> QRect | None:
        if self._roi is None:
            return None
        placed = self._pixmap_rect()
        if placed is None:
            return None
        x1, y1, x2, y2 = self._roi
        left, top = image_to_view(
            x1, y1, placed.x(), placed.y(), placed.width(), placed.height(), self._image_width, self._image_height
        )
        right, bottom = image_to_view(
            x2, y2, placed.x(), placed.y(), placed.width(), placed.height(), self._image_width, self._image_height
        )
        return QRect(left, top, max(1, right - left), max(1, bottom - top))

    def _handle_at(self, point: QPoint) -> str:
        rect = self._roi_view_rect()
        if rect is None:
            return ""
        return hit_handle(point.x(), point.y(), rect.left(), rect.top(), rect.right(), rect.bottom())

    def _handle_points(self, rect: QRect) -> tuple[QPoint, ...]:
        return (
            rect.topLeft(),
            QPoint(rect.center().x(), rect.top()),
            rect.topRight(),
            QPoint(rect.left(), rect.center().y()),
            QPoint(rect.right(), rect.center().y()),
            rect.bottomLeft(),
            QPoint(rect.center().x(), rect.bottom()),
            rect.bottomRight(),
        )
