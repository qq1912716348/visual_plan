"""图像基本处理：缩放、旋转、平移、翻转，以及划定 ROI。"""

import unittest
from pathlib import Path
import tempfile

import cv2
import numpy as np
from PyQt6.QtCore import QEvent, QPoint, QPointF, Qt
from PyQt6.QtGui import QMouseEvent, QPixmap
from PyQt6.QtWidgets import QApplication

from visual_plan.image_view import ImageView
from visual_plan.operations import BASICS, apply_basic, basic_code
from visual_plan.roi import hit_handle, move_box, normalize_box, resize_box, view_to_image
from visual_plan.window import MainWindow


class BasicOperationTests(unittest.TestCase):
    def test_menu_lists_the_table_and_roi(self) -> None:
        self.assertEqual(
            [item.label for item in BASICS],
            ["图像缩放", "图像旋转", "图像平移", "图像翻转", "划定 ROI", "转为灰度图"],
        )

    def test_resize_changes_size_and_leaves_the_source(self) -> None:
        image = np.zeros((4, 5), np.uint8)
        image[0, 0] = 9
        result, code = apply_basic("resize", image, {"width": 10, "height": 6})
        self.assertIsNotNone(result)
        assert result is not None
        self.assertEqual(result.shape, (6, 10))
        self.assertEqual(int(image[0, 0]), 9)
        self.assertEqual(code, "resized = cv2.resize(img, (10, 6))")

    def test_rotate_and_translate_code_use_the_image_size(self) -> None:
        image = np.zeros((4, 5, 3), np.uint8)
        _rotated, rotate_code = apply_basic("rotate", image, {"angle": 45, "scale": 1.0})
        self.assertIn("cv2.getRotationMatrix2D(center, 45, 1.0)", rotate_code)
        self.assertIn("cv2.warpAffine(img, M, (5, 4))", rotate_code)
        _translated, translate_code = apply_basic("translate", image, {"dx": 3, "dy": -1})
        self.assertIn("T = np.float32([[1, 0, 3], [0, 1, -1]])", translate_code)
        self.assertIn("cv2.warpAffine(img, T, (5, 4))", translate_code)
        self.assertEqual(image.shape, (4, 5, 3))

    def test_flip_horizontal_swaps_columns(self) -> None:
        image = np.array([[1, 2], [3, 4]], np.uint8)
        result, code = apply_basic("flip", image, {"code": "1"})
        self.assertTrue(np.array_equal(result, np.array([[2, 1], [4, 3]], np.uint8)))
        self.assertEqual(code, "flipped = cv2.flip(img, 1)")
        self.assertEqual(int(image[0, 0]), 1)

    def test_roi_is_an_independent_slice(self) -> None:
        image = np.arange(20, dtype=np.uint8).reshape(4, 5)
        result, code = apply_basic("select_roi", image, {"y1": 1, "y2": 3, "x1": 2, "x2": 5})
        self.assertIsNotNone(result)
        assert result is not None
        self.assertTrue(np.array_equal(result, image[1:3, 2:5]))
        self.assertEqual(code, "roi = img[1:3, 2:5].copy()")
        result[0, 0] = 0
        self.assertNotEqual(int(image[1, 2]), 0)

    def test_empty_roi_is_rejected(self) -> None:
        image = np.zeros((4, 5), np.uint8)
        result, message = apply_basic("select_roi", image, {"y1": 2, "y2": 2, "x1": 0, "x2": 3})
        self.assertIsNone(result)
        self.assertIn("上沿", message)

    def test_code_preview_matches_apply(self) -> None:
        params = {"y1": 0, "y2": 2, "x1": 1, "x2": 4}
        image = np.zeros((4, 5), np.uint8)
        _result, applied = apply_basic("select_roi", image, params)
        self.assertEqual(applied, basic_code("select_roi", params, image.shape[:2]))

    def test_color_image_becomes_gray_in_one_step(self) -> None:
        image = np.zeros((4, 5, 3), np.uint8)
        image[1, 2] = (10, 20, 30)
        result, code = apply_basic("to_gray", image, {})
        self.assertIsNotNone(result)
        assert result is not None
        self.assertTrue(np.array_equal(result, cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)))
        self.assertEqual(result.shape, (4, 5))
        self.assertEqual(code, "gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)")
        self.assertEqual(int(image[1, 2, 2]), 30)

    def test_gray_image_is_copied(self) -> None:
        image = np.zeros((3, 3), np.uint8)
        image[0, 1] = 8
        result, code = apply_basic("to_gray", image, {})
        self.assertIsNotNone(result)
        assert result is not None
        self.assertTrue(np.array_equal(result, image))
        self.assertEqual(code, "gray = img.copy()")
        result[0, 1] = 1
        self.assertEqual(int(image[0, 1]), 8)


class RoiGeometryTests(unittest.TestCase):
    def test_drag_maps_onto_pixels_and_includes_the_end(self) -> None:
        self.assertIsNone(view_to_image(44, 36, 45, 36, 10, 8, 10, 8))
        self.assertEqual(view_to_image(45, 36, 45, 36, 10, 8, 10, 8), (0, 0))
        self.assertEqual(view_to_image(49, 39, 45, 36, 10, 8, 10, 8), (4, 3))
        self.assertEqual(normalize_box(4, 3, 0, 0, 10, 8), (0, 0, 5, 4))
        self.assertEqual(normalize_box(9, 7, 9, 7, 10, 8), (9, 7, 10, 8))

    def test_box_can_move_and_resize_inside_the_image(self) -> None:
        self.assertEqual(move_box(2, 3, 6, 7, 1, -1, 10, 10), (3, 2, 7, 6))
        self.assertEqual(move_box(0, 0, 2, 2, -5, 0, 10, 10), (0, 0, 2, 2))
        self.assertEqual(resize_box(2, 3, 6, 7, "e", 8, 4, 10, 10), (2, 3, 9, 7))
        self.assertEqual(resize_box(2, 3, 6, 7, "nw", 0, 1, 10, 10), (0, 1, 6, 7))
        self.assertEqual(hit_handle(10, 10, 10, 10, 40, 40), "nw")
        self.assertEqual(hit_handle(25, 10, 10, 10, 40, 40), "n")
        self.assertEqual(hit_handle(25, 25, 10, 10, 40, 40), "move")
        self.assertEqual(hit_handle(0, 0, 10, 10, 40, 40), "")


class RoiDragTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._app = QApplication.instance() or QApplication([])

    def test_press_and_drag_sets_the_box(self) -> None:
        view = ImageView()
        view.setMinimumSize(1, 1)
        view.setFixedSize(100, 80)
        view.set_image_size(10, 8)
        pixmap = QPixmap(10, 8)
        pixmap.fill(Qt.GlobalColor.black)
        view.setPixmap(pixmap)
        view.set_roi_active(True)
        view.show()
        self._app.processEvents()
        self._mouse(view, QEvent.Type.MouseButtonPress, 45, 36, Qt.MouseButton.LeftButton)
        self._mouse(view, QEvent.Type.MouseMove, 49, 39, Qt.MouseButton.LeftButton)
        self._mouse(view, QEvent.Type.MouseButtonRelease, 49, 39, Qt.MouseButton.NoButton)
        self.assertEqual(view.roi(), (0, 0, 5, 4))
        view.close()

    def test_edge_and_interior_drags_adjust_the_box(self) -> None:
        view = ImageView()
        view.setMinimumSize(1, 1)
        view.setFixedSize(200, 160)
        view.set_image_size(200, 160)
        pixmap = QPixmap(200, 160)
        pixmap.fill(Qt.GlobalColor.black)
        view.setPixmap(pixmap)
        view.set_roi((20, 30, 120, 110))
        view.set_roi_active(True)
        view.show()
        self._app.processEvents()
        rect = view._roi_view_rect()
        assert rect is not None
        edge_x = rect.right()
        edge_y = rect.center().y()
        self._mouse(view, QEvent.Type.MouseButtonPress, edge_x, edge_y, Qt.MouseButton.LeftButton)
        self._mouse(view, QEvent.Type.MouseMove, edge_x + 30, edge_y, Qt.MouseButton.LeftButton)
        self._mouse(view, QEvent.Type.MouseButtonRelease, edge_x + 30, edge_y, Qt.MouseButton.NoButton)
        self.assertEqual(view.roi(), (20, 30, 150, 110))
        center_x = rect.center().x()
        center_y = rect.center().y()
        self._mouse(view, QEvent.Type.MouseButtonPress, center_x, center_y, Qt.MouseButton.LeftButton)
        self._mouse(view, QEvent.Type.MouseMove, center_x + 10, center_y + 8, Qt.MouseButton.LeftButton)
        self._mouse(view, QEvent.Type.MouseButtonRelease, center_x + 10, center_y + 8, Qt.MouseButton.NoButton)
        self.assertEqual(view.roi(), (30, 38, 160, 118))
        view.close()

    def _mouse(self, view: ImageView, kind: QEvent.Type, x: int, y: int, buttons: Qt.MouseButton) -> None:
        button = Qt.MouseButton.LeftButton if kind != QEvent.Type.MouseMove else Qt.MouseButton.NoButton
        local = QPointF(x, y)
        event = QMouseEvent(kind, local, QPointF(view.mapToGlobal(QPoint(x, y))), button, buttons, Qt.KeyboardModifier.NoModifier)
        self._app.sendEvent(view, event)


class BasicMenuTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._app = QApplication.instance() or QApplication([])

    def test_basic_button_follows_import(self) -> None:
        window = MainWindow()
        header = window.centralWidget().layout().itemAt(0).layout()
        labels = []
        for index in range(header.count()):
            widget = header.itemAt(index).widget()
            if widget is not None:
                labels.append(widget.text())
        self.assertEqual(labels, ["导入文件", "图像基本处理", "滤波处理", "形态学处理", "阈值处理", "边缘检测", "轮廓检测", "模板匹配"])
        self.assertEqual(
            window._basic_list.operation_ids(),
            ["resize", "rotate", "translate", "flip", "select_roi", "to_gray"],
        )
        window.close()

    def test_choosing_gray_applies_it_immediately(self) -> None:
        image = np.zeros((4, 4, 3), np.uint8)
        image[:, :] = (0, 0, 200)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "color.png"
            ok, encoded = cv2.imencode(".png", image)
            self.assertTrue(ok)
            encoded.tofile(path)
            window = MainWindow()
            self.assertTrue(window.open_path(path))
            window.choose_basic("to_gray")
            self.assertIsNotNone(window._result)
            assert window._result is not None
            self.assertEqual(window._result.ndim, 2)
            self.assertEqual(window._result.shape, (4, 4))
            window.close()


if __name__ == "__main__":
    unittest.main()
