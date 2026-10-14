"""阈值处理：简单阈值、自适应阈值、Otsu、灰度直方图、悬停灰度值。"""

import unittest

import cv2
import numpy as np
from PyQt6.QtCore import QEvent, QPoint, QPointF, Qt
from PyQt6.QtGui import QMouseEvent, QPixmap
from PyQt6.QtWidgets import QApplication

from visual_plan.image_view import ImageView
from visual_plan.operations import THRESHOLDS, apply_threshold, gray_at
from visual_plan.window import MainWindow


class ThresholdOperationTests(unittest.TestCase):
    def test_menu_lists_four_tools(self) -> None:
        self.assertEqual(
            [item.label for item in THRESHOLDS],
            ["简单阈值处理", "自适应阈值处理", "Otsu 阈值处理", "灰度直方图", "直方图均衡化", "自适应直方图均衡化", "显示灰度值"],
        )

    def test_simple_threshold_splits_gray_values(self) -> None:
        image = np.array([[0, 200], [127, 128]], np.uint8)
        result, code = apply_threshold("simple", image, {"thresh": 127, "maxval": 255, "kind": "binary"})
        self.assertTrue(np.array_equal(result, np.array([[0, 255], [0, 255]], np.uint8)))
        self.assertEqual(code, "_, binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)")
        self.assertEqual(int(image[0, 1]), 200)

    def test_color_image_is_converted_before_the_threshold(self) -> None:
        image = np.zeros((4, 4, 3), np.uint8)
        image[:, :] = (10, 20, 180)
        result, code = apply_threshold("simple", image, {"thresh": 127, "maxval": 255, "kind": "binary"})
        self.assertEqual(result.shape, (4, 4))
        self.assertIn("gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)", code)
        self.assertIn("cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)", code)
        self.assertEqual(int(image[0, 0, 2]), 180)

    def test_adaptive_block_must_be_an_odd_number(self) -> None:
        image = np.zeros((16, 16), np.uint8)
        result, message = apply_threshold(
            "adaptive",
            image,
            {"maxval": 255, "method": "mean", "kind": "binary", "block": 4, "c": 2},
        )
        self.assertIsNone(result)
        self.assertIn("块大小", message)

    def test_adaptive_matches_opencv(self) -> None:
        image = np.tile(np.linspace(0, 255, 32, dtype=np.uint8), (32, 1))
        params = {"maxval": 255, "method": "gaussian", "kind": "binary", "block": 11, "c": 2}
        result, code = apply_threshold("adaptive", image, params)
        expected = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
        self.assertTrue(np.array_equal(result, expected))
        self.assertIn("cv2.ADAPTIVE_THRESH_GAUSSIAN_C", code)
        self.assertIn("cv2.adaptiveThreshold(img, 255,", code)

    def test_otsu_reports_the_threshold_it_chose(self) -> None:
        image = np.zeros((20, 20), np.uint8)
        image[:, 10:] = 255
        result, code = apply_threshold("otsu", image, {"maxval": 255, "kind": "binary"})
        value, expected = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        self.assertTrue(np.array_equal(result, expected))
        self.assertIn("cv2.THRESH_BINARY + cv2.THRESH_OTSU", code)
        self.assertIn(f"# Otsu 阈值: {int(round(float(value)))}", code)
        self.assertEqual(int(image[0, 0]), 0)

    def test_histogram_of_a_black_image_rises_on_the_left(self) -> None:
        image = np.zeros((20, 20), np.uint8)
        result, code = apply_threshold("histogram", image, {})
        self.assertEqual(result.ndim, 2)
        ink = result < 60
        axis_y = int(np.argmax(ink.sum(axis=1)))
        axis_x = int(np.argmax(ink.sum(axis=0)))
        self.assertGreater(int(ink[axis_y].sum()), result.shape[1] // 3)
        self.assertGreater(int(ink[:, axis_x].sum()), result.shape[0] // 3)
        self.assertLess(int(result[axis_y + 8 :, :].min()), 80)
        self.assertLess(int(result[:axis_y, :axis_x].min()), 80)
        plot = result[:axis_y, axis_x + 1 :]
        self.assertLess(float(plot[:, :8].mean()), float(plot[:, -8:].mean()))
        self.assertEqual(code, "hist = cv2.calcHist([img], [0], None, [256], [0, 256])")
        self.assertEqual(int(image[0, 0]), 0)


class EqualizeTests(unittest.TestCase):
    def test_histogram_equalization_matches_opencv(self) -> None:
        image = np.tile(np.arange(256, dtype=np.uint8), (8, 1))
        result, code = apply_threshold("equalize", image, {})
        self.assertTrue(np.array_equal(result, cv2.equalizeHist(image)))
        self.assertEqual(code, "equalized = cv2.equalizeHist(img)")
        self.assertEqual(int(image[0, 0]), 0)

    def test_color_equalization_only_changes_brightness(self) -> None:
        image = np.zeros((16, 16, 3), np.uint8)
        image[:, :, 0] = 40
        image[:, :, 1] = 90
        image[:, :, 2] = np.tile(np.linspace(0, 255, 16, dtype=np.uint8), (16, 1))
        result, code = apply_threshold("equalize", image, {})
        yuv = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
        yuv[:, :, 0] = cv2.equalizeHist(yuv[:, :, 0])
        expected = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)
        self.assertTrue(np.array_equal(result, expected))
        self.assertEqual(result.shape, image.shape)
        self.assertIn("cv2.cvtColor(img, cv2.COLOR_BGR2YUV)", code)
        self.assertIn("cv2.equalizeHist(yuv[:, :, 0])", code)
        self.assertEqual(int(image[0, 0, 2]), 0)

    def test_adaptive_equalization_matches_opencv(self) -> None:
        image = np.tile(np.linspace(0, 255, 32, dtype=np.uint8), (32, 1))
        params = {"clip": 2.0, "tile": 8}
        result, code = apply_threshold("clahe", image, params)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        self.assertTrue(np.array_equal(result, clahe.apply(image)))
        self.assertEqual(
            code,
            "clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))\n"
            "equalized = clahe.apply(img)",
        )
        self.assertEqual(int(image[0, 0]), 0)


class ThresholdMenuTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._app = QApplication.instance() or QApplication([])

    def test_threshold_button_is_on_the_header(self) -> None:
        window = MainWindow()
        header = window.centralWidget().layout().itemAt(0).layout()
        labels = []
        for index in range(header.count()):
            widget = header.itemAt(index).widget()
            if widget is not None:
                labels.append(widget.text())
        self.assertEqual(
            labels,
            ["导入文件", "图像基本处理", "滤波处理", "形态学处理", "阈值处理", "边缘检测", "轮廓检测", "模板匹配"],
        )
        self.assertEqual(
            window._threshold_list.operation_ids(),
            ["simple", "adaptive", "otsu", "histogram", "equalize", "clahe", "gray_value"],
        )
        window.close()

    def test_gray_value_toggles_without_changing_the_image(self) -> None:
        window = MainWindow()
        window.choose_threshold("gray_value")
        self.assertTrue(window._original_pane.view.gray_readout())
        self.assertTrue(window._result_pane.view.gray_readout())
        self.assertEqual(window._threshold_list.label_of("gray_value"), "关闭灰度值")
        self.assertIsNone(window._current.array)
        self.assertIsNone(window._result)
        window.choose_threshold("gray_value")
        self.assertFalse(window._original_pane.view.gray_readout())
        self.assertFalse(window._result_pane.view.gray_readout())
        self.assertEqual(window._threshold_list.label_of("gray_value"), "显示灰度值")
        window.close()


class GrayReadoutTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._app = QApplication.instance() or QApplication([])

    def test_gray_at_reads_gray_and_color_pixels(self) -> None:
        gray = np.arange(12, dtype=np.uint8).reshape(3, 4)
        self.assertEqual(gray_at(gray, 2, 1), int(gray[1, 2]))
        color = np.zeros((2, 3, 3), np.uint8)
        color[1, 2] = (10, 20, 30)
        expected = int(cv2.cvtColor(color, cv2.COLOR_BGR2GRAY)[1, 2])
        self.assertEqual(gray_at(color, 2, 1), expected)
        self.assertEqual(int(color[1, 2, 0]), 10)

    def test_hover_shows_the_gray_value_only_while_enabled(self) -> None:
        view = ImageView()
        view.setMinimumSize(1, 1)
        view.setFixedSize(100, 80)
        image = np.zeros((8, 10), np.uint8)
        image[0, 0] = 42
        view.set_pixels(image)
        view.set_image_size(10, 8)
        pixmap = QPixmap(10, 8)
        pixmap.fill(Qt.GlobalColor.black)
        view.setPixmap(pixmap)
        view.show()
        self._app.processEvents()
        self._move(view, 45, 36)
        self.assertEqual(view.gray_text(), "")
        view.set_gray_readout(True)
        self._move(view, 45, 36)
        self.assertEqual(view.gray_text(), "灰度 42")
        self._move(view, 0, 0)
        self.assertEqual(view.gray_text(), "")
        view.set_gray_readout(False)
        self._move(view, 45, 36)
        self.assertEqual(view.gray_text(), "")
        self.assertEqual(int(image[0, 0]), 42)
        view.close()

    def _move(self, view: ImageView, x: int, y: int) -> None:
        local = QPointF(x, y)
        event = QMouseEvent(
            QEvent.Type.MouseMove,
            local,
            QPointF(view.mapToGlobal(QPoint(x, y))),
            Qt.MouseButton.NoButton,
            Qt.MouseButton.NoButton,
            Qt.KeyboardModifier.NoModifier,
        )
        self._app.sendEvent(view, event)


if __name__ == "__main__":
    unittest.main()
