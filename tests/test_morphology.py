"""形态学处理和滤波按钮的名称。"""

import unittest

import cv2
import numpy as np
from PyQt6.QtWidgets import QApplication

from visual_plan.operations import MORPHOLOGY, apply_morphology, morphology_code
from visual_plan.window import MainWindow

_PARAMS = {"k": 3, "iterations": 1, "shape": "rect"}


def _square() -> np.ndarray:
    image = np.zeros((21, 21), np.uint8)
    image[8:13, 8:13] = 255
    image[1, 1] = 255
    return image


def _kernel() -> np.ndarray:
    return cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))


class MorphologyTests(unittest.TestCase):
    def test_lists_the_seven_operations(self) -> None:
        self.assertEqual(
            [item.label for item in MORPHOLOGY],
            ["腐蚀", "膨胀", "开运算", "闭运算", "形态学梯度", "顶帽运算", "黑帽运算"],
        )

    def test_erode_shrinks_foreground_and_shows_code(self) -> None:
        image = _square()
        result, code = apply_morphology("erode", image, _PARAMS)
        self.assertIsNotNone(result)
        assert result is not None
        self.assertLess(int(result.sum()), int(image.sum()))
        self.assertEqual(int(image[1, 1]), 255)
        self.assertIn("cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))", code)
        self.assertIn("eroded = cv2.erode(img, kernel, iterations=1)", code)

    def test_operations_match_their_definitions(self) -> None:
        image = _square()
        kernel = _kernel()
        eroded = cv2.erode(image, kernel, iterations=1)
        dilated = cv2.dilate(image, kernel, iterations=1)
        opened = cv2.dilate(eroded, kernel, iterations=1)
        closed = cv2.erode(dilated, kernel, iterations=1)
        expected = {
            "dilate": dilated,
            "open": opened,
            "close": closed,
            "gradient": cv2.subtract(dilated, eroded),
            "tophat": cv2.subtract(image, opened),
            "blackhat": cv2.subtract(closed, image),
        }
        for morph_id, want in expected.items():
            result, _code = apply_morphology(morph_id, image, _PARAMS)
            self.assertIsNotNone(result, morph_id)
            self.assertTrue(np.array_equal(result, want), morph_id)

    def test_shape_and_iterations_appear_in_code(self) -> None:
        code = morphology_code("open", {"k": 5, "iterations": 2, "shape": "ellipse"}, 1)
        self.assertIn("cv2.MORPH_ELLIPSE", code)
        self.assertIn("cv2.MORPH_OPEN", code)
        self.assertIn("iterations=2", code)

    def test_bad_kernel_does_not_change_the_image(self) -> None:
        image = _square()
        result, message = apply_morphology("erode", image, {"k": 0, "iterations": 1, "shape": "rect"})
        self.assertIsNone(result)
        self.assertIn("核大小", message)
        self.assertEqual(int(image[8, 8]), 255)


class WindowMenuTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._app = QApplication.instance() or QApplication([])

    def test_filter_menu_is_renamed_and_morphology_sits_beside_it(self) -> None:
        window = MainWindow()
        header = window.centralWidget().layout().itemAt(0).layout()
        labels = []
        for index in range(header.count()):
            widget = header.itemAt(index).widget()
            if widget is not None:
                labels.append(widget.text())
        self.assertEqual(labels, ["导入文件", "图像基本处理", "滤波处理", "形态学处理", "阈值处理", "边缘检测", "轮廓检测", "模板匹配"])
        self.assertEqual(window._filter_list.operation_ids(), ["gaussian", "mean", "median", "bilateral"])
        self.assertEqual(
            window._morph_list.operation_ids(),
            ["erode", "dilate", "open", "close", "gradient", "tophat", "blackhat"],
        )
        window.close()


if __name__ == "__main__":
    unittest.main()
