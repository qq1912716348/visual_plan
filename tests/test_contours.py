"""轮廓检测：二值化之后查找轮廓，并用绿线画出来。"""

import unittest

import numpy as np
from PyQt6.QtWidgets import QApplication

from visual_plan.operations import CONTOURS, apply_contour
from visual_plan.window import MainWindow


def _square(top: int, bottom: int, left: int, right: int, size: int = 60) -> np.ndarray:
    image = np.zeros((size, size), np.uint8)
    image[top:bottom, left:right] = 255
    return image


class ContourOperationTests(unittest.TestCase):
    def test_menu_lists_find_contours(self) -> None:
        self.assertEqual([item.label for item in CONTOURS], ["查找轮廓"])

    def test_outer_contour_is_drawn_in_green_and_counted(self) -> None:
        image = _square(10, 30, 10, 30, 40)
        params = {"thresh": 127, "mode": "external", "method": "simple", "thickness": 2, "min_area": 0}
        result, code = apply_contour("find", image, params)
        self.assertEqual(result.shape, (40, 40, 3))
        self.assertEqual(
            code,
            "_, binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)\n"
            "contours, hierarchy = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)\n"
            "drawn = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)\n"
            "cv2.drawContours(drawn, contours, -1, (0, 255, 0), 2)\n"
            "# 轮廓数: 1",
        )
        green = (result[:, :, 1] == 255) & (result[:, :, 0] == 0) & (result[:, :, 2] == 0)
        self.assertGreater(int(green.sum()), 0)
        self.assertTrue(np.array_equal(result[0, 0], np.array([0, 0, 0], np.uint8)))
        self.assertEqual(int(image[15, 15]), 255)

    def test_color_image_is_thresholded_in_gray_and_drawn_on_a_copy(self) -> None:
        image = np.zeros((30, 30, 3), np.uint8)
        image[8:22, 8:22] = (80, 255, 200)
        params = {"thresh": 127, "mode": "tree", "method": "none", "thickness": 2, "min_area": 0}
        result, code = apply_contour("find", image, params)
        self.assertEqual(result.shape, image.shape)
        self.assertIn("gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)", code)
        self.assertIn("cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)", code)
        self.assertIn("drawn = img.copy()", code)
        self.assertIn("# 轮廓数: 1", code)
        self.assertEqual(int(image[10, 10, 2]), 200)
        self.assertFalse(np.array_equal(result, image))

    def test_a_hole_is_kept_only_when_the_search_asks_for_it(self) -> None:
        image = _square(10, 50, 10, 50)
        image[25:35, 25:35] = 0
        outer = {"thresh": 127, "mode": "external", "method": "simple", "thickness": 2, "min_area": 0}
        tree = {**outer, "mode": "tree"}
        _outer_image, outer_code = apply_contour("find", image, outer)
        _tree_image, tree_code = apply_contour("find", image, tree)
        self.assertIn("# 轮廓数: 1", outer_code)
        self.assertIn("# 轮廓数: 2", tree_code)
        self.assertEqual(int(image[30, 30]), 0)

    def test_small_contours_can_be_dropped_by_area(self) -> None:
        image = np.zeros((50, 50), np.uint8)
        image[2:5, 2:5] = 255
        image[15:40, 15:40] = 255
        params = {"thresh": 127, "mode": "external", "method": "simple", "thickness": 1, "min_area": 50}
        _result, code = apply_contour("find", image, params)
        self.assertIn("cv2.contourArea(contour) >= 50", code)
        self.assertIn("# 轮廓数: 1", code)
        self.assertEqual(int(image[3, 3]), 255)


class ContourMenuTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._app = QApplication.instance() or QApplication([])

    def test_contour_button_is_on_the_header(self) -> None:
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
        self.assertEqual(window._contour_list.operation_ids(), ["find"])
        window.close()


if __name__ == "__main__":
    unittest.main()
