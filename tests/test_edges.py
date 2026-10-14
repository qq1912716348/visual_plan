"""边缘检测：Sobel、Scharr、Laplacian、Canny。"""

import unittest

import cv2
import numpy as np
from PyQt6.QtWidgets import QApplication

from visual_plan.operations import EDGES, apply_edge
from visual_plan.window import MainWindow


def _step() -> np.ndarray:
    image = np.zeros((32, 32), np.uint8)
    image[:, 16:] = 255
    return image


class EdgeOperationTests(unittest.TestCase):
    def test_menu_lists_four_detectors(self) -> None:
        self.assertEqual(
            [item.label for item in EDGES],
            ["Sobel 算子", "Scharr 算子", "Laplacian 算子", "Canny 边缘检测"],
        )

    def test_sobel_x_matches_opencv_and_keeps_the_source(self) -> None:
        image = _step()
        result, code = apply_edge("sobel", image, {"direction": "x", "ksize": 3})
        expected = cv2.convertScaleAbs(cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3))
        self.assertTrue(np.array_equal(result, expected))
        self.assertEqual(
            code,
            "sobel_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)\n"
            "sobel_x = cv2.convertScaleAbs(sobel_x)",
        )
        self.assertEqual(int(image[0, 0]), 0)
        self.assertGreater(int(result[:, 14:18].max()), int(result[:, :8].max()))

    def test_sobel_magnitude_uses_both_directions(self) -> None:
        image = _step()
        result, code = apply_edge("sobel", image, {"direction": "magnitude", "ksize": 3})
        gx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
        gy = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
        expected = cv2.convertScaleAbs(cv2.magnitude(gx, gy))
        self.assertTrue(np.array_equal(result, expected))
        self.assertIn("cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)", code)
        self.assertIn("cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)", code)
        self.assertIn("cv2.magnitude(sobel_x, sobel_y)", code)

    def test_sobel_kernel_must_be_1_3_5_or_7(self) -> None:
        image = _step()
        result, message = apply_edge("sobel", image, {"direction": "x", "ksize": 4})
        self.assertIsNone(result)
        self.assertIn("核大小", message)

    def test_scharr_matches_opencv(self) -> None:
        image = _step()
        result, code = apply_edge("scharr", image, {"direction": "y"})
        expected = cv2.convertScaleAbs(cv2.Scharr(image, cv2.CV_64F, 0, 1))
        self.assertTrue(np.array_equal(result, expected))
        self.assertEqual(
            code,
            "scharr_y = cv2.Scharr(img, cv2.CV_64F, 0, 1)\n"
            "scharr_y = cv2.convertScaleAbs(scharr_y)",
        )

    def test_laplacian_matches_opencv(self) -> None:
        image = _step()
        result, code = apply_edge("laplacian", image, {"ksize": 3})
        expected = cv2.convertScaleAbs(cv2.Laplacian(image, cv2.CV_64F, ksize=3))
        self.assertTrue(np.array_equal(result, expected))
        self.assertEqual(
            code,
            "lap = cv2.Laplacian(img, cv2.CV_64F, ksize=3)\n"
            "lap = cv2.convertScaleAbs(lap)",
        )

    def test_canny_keeps_the_step_and_drops_the_flat_regions(self) -> None:
        image = _step()
        result, code = apply_edge(
            "canny",
            image,
            {"low": 50, "high": 150, "aperture": 3, "norm": "l1"},
        )
        expected = cv2.Canny(image, 50, 150, apertureSize=3, L2gradient=False)
        self.assertTrue(np.array_equal(result, expected))
        self.assertEqual(code, "edges = cv2.Canny(img, 50, 150, apertureSize=3, L2gradient=False)")
        self.assertEqual(int(result[:, :8].max()), 0)
        self.assertEqual(int(result[:, 14:18].max()), 255)
        self.assertEqual(int(image[0, 31]), 255)

    def test_canny_low_threshold_cannot_exceed_the_high_one(self) -> None:
        image = _step()
        result, message = apply_edge(
            "canny",
            image,
            {"low": 200, "high": 80, "aperture": 3, "norm": "l2"},
        )
        self.assertIsNone(result)
        self.assertIn("低阈值", message)

    def test_color_image_is_converted_before_canny(self) -> None:
        image = np.zeros((16, 16, 3), np.uint8)
        image[:, 8:] = (0, 0, 255)
        result, code = apply_edge(
            "canny",
            image,
            {"low": 50, "high": 150, "aperture": 3, "norm": "l1"},
        )
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        expected = cv2.Canny(gray, 50, 150, apertureSize=3, L2gradient=False)
        self.assertTrue(np.array_equal(result, expected))
        self.assertIn("gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)", code)
        self.assertIn("cv2.Canny(gray, 50, 150,", code)
        self.assertEqual(int(image[0, 15, 2]), 255)


class EdgeMenuTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._app = QApplication.instance() or QApplication([])

    def test_edge_button_is_on_the_header(self) -> None:
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
        self.assertEqual(window._edge_list.operation_ids(), ["sobel", "scharr", "laplacian", "canny"])
        window.close()


if __name__ == "__main__":
    unittest.main()
