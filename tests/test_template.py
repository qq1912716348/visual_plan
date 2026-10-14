"""模板预览：最长边缩成 200，浮在右边画面的右上角。"""

import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np
from PyQt6.QtWidgets import QApplication

from visual_plan.template import apply_match, fit_long_edge
from visual_plan.window import MainWindow


class FitLongEdgeTests(unittest.TestCase):
    def test_wide_image_shrinks_until_the_width_is_200(self) -> None:
        image = np.full((100, 400), 7, np.uint8)
        image[0, 0] = 3
        result = fit_long_edge(image)
        self.assertEqual(result.shape, (50, 200))
        self.assertEqual(int(image[0, 0]), 3)
        result[0, 0] = 1
        self.assertEqual(int(image[0, 0]), 3)

    def test_tall_image_grows_until_the_height_is_200(self) -> None:
        image = np.zeros((40, 10, 3), np.uint8)
        image[0, 0] = (1, 2, 3)
        result = fit_long_edge(image)
        self.assertEqual(result.shape, (200, 50, 3))
        self.assertEqual(int(image[0, 0, 2]), 3)

    def test_square_uses_the_long_edge_on_both_sides(self) -> None:
        image = np.zeros((100, 100), np.uint8)
        result = fit_long_edge(image, 200)
        self.assertEqual(result.shape, (200, 200))
        self.assertIsNot(result, image)

    def test_a_thin_edge_stays_at_least_one_pixel(self) -> None:
        image = np.zeros((1, 1000), np.uint8)
        result = fit_long_edge(image)
        self.assertEqual(result.shape, (1, 200))


def _write_png(directory: Path, name: str, image: np.ndarray) -> Path:
    path = directory / name
    ok, encoded = cv2.imencode(".png", image)
    if not ok:
        raise RuntimeError(name)
    encoded.tofile(path)
    return path


class TemplatePreviewTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._app = QApplication.instance() or QApplication([])

    def test_template_button_follows_contours(self) -> None:
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
        window.close()

    def test_preview_uses_the_long_edge_and_close_drops_it(self) -> None:
        image = np.full((80, 400), 9, np.uint8)
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            path = _write_png(folder, "template.png", image)
            window = MainWindow()
            self.assertTrue(window.open_template(path))
            self.assertFalse(window._template_card.isHidden())
            pixmap = window._template_card.view.pixmap()
            self.assertIsNotNone(pixmap)
            assert pixmap is not None
            self.assertEqual(pixmap.width(), 200)
            self.assertEqual(pixmap.height(), 40)
            self.assertEqual(window._template.shape, (80, 400))
            self.assertIsNone(window._result)
            self.assertFalse(window.open_template(folder / "missing.png"))
            self.assertEqual(window._template.shape, (80, 400))
            self.assertFalse(window._template_card.isHidden())
            window.close_template()
            self.assertTrue(window._template_card.isHidden())
            self.assertIsNone(window._template)
            window.close()


class TemplateMatchTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._app = QApplication.instance() or QApplication([])

    def test_run_sits_to_the_left_of_close(self) -> None:
        image = np.zeros((20, 30), np.uint8)
        with tempfile.TemporaryDirectory() as directory:
            path = _write_png(Path(directory), "template.png", image)
            window = MainWindow()
            self.assertTrue(window.open_template(path))
            row = window._template_card.layout().itemAt(0).layout()
            labels = []
            for index in range(row.count()):
                widget = row.itemAt(index).widget()
                if widget is not None:
                    labels.append(widget.text())
            self.assertEqual(labels, ["运行", "关闭"])
            window.close()

    def test_best_match_is_boxed_and_scored(self) -> None:
        scene = np.zeros((40, 50), np.uint8)
        patch = np.arange(12, dtype=np.uint8).reshape(3, 4)
        scene[10:13, 20:24] = patch
        result, code = apply_match(scene, patch, {"mode": "ccoeff_normed"})
        self.assertEqual(result.shape, (40, 50, 3))
        self.assertIn("cv2.matchTemplate(img, template, cv2.TM_CCOEFF_NORMED)", code)
        self.assertIn("x, y = max_loc", code)
        self.assertIn("# 得分: 1.00", code)
        self.assertTrue(np.array_equal(result[10, 20], np.array([0, 255, 0], np.uint8)))
        self.assertEqual(int(scene[10, 20]), int(patch[0, 0]))
        self.assertEqual(int(patch[0, 0]), 0)

    def test_square_difference_uses_the_minimum(self) -> None:
        scene = np.zeros((30, 30), np.uint8)
        patch = np.full((4, 5), 40, np.uint8)
        scene[6:10, 8:13] = patch
        _result, code = apply_match(scene, patch, {"mode": "sqdiff"})
        self.assertIn("cv2.TM_SQDIFF", code)
        self.assertNotIn("TM_SQDIFF_NORMED", code)
        self.assertIn("x, y = min_loc", code)

    def test_template_must_be_smaller_than_the_image(self) -> None:
        image = np.zeros((10, 12), np.uint8)
        template = np.zeros((11, 8), np.uint8)
        result, message = apply_match(image, template, {"mode": "ccoeff_normed"})
        self.assertIsNone(result)
        self.assertIn("模板", message)
        self.assertEqual(image.shape, (10, 12))


if __name__ == "__main__":
    unittest.main()
