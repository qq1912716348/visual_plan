"""把处理结果写成图像文件。"""

import tempfile
import unittest
from pathlib import Path

import numpy as np
from PyQt6.QtWidgets import QApplication

from visual_plan.reading import read_image, save_image
from visual_plan.window import MainWindow


class SaveImageTests(unittest.TestCase):
    def test_png_roundtrip_leaves_the_array_alone(self) -> None:
        image = np.zeros((4, 5), np.uint8)
        image[1, 2] = 9
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "结果.png"
            self.assertTrue(save_image(image, path))
            loaded = read_image(path)
            self.assertIsNotNone(loaded)
            assert loaded is not None
            self.assertTrue(np.array_equal(loaded, image))
            self.assertEqual(loaded.shape, (4, 5))
            self.assertEqual(int(image[1, 2]), 9)

    def test_color_image_with_a_chinese_name_roundtrips(self) -> None:
        image = np.zeros((3, 3, 3), np.uint8)
        image[0, 1] = (8, 9, 10)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "处理 结果.png"
            self.assertTrue(save_image(image, path))
            loaded = read_image(path)
            assert loaded is not None
            self.assertTrue(np.array_equal(loaded, image))
            self.assertEqual(int(image[0, 1, 2]), 10)

    def test_missing_suffix_is_written_as_png(self) -> None:
        image = np.zeros((2, 2), np.uint8)
        image[0, 0] = 4
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "out"
            self.assertTrue(save_image(image, path))
            self.assertFalse(path.is_file())
            loaded = read_image(path.with_suffix(".png"))
            assert loaded is not None
            self.assertEqual(int(loaded[0, 0]), 4)


class SaveResultTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._app = QApplication.instance() or QApplication([])

    def test_save_follows_the_result_and_keeps_it(self) -> None:
        window = MainWindow()
        self.assertEqual(window._result_pane.extra_button.text(), "保存")
        self.assertFalse(window._result_pane.extra_button.isEnabled())
        self.assertEqual(window._save_action.text(), "保存处理结果")
        self.assertFalse(window._save_action.isEnabled())
        image = np.zeros((2, 3), np.uint8)
        image[0, 1] = 6
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "kept.png"
            self.assertFalse(window.save_result_to(path))
            self.assertFalse(path.is_file())
            window._result = image
            window._refresh()
            self.assertTrue(window._result_pane.extra_button.isEnabled())
            self.assertTrue(window._save_action.isEnabled())
            self.assertTrue(window.save_result_to(path))
            loaded = read_image(path)
            assert loaded is not None
            self.assertTrue(np.array_equal(loaded, image))
            self.assertIs(window._result, image)
            self.assertEqual(int(image[0, 1]), 6)
        window.close()


if __name__ == "__main__":
    unittest.main()
