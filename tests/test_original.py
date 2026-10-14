"""当前图像和原图：处理结果可以接着用，原图可以回来。"""

import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np
from PyQt6.QtWidgets import QApplication

from visual_plan.window import EMPTY_RESULT, MainWindow


def _write_png(directory: Path, name: str, image: np.ndarray) -> Path:
    path = directory / name
    ok, encoded = cv2.imencode(".png", image)
    if not ok:
        raise RuntimeError(name)
    encoded.tofile(path)
    return path


class OriginalImageTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._app = QApplication.instance() or QApplication([])

    def test_buttons_sit_beside_the_pane_titles(self) -> None:
        window = MainWindow()
        self.assertEqual(window._original_pane.action_button.text(), "回到原图")
        self.assertEqual(window._result_pane.action_button.text(), "设为当前图像")
        self.assertFalse(window._original_pane.action_button.isEnabled())
        self.assertFalse(window._result_pane.action_button.isEnabled())
        window.close()

    def test_result_becomes_the_current_image_and_the_original_stays(self) -> None:
        image = np.zeros((4, 5), np.uint8)
        image[0, 0] = 9
        result = np.full((6, 7), 40, np.uint8)
        with tempfile.TemporaryDirectory() as directory:
            path = _write_png(Path(directory), "sample.png", image)
            window = MainWindow()
            self.assertTrue(window.open_path(path))
            self.assertFalse(window._original_pane.action_button.isEnabled())
            self.assertFalse(window._result_pane.action_button.isEnabled())

            window._result = result
            window._refresh()
            self.assertFalse(window._original_pane.action_button.isEnabled())
            self.assertTrue(window._result_pane.action_button.isEnabled())
            window._original_pane.view.set_roi((0, 0, 2, 2))

            window.use_result()
            self.assertEqual(window._current.array.shape, (6, 7))
            self.assertEqual(int(window._current.array[0, 0]), 40)
            result[0, 0] = 1
            self.assertEqual(int(window._current.array[0, 0]), 40)
            self.assertIsNone(window._result)
            self.assertEqual(window._result_pane.view.text(), EMPTY_RESULT)
            self.assertFalse(window._result_pane.action_button.isEnabled())
            self.assertTrue(window._original_pane.action_button.isEnabled())
            self.assertIsNone(window._original_pane.view.roi())

            window._current.array[0, 0] = 8
            window.restore_original()
            self.assertEqual(window._current.array.shape, (4, 5))
            self.assertEqual(int(window._current.array[0, 0]), 9)
            self.assertIsNone(window._result)
            self.assertFalse(window._original_pane.action_button.isEnabled())
            self.assertEqual(window._current.source, path)
            window.close()

    def test_restore_does_nothing_while_the_current_image_is_the_original(self) -> None:
        image = np.zeros((3, 3), np.uint8)
        image[1, 1] = 12
        result = np.full((3, 3), 5, np.uint8)
        with tempfile.TemporaryDirectory() as directory:
            path = _write_png(Path(directory), "sample.png", image)
            window = MainWindow()
            self.assertTrue(window.open_path(path))
            window._result = result
            window._result_note = "Otsu 阈值: 7"
            window._refresh()
            window.restore_original()
            self.assertEqual(int(window._current.array[1, 1]), 12)
            self.assertIs(window._result, result)
            self.assertEqual(window._result_note, "Otsu 阈值: 7")
            self.assertTrue(window._result_pane.action_button.isEnabled())
            window.close()

    def test_a_new_read_replaces_the_original(self) -> None:
        first = np.zeros((2, 2), np.uint8)
        first[0, 0] = 3
        second = np.zeros((2, 2), np.uint8)
        second[0, 0] = 6
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            first_path = _write_png(folder, "first.png", first)
            second_path = _write_png(folder, "second.png", second)
            window = MainWindow()
            self.assertTrue(window.open_path(first_path))
            window._result = np.full((2, 2), 9, np.uint8)
            window.use_result()
            self.assertFalse(window.open_path(folder / "missing.png"))
            self.assertEqual(int(window._current.array[0, 0]), 9)
            self.assertTrue(window.open_path(second_path))
            self.assertEqual(int(window._current.array[0, 0]), 6)
            self.assertIsNone(window._result)
            self.assertFalse(window._original_pane.action_button.isEnabled())
            window._result = np.full((2, 2), 1, np.uint8)
            window.use_result()
            window.restore_original()
            self.assertEqual(int(window._current.array[0, 0]), 6)
            window.close()


if __name__ == "__main__":
    unittest.main()
