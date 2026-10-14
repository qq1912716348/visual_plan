"""左边显示当前图像，右边显示处理结果。悬停选择图像基本处理、滤波、形态学、阈值、边缘或轮廓检测，也可以导入模板。"""

from pathlib import Path

import numpy as np
from PyQt6.QtCore import QPoint, Qt, QTimer
from PyQt6.QtGui import QAction, QCursor, QFont, QPixmap
from PyQt6.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSizePolicy,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from visual_plan.current_image import CurrentImage
from visual_plan.display import to_qimage
from visual_plan.image_view import ImageView
from visual_plan.operations import (
    BASICS,
    CONTOURS,
    EDGES,
    FILTERS,
    MORPHOLOGY,
    THRESHOLDS,
    Operation,
    Param,
    apply_basic,
    apply_contour,
    apply_edge,
    apply_filter,
    apply_morphology,
    apply_threshold,
    basic_by_id,
    contour_by_id,
    edge_by_id,
    filter_by_id,
    morphology_by_id,
    operation_code,
    operation_error,
    threshold_by_id,
)
from visual_plan.reading import image_facts, read_image, save_image
from visual_plan.template import MATCH_MODES, fit_long_edge, match_code, apply_match

EMPTY_VIEW = "还没有图像"
EMPTY_FACTS = "先导入一个图像文件。"
EMPTY_RESULT = "还没有处理结果"
EMPTY_RESULT_FACTS = "选择一种处理，结果显示在这里。"


def _note_from_code(code: str) -> str:
    for line in code.splitlines():
        if line.startswith("# "):
            return line[2:]
    return ""
READ_FAILED = "无法读入这个文件。"
SAVE_FAILED = "无法保存这个文件。"
NEED_IMAGE = "先导入图像文件。"
IMAGE_FILTER = "图像文件 (*.png *.jpg *.jpeg *.bmp *.tif *.tiff *.webp);;所有文件 (*.*)"
SAVE_FILTER = "PNG (*.png);;JPEG (*.jpg *.jpeg);;BMP (*.bmp);;TIFF (*.tif *.tiff);;WebP (*.webp)"

MATCH = Operation(
    "match",
    "模板匹配",
    "template",
    (Param("mode", "相似度", "choice", "ccoeff_normed", choices=MATCH_MODES),),
    hint="在当前图像上滑动模板。归一化相关系数最常用，越接近 1 越像。平方差则是越小越像。",
)

PAPER = "#F1F2EF"
WELL = "#1E2622"
INK = "#1C211F"
NOTE = "#4A534E"
RULE = "#C9CFCA"
PRESS = "#E2E6E1"
FONT_UI = '"DengXian", "等线", "Microsoft YaHei UI", "微软雅黑"'
FONT_CODE = '"Cascadia Mono", "Consolas"'

WINDOW_STYLE = f"""
QMainWindow {{
    background: {PAPER};
}}
QMenuBar {{
    background: {PAPER};
    color: {INK};
    font-family: {FONT_UI};
    font-size: 10pt;
}}
QMenuBar::item {{
    background: transparent;
    padding: 4px 10px;
}}
QMenuBar::item:selected {{
    background: {PRESS};
}}
QMenu {{
    background: {PAPER};
    color: {INK};
    border: 1px solid {RULE};
    font-family: {FONT_UI};
    font-size: 10pt;
}}
QMenu::item {{
    padding: 6px 18px;
}}
QMenu::item:selected {{
    background: {PRESS};
}}
QWidget#sheet, QWidget#imagePane {{
    background: {PAPER};
    color: {INK};
    font-family: {FONT_UI};
    font-size: 10pt;
}}
QWidget#sheet QPushButton {{
    background: {PAPER};
    color: {INK};
    border: 1px solid {RULE};
    padding: 6px 14px;
    font-family: {FONT_UI};
    font-size: 10pt;
}}
QWidget#sheet QPushButton:hover {{
    background: {PRESS};
}}
QWidget#sheet QPushButton:focus {{
    border: 1px solid {INK};
}}
QWidget#sheet QPushButton:disabled {{
    color: {NOTE};
    background: {PAPER};
    border: 1px solid {RULE};
}}
QLabel#paneTitle {{
    color: {INK};
    background: transparent;
    font-family: {FONT_UI};
    font-size: 11pt;
    font-weight: 600;
}}
QLabel#paneView {{
    background-color: {WELL};
    color: {PAPER};
    font-family: {FONT_UI};
    font-size: 12pt;
}}
QLabel#paneFacts {{
    color: {NOTE};
    background: transparent;
    font-family: {FONT_UI};
    font-size: 9pt;
}}
QFrame#templateCard {{
    background: {PAPER};
    color: {INK};
    border: 1px solid {RULE};
}}
QFrame#templateCard QLabel {{
    background: {PAPER};
    color: {INK};
}}
"""

DIALOG_STYLE = f"""
QDialog {{
    background: {PAPER};
    color: {INK};
    font-family: {FONT_UI};
    font-size: 10pt;
}}
QDialog QLabel {{
    background: transparent;
    color: {INK};
    font-family: {FONT_UI};
    font-size: 10pt;
}}
QLabel#filterHint {{
    color: {NOTE};
    font-size: 9pt;
}}
QLabel#filterCode {{
    background: {PRESS};
    color: {INK};
    border: 1px solid {RULE};
    padding: 8px 10px;
    font-family: {FONT_CODE};
    font-size: 10pt;
}}
QSpinBox, QDoubleSpinBox, QComboBox {{
    background: {PAPER};
    color: {INK};
    border: 1px solid {RULE};
    padding: 2px 6px;
    font-family: {FONT_UI};
    font-size: 10pt;
}}
QDialogButtonBox QPushButton {{
    background: {PAPER};
    color: {INK};
    border: 1px solid {RULE};
    padding: 6px 14px;
    font-family: {FONT_UI};
    font-size: 10pt;
}}
QDialogButtonBox QPushButton:hover {{
    background: {PRESS};
}}
QDialogButtonBox QPushButton:focus {{
    border: 1px solid {INK};
}}
"""

FILTER_LIST_STYLE = f"""
#filterList {{
    background: {PAPER};
    border: 1px solid {RULE};
}}
#filterList QPushButton {{
    text-align: left;
    padding: 8px 16px;
    border: none;
    background: {PAPER};
    color: {INK};
    font-family: {FONT_UI};
    font-size: 10pt;
}}
#filterList QPushButton:hover {{
    background: {PRESS};
}}
"""


def _ui_font(point_size: int, weight: QFont.Weight = QFont.Weight.Normal) -> QFont:
    font = QFont()
    font.setFamilies(["DengXian", "等线", "Microsoft YaHei UI", "微软雅黑"])
    font.setPointSize(point_size)
    font.setWeight(weight)
    return font


def _code_font() -> QFont:
    font = QFont()
    font.setFamilies(["Cascadia Mono", "Consolas"])
    font.setPointSize(10)
    return font


class ImagePane(QWidget):
    """窗口里的一块画面：标题、图像、高度宽度通道数。"""

    def __init__(self, title: str, action: str, extra: str | None = None) -> None:
        super().__init__()
        self.setObjectName("imagePane")
        self.view = ImageView()
        self.view.setObjectName("paneView")
        self.view.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.view.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Ignored)
        self.view.setMinimumSize(1, 1)
        self.view.setFont(_ui_font(12))
        self.facts = QLabel()
        self.facts.setObjectName("paneFacts")
        self.facts.setFont(_ui_font(9))
        self.facts.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.facts.setWordWrap(True)

        heading = QLabel(title)
        heading.setObjectName("paneTitle")
        heading.setFont(_ui_font(11, QFont.Weight.DemiBold))
        heading.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        self.action_button = QPushButton(action)
        self.action_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self.extra_button: QPushButton | None = None
        if extra:
            self.extra_button = QPushButton(extra)
            self.extra_button.setCursor(Qt.CursorShape.PointingHandCursor)

        heading_row = QHBoxLayout()
        heading_row.setContentsMargins(0, 0, 0, 0)
        heading_row.setSpacing(8)
        heading_row.addWidget(heading)
        heading_row.addStretch(1)
        if self.extra_button is not None:
            heading_row.addWidget(self.extra_button)
        heading_row.addWidget(self.action_button)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)
        layout.addLayout(heading_row)
        layout.addWidget(self.view, 1)
        layout.addWidget(self.facts)


class TemplateCard(QFrame):
    """浮在处理结果右上角的模板预览。"""

    def __init__(self, parent: QWidget) -> None:
        super().__init__(parent)
        self.setObjectName("templateCard")
        self.view = QLabel()
        self.view.setObjectName("templateView")
        self.view.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.view.setScaledContents(False)
        self.close_button = QPushButton("关闭")
        self.close_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self.run_button = QPushButton("运行")
        self.run_button.setCursor(Qt.CursorShape.PointingHandCursor)

        buttons = QHBoxLayout()
        buttons.setContentsMargins(0, 0, 0, 0)
        buttons.setSpacing(8)
        buttons.addStretch(1)
        buttons.addWidget(self.run_button)
        buttons.addWidget(self.close_button)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)
        layout.addLayout(buttons)
        layout.addWidget(self.view)


class OperationList(QFrame):
    """悬停时浮在按钮下面的操作列表。"""

    def __init__(self, window: "MainWindow", operations: tuple[Operation, ...], choose) -> None:
        super().__init__(window, Qt.WindowType.Tool | Qt.WindowType.FramelessWindowHint)
        self._window = window
        self._choose_callback = choose
        self.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating, True)
        self.setObjectName("filterList")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(1, 1, 1, 1)
        layout.setSpacing(0)
        self._buttons: list[tuple[str, QPushButton]] = []
        for operation in operations:
            button = QPushButton(operation.label)
            button.setCursor(Qt.CursorShape.PointingHandCursor)
            button.clicked.connect(lambda _checked=False, operation_id=operation.id: self._choose(operation_id))
            layout.addWidget(button)
            self._buttons.append((operation.id, button))
        self.setFont(_ui_font(10))
        self.setStyleSheet(FILTER_LIST_STYLE)

    def operation_ids(self) -> list[str]:
        return [operation_id for operation_id, _button in self._buttons]

    def label_of(self, operation_id: str) -> str:
        return self._button(operation_id).text()

    def set_operation_label(self, operation_id: str, label: str) -> None:
        self._button(operation_id).setText(label)

    def _button(self, operation_id: str) -> QPushButton:
        for item_id, button in self._buttons:
            if item_id == operation_id:
                return button
        raise KeyError(operation_id)

    def leaveEvent(self, event) -> None:
        super().leaveEvent(event)
        self._window.schedule_hide_menu()

    def _choose(self, operation_id: str) -> None:
        self.hide()
        self._choose_callback(operation_id)


class FilterDialog(QDialog):
    def __init__(
        self,
        operation: Operation,
        channels: int,
        parent: QWidget | None = None,
        initial: dict | None = None,
        bounds: dict | None = None,
        shape: tuple[int, int] | None = None,
        describe=None,
    ) -> None:
        super().__init__(parent)
        self._operation = operation
        self._channels = channels
        self._shape = shape
        self._initial = initial or {}
        self._bounds = bounds or {}
        self._describe = describe
        self._widgets: dict = {}
        self.setWindowTitle(operation.label)
        self.setModal(True)

        form = QFormLayout()
        for param in operation.params:
            widget = self._make_param(param)
            form.addRow(param.label, widget)
            self._widgets[param.key] = widget

        self._code = QLabel()
        self._code.setObjectName("filterCode")
        self._code.setFont(_code_font())
        self._code.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self._code.setWordWrap(True)
        hint = QLabel(operation.hint)
        hint.setObjectName("filterHint")
        hint.setFont(_ui_font(9))
        hint.setWordWrap(True)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        ok_button = buttons.button(QDialogButtonBox.StandardButton.Ok)
        cancel_button = buttons.button(QDialogButtonBox.StandardButton.Cancel)
        if ok_button is not None:
            ok_button.setText("确定")
        if cancel_button is not None:
            cancel_button.setText("取消")
        buttons.accepted.connect(self._try_accept)
        buttons.rejected.connect(self.reject)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(18, 18, 18, 18)
        layout.setSpacing(12)
        form.setLabelAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        form.setHorizontalSpacing(16)
        form.setVerticalSpacing(8)
        layout.addLayout(form)
        layout.addWidget(hint)
        layout.addWidget(self._code)
        layout.addWidget(buttons)
        self.setFont(_ui_font(10))
        self.setStyleSheet(DIALOG_STYLE)
        self._refresh_code()
        self._code.ensurePolished()
        self._code.setWordWrap(False)
        line_width = self._code.sizeHint().width()
        self._code.setWordWrap(True)
        self._code.setMinimumWidth(line_width)

    def values(self) -> dict:
        values = {}
        for key, widget in self._widgets.items():
            if isinstance(widget, QDoubleSpinBox):
                values[key] = float(widget.value())
            elif isinstance(widget, QComboBox):
                values[key] = widget.currentData()
            else:
                values[key] = int(widget.value())
        return values

    def set_values(self, updates: dict) -> None:
        for key, value in updates.items():
            widget = self._widgets.get(key)
            if isinstance(widget, QSpinBox):
                widget.blockSignals(True)
                widget.setValue(int(value))
                widget.blockSignals(False)
            elif isinstance(widget, QDoubleSpinBox):
                widget.blockSignals(True)
                widget.setValue(float(value))
                widget.blockSignals(False)
        self._refresh_code()

    def _make_param(self, param: Param) -> QWidget:
        if param.kind == "choice":
            box = QComboBox()
            for label, value in param.choices:
                box.addItem(label, value)
            selected = self._initial.get(param.key, param.default)
            index = box.findData(selected)
            if index < 0:
                index = box.findData(param.default)
            if index >= 0:
                box.setCurrentIndex(index)
            box.currentIndexChanged.connect(self._refresh_code)
            return box
        if param.kind == "float":
            box = QDoubleSpinBox()
            box.setDecimals(1)
            box.setRange(float(param.minimum), float(param.maximum))
            box.setSingleStep(float(param.step))
            if param.key in self._bounds:
                box.setMaximum(float(self._bounds[param.key]))
            box.setValue(float(self._initial.get(param.key, param.default)))
            box.valueChanged.connect(self._refresh_code)
            return box
        box = QSpinBox()
        box.setRange(int(param.minimum), int(param.maximum))
        box.setSingleStep(max(1, int(param.step)))
        if param.key in self._bounds:
            box.setMaximum(int(self._bounds[param.key]))
        value = int(self._initial.get(param.key, param.default))
        if param.key in self._initial:
            box.setMaximum(max(box.maximum(), value))
        box.setValue(value)
        box.valueChanged.connect(self._refresh_code)
        return box

    def _refresh_code(self) -> None:
        if self._describe is not None:
            text = self._describe(self.values())
        else:
            text = operation_code(self._operation.id, self.values(), self._channels, self._shape)
        self._code.setText(text)

    def _try_accept(self) -> None:
        if self._describe is None:
            error = operation_error(self._operation.id, self.values(), self._shape)
            if error:
                QMessageBox.warning(self, self._operation.label, error)
                return
        self.accept()


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self._current = CurrentImage()
        self._original: np.ndarray | None = None
        self._on_original = True
        self._result = None
        self._result_note = ""
        self._gray_readout = False
        self._template: np.ndarray | None = None
        self._refreshing = False
        self._refresh_again = False
        self._roi_dialog: FilterDialog | None = None
        self._roi_closing = False
        self.setWindowTitle("视觉学习")
        self.setMinimumSize(800, 560)
        self.setFont(_ui_font(10))
        self.setStyleSheet(WINDOW_STYLE)

        self._import_button = QPushButton("导入文件")
        self._import_button.clicked.connect(self.choose_file)
        self._filter_button = QPushButton("滤波处理")
        self._filter_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self._morph_button = QPushButton("形态学处理")
        self._morph_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self._basic_button = QPushButton("图像基本处理")
        self._basic_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self._threshold_button = QPushButton("阈值处理")
        self._threshold_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self._edge_button = QPushButton("边缘检测")
        self._edge_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self._contour_button = QPushButton("轮廓检测")
        self._contour_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self._template_button = QPushButton("模板匹配")
        self._template_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self._template_button.clicked.connect(self.choose_template)

        self._original_pane = ImagePane("当前图像", "回到原图")
        self._result_pane = ImagePane("处理结果", "设为当前图像", "保存")
        self._original_pane.action_button.clicked.connect(self.restore_original)
        self._result_pane.action_button.clicked.connect(self.use_result)
        if self._result_pane.extra_button is not None:
            self._result_pane.extra_button.clicked.connect(self.save_result)
        self._template_card = TemplateCard(self._result_pane)
        self._template_card.run_button.clicked.connect(self.run_template)
        self._template_card.close_button.clicked.connect(self.close_template)
        self._template_card.hide()

        header = QHBoxLayout()
        header.setSpacing(8)
        header.addWidget(self._import_button)
        header.addWidget(self._basic_button)
        header.addWidget(self._filter_button)
        header.addWidget(self._morph_button)
        header.addWidget(self._threshold_button)
        header.addWidget(self._edge_button)
        header.addWidget(self._contour_button)
        header.addWidget(self._template_button)
        header.addStretch(1)

        views = QHBoxLayout()
        views.setSpacing(20)
        views.addWidget(self._original_pane, 1)
        views.addWidget(self._result_pane, 1)

        root = QWidget()
        root.setObjectName("sheet")
        layout = QVBoxLayout(root)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(16)
        layout.addLayout(header)
        layout.addLayout(views, 1)
        self.setCentralWidget(root)

        self._filter_list = OperationList(self, FILTERS, self.choose_filter)
        self._morph_list = OperationList(self, MORPHOLOGY, self.choose_morphology)
        self._basic_list = OperationList(self, BASICS, self.choose_basic)
        self._threshold_list = OperationList(self, THRESHOLDS, self.choose_threshold)
        self._edge_list = OperationList(self, EDGES, self.choose_edge)
        self._contour_list = OperationList(self, CONTOURS, self.choose_contour)
        self._menus = {
            self._filter_button: self._filter_list,
            self._morph_button: self._morph_list,
            self._basic_button: self._basic_list,
            self._threshold_button: self._threshold_list,
            self._edge_button: self._edge_list,
            self._contour_button: self._contour_list,
        }
        self._hide_timer = QTimer(self)
        self._hide_timer.setSingleShot(True)
        self._hide_timer.setInterval(250)
        self._hide_timer.timeout.connect(self._hide_menu_if_outside)
        for button in self._menus:
            button.installEventFilter(self)

        import_action = QAction("导入文件", self)
        import_action.triggered.connect(self.choose_file)
        self._save_action = QAction("保存处理结果", self)
        self._save_action.triggered.connect(self.save_result)
        quit_action = QAction("退出", self)
        quit_action.triggered.connect(self.close)
        file_menu = self.menuBar().addMenu("文件")
        file_menu.addAction(import_action)
        file_menu.addAction(self._save_action)
        file_menu.addSeparator()
        file_menu.addAction(quit_action)
        self._original_pane.view.roi_changed.connect(self._roi_from_view)
        self._refresh()

    def eventFilter(self, watched, event) -> bool:
        if watched in self._menus:
            if event.type() == event.Type.Enter:
                self._show_menu(watched)
            elif event.type() == event.Type.Leave:
                self.schedule_hide_menu()
        return super().eventFilter(watched, event)

    def _show_menu(self, button: QPushButton) -> None:
        self._hide_timer.stop()
        for other, menu in self._menus.items():
            if other is not button:
                menu.hide()
        menu = self._menus[button]
        menu.adjustSize()
        anchor = button.mapToGlobal(QPoint(button.width(), button.height()))
        menu.move(anchor.x() - menu.width(), anchor.y())
        menu.show()

    def schedule_hide_menu(self) -> None:
        self._hide_timer.start()

    def _hide_menu_if_outside(self) -> None:
        if self._pointer_on_menu():
            return
        for menu in self._menus.values():
            menu.hide()

    def _pointer_on_menu(self) -> bool:
        pos = QCursor.pos()
        for button, menu in self._menus.items():
            if button.rect().contains(button.mapFromGlobal(pos)):
                return True
            if not menu.isVisible():
                continue
            local = menu.mapFromGlobal(pos)
            if menu.rect().adjusted(-8, -16, 8, 8).contains(local):
                return True
        return False

    def choose_file(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "导入文件", "", IMAGE_FILTER)
        if not path:
            return
        if not self.open_path(Path(path)):
            QMessageBox.warning(self, "无法读入", READ_FAILED)

    def choose_template(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "导入模板", "", IMAGE_FILTER)
        if not path:
            return
        if not self.open_template(Path(path)):
            QMessageBox.warning(self, "无法读入", READ_FAILED)

    def open_template(self, path: Path) -> bool:
        """读入成功才更换模板。失败时保留原来的那张。"""
        array = read_image(path)
        if array is None:
            return False
        self._template = array.copy()
        preview = fit_long_edge(self._template)
        pixmap = QPixmap.fromImage(to_qimage(preview))
        self._template_card.view.setPixmap(pixmap)
        self._template_card.view.setFixedSize(pixmap.size())
        self._template_card.show()
        self._template_card.adjustSize()
        self._place_template_card()
        return True

    def close_template(self) -> None:
        self._template = None
        self._template_card.view.clear()
        self._template_card.hide()

    def run_template(self) -> None:
        if self._current.array is None:
            QMessageBox.information(self, "导入文件", NEED_IMAGE)
            return
        if self._template is None:
            return
        image = self._current.array
        template = self._template
        image_channels = 1 if image.ndim == 2 else int(image.shape[2])
        template_channels = 1 if template.ndim == 2 else int(template.shape[2])
        dialog = FilterDialog(
            MATCH,
            image_channels,
            self,
            describe=lambda values: match_code(values, image_channels, template_channels),
        )
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        self.run_match(dialog.values())

    def run_match(self, params: dict) -> None:
        if self._current.array is None:
            QMessageBox.information(self, "导入文件", NEED_IMAGE)
            return
        if self._template is None:
            return
        result, code = apply_match(self._current.array, self._template, params)
        if result is None:
            QMessageBox.warning(self, "模板匹配", code)
            return
        self._end_roi()
        self._result = result
        self._result_note = _note_from_code(code)
        self._refresh()

    def _place_template_card(self) -> None:
        card = self._template_card
        if card.isHidden():
            return
        view = self._result_pane.view
        corner = view.mapTo(self._result_pane, QPoint(view.width(), 0))
        card.adjustSize()
        x = corner.x() - card.width() - 8
        y = corner.y() + 8
        card.move(max(0, x), max(0, y))
        card.raise_()

    def save_result(self) -> None:
        """把处理结果写成图像文件。还没有结果时不打开对话框。"""
        if self._result is None:
            QMessageBox.information(self, "保存处理结果", "还没有处理结果。")
            return
        path, _selected = QFileDialog.getSaveFileName(self, "保存处理结果", self._result_suggestion(), SAVE_FILTER)
        if not path:
            return
        if not self.save_result_to(Path(path)):
            QMessageBox.warning(self, "无法保存", SAVE_FAILED)

    def save_result_to(self, path: Path) -> bool:
        if self._result is None:
            return False
        return save_image(self._result, path)

    def _result_suggestion(self) -> str:
        source = self._current.source
        if source is None:
            return "结果.png"
        return str(source.with_name(f"{source.stem}_结果.png"))

    def open_path(self, path: Path) -> bool:
        """读入成功才更换当前图像。失败时保留原来的那张。"""
        array = read_image(path)
        if array is None:
            return False
        self._end_roi()
        self._original = array.copy()
        self._current.replace(array, path)
        self._on_original = True
        self._result = None
        self._result_note = ""
        self._refresh()
        return True

    def use_result(self) -> None:
        """处理结果成为当前图像。下一次操作从这张图开始。"""
        if self._result is None or self._current.array is None:
            return
        self._end_roi()
        self._current.adopt(self._result)
        self._result = None
        self._result_note = ""
        self._on_original = False
        self._refresh()

    def restore_original(self) -> None:
        """当前图像换回最近一次读入的那张。左边已经是原图时不动。"""
        if self._on_original or self._original is None or self._current.array is None:
            return
        self._end_roi()
        self._current.adopt(self._original)
        self._result = None
        self._result_note = ""
        self._on_original = True
        self._refresh()

    def choose_filter(self, filter_id: str) -> None:
        self._choose(filter_id, filter_by_id, apply_filter)

    def choose_morphology(self, morph_id: str) -> None:
        self._choose(morph_id, morphology_by_id, apply_morphology)

    def choose_basic(self, operation_id: str) -> None:
        if operation_id == "select_roi":
            self._begin_roi()
            return
        if operation_id == "to_gray":
            self._apply_now(operation_id, basic_by_id, apply_basic)
            return
        array = self._current.array
        initial = None
        if array is not None and operation_id == "resize":
            initial = {"width": int(array.shape[1]), "height": int(array.shape[0])}
        self._choose(operation_id, basic_by_id, apply_basic, initial=initial)

    def choose_threshold(self, operation_id: str) -> None:
        if operation_id == "gray_value":
            self._toggle_gray_readout()
            return
        self._choose(operation_id, threshold_by_id, apply_threshold)

    def _toggle_gray_readout(self) -> None:
        self._gray_readout = not self._gray_readout
        self._original_pane.view.set_gray_readout(self._gray_readout)
        self._result_pane.view.set_gray_readout(self._gray_readout)
        label = "关闭灰度值" if self._gray_readout else "显示灰度值"
        self._threshold_list.set_operation_label("gray_value", label)

    def choose_edge(self, operation_id: str) -> None:
        self._choose(operation_id, edge_by_id, apply_edge)

    def choose_contour(self, operation_id: str) -> None:
        self._choose(operation_id, contour_by_id, apply_contour)

    def _apply_now(self, operation_id: str, lookup, apply) -> None:
        """不打开参数对话框，直接对当前图像做这一次处理。"""
        self._end_roi()
        array = self._current.array
        if array is None:
            QMessageBox.information(self, "导入文件", NEED_IMAGE)
            return
        result, code = apply(operation_id, array, {})
        if result is None:
            QMessageBox.warning(self, lookup(operation_id).label, code)
            return
        self._result = result
        self._result_note = _note_from_code(code)
        self._refresh()

    def _choose(self, operation_id: str, lookup, apply, initial: dict | None = None) -> None:
        self._end_roi()
        array = self._current.array
        if array is None:
            QMessageBox.information(self, "导入文件", NEED_IMAGE)
            return
        operation = lookup(operation_id)
        channels = 1 if array.ndim == 2 else int(array.shape[2])
        dialog = FilterDialog(operation, channels, self, initial=initial, shape=array.shape[:2])
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        result, code = apply(operation_id, array, dialog.values())
        if result is None:
            QMessageBox.warning(self, operation.label, code)
            return
        self._result = result
        self._result_note = _note_from_code(code)
        self._refresh()

    def _begin_roi(self) -> None:
        array = self._current.array
        if array is None:
            QMessageBox.information(self, "导入文件", NEED_IMAGE)
            return
        self._end_roi()
        height, width = array.shape[:2]
        channels = 1 if array.ndim == 2 else int(array.shape[2])
        dialog = FilterDialog(
            basic_by_id("select_roi"),
            channels,
            self,
            bounds={"y1": height, "y2": height, "x1": width, "x2": width},
            shape=(height, width),
        )
        dialog.setModal(False)
        for widget in dialog._widgets.values():
            if isinstance(widget, QSpinBox):
                widget.valueChanged.connect(self._roi_from_dialog)
        dialog.accepted.connect(self._apply_roi)
        dialog.rejected.connect(self._end_roi)
        self._roi_dialog = dialog
        self._original_pane.view.set_roi_active(True)
        dialog.show()
        dialog.adjustSize()
        self._place_beside(dialog)

    def _place_beside(self, dialog: QDialog) -> None:
        frame = self.frameGeometry()
        x = frame.right() + 12
        y = frame.top()
        screen = self.screen()
        if screen is not None:
            area = screen.availableGeometry()
            if x + dialog.width() > area.right():
                x = max(area.left(), area.right() - dialog.width())
            y = min(max(y, area.top()), max(area.top(), area.bottom() - dialog.height()))
        dialog.move(x, y)

    def _roi_from_view(self, x1: int, y1: int, x2: int, y2: int) -> None:
        dialog = self._roi_dialog
        if dialog is None:
            return
        dialog.set_values({"y1": y1, "y2": y2, "x1": x1, "x2": x2})

    def _roi_from_dialog(self) -> None:
        dialog = self._roi_dialog
        if dialog is None:
            return
        values = dialog.values()
        x1 = int(values["x1"])
        y1 = int(values["y1"])
        x2 = int(values["x2"])
        y2 = int(values["y2"])
        if x2 <= x1 or y2 <= y1:
            self._original_pane.view.set_roi(None)
            return
        self._original_pane.view.set_roi((x1, y1, x2, y2))

    def _apply_roi(self) -> None:
        dialog = self._roi_dialog
        array = self._current.array
        if dialog is None or array is None:
            return
        result, message = apply_basic("select_roi", array, dialog.values())
        self._end_roi()
        if result is None:
            QMessageBox.warning(self, "划定 ROI", message)
            return
        self._result = result
        self._result_note = ""
        self._refresh()

    def _end_roi(self) -> None:
        if self._roi_closing:
            return
        self._roi_closing = True
        try:
            self._original_pane.view.set_roi_active(False)
            self._original_pane.view.set_roi(None)
            dialog = self._roi_dialog
            self._roi_dialog = None
            if dialog is not None:
                dialog.close()
        finally:
            self._roi_closing = False

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._refresh()

    def _refresh(self) -> None:
        if self._refreshing:
            self._refresh_again = True
            return
        self._refreshing = True
        try:
            self._paint()
        finally:
            self._refreshing = False
        if self._refresh_again:
            self._refresh_again = False
            self._refresh()

    def _paint(self) -> None:
        array = self._current.array
        if array is None:
            self.setWindowTitle("视觉学习")
            self._clear_pane(self._original_pane, EMPTY_VIEW, EMPTY_FACTS)
            self._clear_pane(self._result_pane, EMPTY_RESULT, EMPTY_RESULT_FACTS)
            self._sync_source_buttons()
            self._place_template_card()
            return

        source = self._current.source
        name = source.name if source is not None else ""
        self.setWindowTitle(f"视觉学习 — {name}")
        self._show_array(self._original_pane, array, "当前图像")
        if self._result is None:
            self._clear_pane(self._result_pane, EMPTY_RESULT, EMPTY_RESULT_FACTS)
        else:
            self._show_array(self._result_pane, self._result, "处理结果", self._result_note)
        self._sync_source_buttons()
        self._place_template_card()

    def _sync_source_buttons(self) -> None:
        self._original_pane.action_button.setEnabled(self._current.array is not None and not self._on_original)
        self._result_pane.action_button.setEnabled(self._result is not None)
        if self._result_pane.extra_button is not None:
            self._result_pane.extra_button.setEnabled(self._result is not None)
        self._save_action.setEnabled(self._result is not None)

    def _clear_pane(self, pane: ImagePane, message: str, facts: str) -> None:
        pane.view.setPixmap(QPixmap())
        pane.view.set_pixels(None)
        pane.view.set_image_size(0, 0)
        pane.view.setText(message)
        pane.facts.setText(facts)

    def _show_array(self, pane: ImagePane, array, label: str, note: str = "") -> None:
        height, width, channels = image_facts(array)
        text = f"{label}    高度 {height}    宽度 {width}    通道数 {channels}"
        if note:
            text = f"{text}    {note}"
        pane.facts.setText(text)
        pane.view.setText("")
        pane.view.set_pixels(array)
        pane.view.set_image_size(width, height)
        pixmap = QPixmap.fromImage(to_qimage(array))
        view_width = max(1, pane.view.width())
        view_height = max(1, pane.view.height())
        if width > view_width or height > view_height:
            pixmap = pixmap.scaled(
                view_width,
                view_height,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
        pane.view.setPixmap(pixmap)
