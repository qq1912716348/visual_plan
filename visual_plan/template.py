"""把模板缩成预览。最长边变成指定长度，短边按比例。"""

import cv2
import numpy as np


def fit_long_edge(image: np.ndarray, long_edge: int = 200) -> np.ndarray:
    """最长边变成 long_edge。缩小用面积插值，放大用线性插值。不修改传入的数组。"""
    height = int(image.shape[0])
    width = int(image.shape[1])
    if width >= height:
        new_width = long_edge
        new_height = max(1, int(round(height * long_edge / width)))
    else:
        new_height = long_edge
        new_width = max(1, int(round(width * long_edge / height)))
    if new_width == width and new_height == height:
        return image.copy()
    longest = max(height, width)
    interpolation = cv2.INTER_AREA if longest > long_edge else cv2.INTER_LINEAR
    return cv2.resize(image, (new_width, new_height), interpolation=interpolation)


MATCH_MODES = (
    ("归一化相关系数", "ccoeff_normed"),
    ("相关系数", "ccoeff"),
    ("归一化相关", "ccorr_normed"),
    ("相关", "ccorr"),
    ("归一化平方差", "sqdiff_normed"),
    ("平方差", "sqdiff"),
)
_MATCH_FLAG = {
    "ccoeff_normed": cv2.TM_CCOEFF_NORMED,
    "ccoeff": cv2.TM_CCOEFF,
    "ccorr_normed": cv2.TM_CCORR_NORMED,
    "ccorr": cv2.TM_CCORR,
    "sqdiff_normed": cv2.TM_SQDIFF_NORMED,
    "sqdiff": cv2.TM_SQDIFF,
}
_MATCH_CODE = {
    "ccoeff_normed": "cv2.TM_CCOEFF_NORMED",
    "ccoeff": "cv2.TM_CCOEFF",
    "ccorr_normed": "cv2.TM_CCORR_NORMED",
    "ccorr": "cv2.TM_CCORR",
    "sqdiff_normed": "cv2.TM_SQDIFF_NORMED",
    "sqdiff": "cv2.TM_SQDIFF",
}
_SMALLER_IS_BETTER = {"sqdiff", "sqdiff_normed"}


def _channels(image: np.ndarray) -> int:
    if image.ndim == 2:
        return 1
    return int(image.shape[2])


def _as_gray(image: np.ndarray) -> np.ndarray:
    channels = _channels(image)
    if channels == 1 and image.ndim == 3:
        return image[:, :, 0]
    if channels == 1:
        return image
    if channels == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def _color_canvas(image: np.ndarray) -> np.ndarray:
    channels = _channels(image)
    if channels == 1 and image.ndim == 3:
        return cv2.cvtColor(image[:, :, 0], cv2.COLOR_GRAY2BGR)
    if channels == 1:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    if channels == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    return image.copy()


def match_code(params: dict, image_channels: int, template_channels: int) -> str:
    mode = str(params.get("mode", "ccoeff_normed"))
    lines: list[str] = []
    image_name = "img"
    template_name = "template"
    if image_channels == 4:
        lines.append("gray = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)")
        image_name = "gray"
    elif image_channels != 1:
        lines.append("gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)")
        image_name = "gray"
    if template_channels == 4:
        lines.append("tpl = cv2.cvtColor(template, cv2.COLOR_BGRA2GRAY)")
        template_name = "tpl"
    elif template_channels != 1:
        lines.append("tpl = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)")
        template_name = "tpl"
    lines.append(f"matched = cv2.matchTemplate({image_name}, {template_name}, {_MATCH_CODE[mode]})")
    lines.append("min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(matched)")
    lines.append("x, y = min_loc" if mode in _SMALLER_IS_BETTER else "x, y = max_loc")
    lines.append("h, w = template.shape[:2]")
    if image_channels == 4:
        lines.append("drawn = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)")
    elif image_channels == 1:
        lines.append("drawn = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)")
    else:
        lines.append("drawn = img.copy()")
    lines.append("cv2.rectangle(drawn, (x, y), (x + w, y + h), (0, 255, 0), 2)")
    return "\n".join(lines)


def apply_match(
    image: np.ndarray,
    template: np.ndarray,
    params: dict,
) -> tuple[np.ndarray | None, str]:
    """在当前图像上找出和模板最像的一处，画上绿框。不修改传入的数组。"""
    mode = str(params.get("mode", "ccoeff_normed"))
    code = match_code(params, _channels(image), _channels(template))
    try:
        gray = _as_gray(image)
        tpl = _as_gray(template)
        if tpl.shape[0] > gray.shape[0] or tpl.shape[1] > gray.shape[1]:
            return None, "模板要比当前图像小"
        matched = cv2.matchTemplate(gray, tpl, _MATCH_FLAG[mode])
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(matched)
        if mode in _SMALLER_IS_BETTER:
            score, loc = float(min_val), min_loc
        else:
            score, loc = float(max_val), max_loc
        x, y = int(loc[0]), int(loc[1])
        height, width = int(tpl.shape[0]), int(tpl.shape[1])
        drawn = _color_canvas(image)
        cv2.rectangle(drawn, (x, y), (x + width, y + height), (0, 255, 0), 2)
    except cv2.error:
        return None, "这张图不能做这种处理"
    return drawn, f"{code}\n# 得分: {score:.2f}"
