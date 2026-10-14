"""教程第 6、7 节：对当前图像做一次操作，并给出对应的代码。"""

from dataclasses import dataclass

import cv2
import numpy as np

@dataclass(frozen=True)
class Param:
    key: str
    label: str
    kind: str
    default: int | float | str
    minimum: int | float = 0
    maximum: int | float = 0
    step: float = 1
    choices: tuple[tuple[str, str], ...] = ()


@dataclass(frozen=True)
class Operation:
    id: str
    label: str
    group: str
    params: tuple[Param, ...] = ()
    needs_other: bool = False
    hint: str = ""


_FLIP = (
    ("垂直", "0"),
    ("水平", "1"),
    ("双向", "-1"),
)
_CHANNELS = (
    ("原样合并", "merge"),
    ("只留蓝", "blue"),
    ("只留绿", "green"),
    ("只留红", "red"),
    ("交换红蓝", "swap"),
)

OPERATIONS: tuple[Operation, ...] = (
    Operation("original", "原图", ""),
    Operation(
        "pixel",
        "6 · 读改像素",
        "6",
        (
            Param("y", "行 y", "int", 100, 0, 10000),
            Param("x", "列 x", "int", 150, 0, 10000),
            Param("b", "蓝", "int", 255, 0, 255),
            Param("g", "绿", "int", 255, 0, 255),
            Param("r", "红", "int", 255, 0, 255),
        ),
    ),
    Operation(
        "roi",
        "6 · ROI",
        "6",
        (
            Param("y1", "上 y1", "int", 0, 0, 10000),
            Param("y2", "下 y2", "int", 100, 0, 10000),
            Param("x1", "左 x1", "int", 0, 0, 10000),
            Param("x2", "右 x2", "int", 100, 0, 10000),
            Param("b", "蓝", "int", 0, 0, 255),
            Param("g", "绿", "int", 255, 0, 255),
            Param("r", "红", "int", 0, 0, 255),
        ),
    ),
    Operation(
        "channels",
        "6 · 拆合通道",
        "6",
        (Param("mode", "通道", "choice", "swap", choices=_CHANNELS),),
    ),
    Operation(
        "resize",
        "6 · 缩放",
        "6",
        (
            Param("width", "宽度", "int", 200, 1, 8000),
            Param("height", "高度", "int", 200, 1, 8000),
        ),
    ),
    Operation(
        "rotate",
        "6 · 旋转",
        "6",
        (
            Param("angle", "角度", "int", 45, -360, 360),
            Param("scale", "缩放", "float", 1.0, 0.1, 5.0, 0.1),
        ),
    ),
    Operation(
        "translate",
        "6 · 平移",
        "6",
        (
            Param("dx", "水平位移", "int", 100, -4000, 4000),
            Param("dy", "垂直位移", "int", 50, -4000, 4000),
        ),
    ),
    Operation(
        "flip",
        "6 · 翻转",
        "6",
        (Param("code", "方向", "choice", "1", choices=_FLIP),),
    ),
    Operation("add", "7 · 相加（cv2.add）", "7", needs_other=True),
    Operation("subtract", "7 · 相减（cv2.subtract）", "7", needs_other=True),
    Operation("multiply", "7 · 相乘（cv2.multiply）", "7", needs_other=True),
    Operation("divide", "7 · 相除（cv2.divide）", "7", needs_other=True),
    Operation(
        "blend",
        "7 · 加权混合（cv2.addWeighted）",
        "7",
        (
            Param("alpha", "α", "float", 0.7, 0.0, 1.0, 0.1),
            Param("beta", "β", "float", 0.3, 0.0, 1.0, 0.1),
            Param("gamma", "γ", "float", 0.0, -255.0, 255.0, 1.0),
        ),
        needs_other=True,
    ),
    Operation("bitwise_and", "7 · 按位与（cv2.bitwise_and）", "7", needs_other=True),
    Operation("bitwise_or", "7 · 按位或（cv2.bitwise_or）", "7", needs_other=True),
    Operation("bitwise_not", "7 · 按位取反（cv2.bitwise_not）", "7"),
    Operation("bitwise_xor", "7 · 按位异或（cv2.bitwise_xor）", "7", needs_other=True),
)

_BY_ID = {item.id: item for item in OPERATIONS}

_NEED_OTHER = """# 先打开第二张图像
# 算术和按位运算要用 img1 和 img2
"""


def operation_by_id(operation_id: str) -> Operation:
    return _BY_ID[operation_id]


def apply_operation(
    operation_id: str,
    image: np.ndarray,
    other: np.ndarray | None,
    params: dict,
) -> tuple[np.ndarray | None, str]:
    """从当前图像算出处理结果。不修改传入的数组。"""
    operation = operation_by_id(operation_id)
    values = _values(operation, params)
    source = _single_channel(image)
    if operation.id == "original":
        return source, "# 原图"
    if operation.id == "pixel":
        return _pixel(source, values)
    if operation.id == "roi":
        return _roi(source, values)
    if operation.id == "channels":
        return _channels(source, str(values["mode"]))
    if operation.id == "resize":
        width = int(values["width"])
        height = int(values["height"])
        resized = cv2.resize(source, (width, height))
        code = f"resized = cv2.resize(img, ({width}, {height}))"
        return resized, code
    if operation.id == "rotate":
        return _rotate(source, int(values["angle"]), float(values["scale"]))
    if operation.id == "translate":
        return _translate(source, int(values["dx"]), int(values["dy"]))
    if operation.id == "flip":
        code_value = int(values["code"])
        flipped = cv2.flip(source, code_value)
        return flipped, f"flipped = cv2.flip(img, {code_value})"
    if operation.id == "bitwise_not":
        return cv2.bitwise_not(source), "not_img = cv2.bitwise_not(img1)"
    return _binary(operation.id, source, other, values)


def bounds_for(image: np.ndarray) -> dict[str, int]:
    """坐标类参数的上限，随当前图像的高宽变化。"""
    height, width = image.shape[:2]
    return {
        "y": max(0, height - 1),
        "x": max(0, width - 1),
        "y1": height,
        "y2": height,
        "x1": width,
        "x2": width,
    }


def _values(operation: Operation, params: dict) -> dict:
    values = {item.key: item.default for item in operation.params}
    values.update(params)
    for item in operation.params:
        if item.kind == "float":
            values[item.key] = round(float(values[item.key]), 2)
        elif item.kind == "int":
            values[item.key] = int(values[item.key])
    return values


def _single_channel(array: np.ndarray) -> np.ndarray:
    if array.ndim == 3 and array.shape[2] == 1:
        return np.ascontiguousarray(array[:, :, 0])
    return array


def _channels_of(array: np.ndarray) -> int:
    if array.ndim == 2:
        return 1
    return int(array.shape[2])


def _float_text(value: float) -> str:
    number = float(value)
    if number.is_integer():
        return f"{number:.1f}"
    return f"{number:.2f}".rstrip("0").rstrip(".")


def _color_text(array: np.ndarray, blue: int, green: int, red: int) -> str:
    if array.ndim == 2:
        return str(int(blue))
    if _channels_of(array) == 4:
        return f"[{int(blue)}, {int(green)}, {int(red)}, 255]"
    return f"[{int(blue)}, {int(green)}, {int(red)}]"


def _pixel(image: np.ndarray, values: dict) -> tuple[np.ndarray, str]:
    height, width = image.shape[:2]
    y = min(max(0, int(values["y"])), height - 1)
    x = min(max(0, int(values["x"])), width - 1)
    result = image.copy()
    color = _color_text(result, int(values["b"]), int(values["g"]), int(values["r"]))
    if result.ndim == 2:
        result[y, x] = int(values["b"])
    elif result.shape[2] == 4:
        result[y, x] = [int(values["b"]), int(values["g"]), int(values["r"]), 255]
    else:
        result[y, x] = [int(values["b"]), int(values["g"]), int(values["r"])]
    code = f"img = img.copy()\nimg[{y}, {x}] = {color}"
    return result, code


def _roi(image: np.ndarray, values: dict) -> tuple[np.ndarray, str]:
    height, width = image.shape[:2]
    y1 = min(max(0, int(values["y1"])), height)
    y2 = min(max(0, int(values["y2"])), height)
    x1 = min(max(0, int(values["x1"])), width)
    x2 = min(max(0, int(values["x2"])), width)
    if y1 >= y2 or x1 >= x2:
        return image.copy(), "# y1 要小于 y2，x1 要小于 x2\n# 这次没有改图像"
    result = image.copy()
    roi = result[y1:y2, x1:x2].copy()
    color = _color_text(result, int(values["b"]), int(values["g"]), int(values["r"]))
    if result.ndim == 2:
        roi[:, :] = int(values["b"])
    elif result.shape[2] == 4:
        roi[:, :] = [int(values["b"]), int(values["g"]), int(values["r"]), 255]
    else:
        roi[:, :] = [int(values["b"]), int(values["g"]), int(values["r"])]
    result[y1:y2, x1:x2] = roi
    code = (
        "img = img.copy()\n"
        f"roi = img[{y1}:{y2}, {x1}:{x2}].copy()\n"
        f"roi[:, :] = {color}\n"
        f"img[{y1}:{y2}, {x1}:{x2}] = roi"
    )
    return result, code


def _channels(image: np.ndarray, mode: str) -> tuple[np.ndarray, str]:
    if image.ndim == 2:
        return image, "# 灰度图是二维的，没有蓝、绿、红可以拆"
    if image.shape[2] == 4:
        blue, green, red, alpha = cv2.split(image)
        zeros = np.zeros_like(blue)
        planes = {
            "merge": (blue, green, red, alpha),
            "blue": (blue, zeros, zeros, alpha),
            "green": (zeros, green, zeros, alpha),
            "red": (zeros, zeros, red, alpha),
            "swap": (red, green, blue, alpha),
        }
        order = {
            "merge": "[b, g, r, a]",
            "blue": "[b, zeros, zeros, a]",
            "green": "[zeros, g, zeros, a]",
            "red": "[zeros, zeros, r, a]",
            "swap": "[r, g, b, a]",
        }
        merged = cv2.merge(list(planes[mode]))
        header = "b, g, r, a = cv2.split(img)\n"
        if mode != "merge" and mode != "swap":
            header += "zeros = np.zeros_like(b)\n"
        return merged, header + f"merged = cv2.merge({order[mode]})"
    blue, green, red = cv2.split(image)
    zeros = np.zeros_like(blue)
    planes = {
        "merge": (blue, green, red),
        "blue": (blue, zeros, zeros),
        "green": (zeros, green, zeros),
        "red": (zeros, zeros, red),
        "swap": (red, green, blue),
    }
    order = {
        "merge": "[b, g, r]",
        "blue": "[b, zeros, zeros]",
        "green": "[zeros, g, zeros]",
        "red": "[zeros, zeros, r]",
        "swap": "[r, g, b]",
    }
    merged = cv2.merge(list(planes[mode]))
    header = "b, g, r = cv2.split(img)\n"
    if mode not in ("merge", "swap"):
        header += "zeros = np.zeros_like(b)\n"
    return merged, header + f"merged = cv2.merge({order[mode]})"


def _rotate(image: np.ndarray, angle: int, scale: float) -> tuple[np.ndarray, str]:
    height, width = image.shape[:2]
    center = (width // 2, height // 2)
    matrix = cv2.getRotationMatrix2D(center, angle, scale)
    rotated = cv2.warpAffine(image, matrix, (width, height))
    code = (
        f"h, w = img.shape[:2]\n"
        f"center = {center}\n"
        f"M = cv2.getRotationMatrix2D(center, {angle}, {_float_text(scale)})\n"
        f"rotated = cv2.warpAffine(img, M, ({width}, {height}))"
    )
    return rotated, code


def _translate(image: np.ndarray, dx: int, dy: int) -> tuple[np.ndarray, str]:
    height, width = image.shape[:2]
    matrix = np.float32([[1, 0, dx], [0, 1, dy]])
    translated = cv2.warpAffine(image, matrix, (width, height))
    code = (
        f"T = np.float32([[1, 0, {dx}], [0, 1, {dy}]])\n"
        f"translated = cv2.warpAffine(img, T, ({width}, {height}))"
    )
    return translated, code


def _binary(
    operation_id: str,
    image: np.ndarray,
    other: np.ndarray | None,
    values: dict,
) -> tuple[np.ndarray | None, str]:
    calls = {
        "add": "added = cv2.add(img1, img2)",
        "subtract": "subbed = cv2.subtract(img1, img2)",
        "multiply": "mul = cv2.multiply(img1, img2)",
        "divide": "div = cv2.divide(img1, img2)",
        "bitwise_and": "and_img = cv2.bitwise_and(img1, img2)",
        "bitwise_or": "or_img = cv2.bitwise_or(img1, img2)",
        "bitwise_xor": "xor_img = cv2.bitwise_xor(img1, img2)",
    }
    if operation_id == "blend":
        alpha = _float_text(float(values["alpha"]))
        beta = _float_text(float(values["beta"]))
        gamma = _float_text(float(values["gamma"]))
        call = f"blend = cv2.addWeighted(img1, {alpha}, img2, {beta}, {gamma})"
    else:
        call = calls[operation_id]
    if other is None:
        return None, _NEED_OTHER + call
    aligned, lines = _align(image, other)
    if aligned is None:
        return None, "# 两张图的通道对不齐，没法运算\n" + "\n".join([*lines, call])
    runners = {
        "add": cv2.add,
        "subtract": cv2.subtract,
        "multiply": cv2.multiply,
        "divide": cv2.divide,
        "bitwise_and": cv2.bitwise_and,
        "bitwise_or": cv2.bitwise_or,
        "bitwise_xor": cv2.bitwise_xor,
    }
    if operation_id == "blend":
        result = cv2.addWeighted(
            image,
            float(values["alpha"]),
            aligned,
            float(values["beta"]),
            float(values["gamma"]),
        )
    else:
        result = runners[operation_id](image, aligned)
    return result, "\n".join([*lines, call])


def _align(image: np.ndarray, other: np.ndarray) -> tuple[np.ndarray | None, list[str]]:
    """把第二张图像调成和当前图像一样的尺寸、通道和类型。"""
    aligned = _single_channel(other)
    lines: list[str] = []
    height, width = image.shape[:2]
    if aligned.shape[:2] != (height, width):
        aligned = cv2.resize(aligned, (width, height))
        lines.append(f"img2 = cv2.resize(img2, ({width}, {height}))")
    target = _channels_of(image)
    current = _channels_of(aligned)
    if current != target:
        aligned, line = _recolor(aligned, current, target)
        if aligned is None or line is None:
            return None, lines
        lines.append(line)
    if aligned.dtype != image.dtype:
        aligned = aligned.astype(image.dtype)
        lines.append("img2 = img2.astype(img1.dtype)")
    return aligned, lines


def _recolor(
    image: np.ndarray, current: int, target: int
) -> tuple[np.ndarray | None, str | None]:
    conversion = {
        (1, 3): (cv2.COLOR_GRAY2BGR, "img2 = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)"),
        (1, 4): (cv2.COLOR_GRAY2BGRA, "img2 = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGRA)"),
        (3, 1): (cv2.COLOR_BGR2GRAY, "img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)"),
        (4, 1): (cv2.COLOR_BGRA2GRAY, "img2 = cv2.cvtColor(img2, cv2.COLOR_BGRA2GRAY)"),
        (4, 3): (cv2.COLOR_BGRA2BGR, "img2 = cv2.cvtColor(img2, cv2.COLOR_BGRA2BGR)"),
        (3, 4): (cv2.COLOR_BGR2BGRA, "img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2BGRA)"),
    }
    found = conversion.get((current, target))
    if found is None:
        return None, None
    flag, line = found
    return cv2.cvtColor(image, flag), line


FILTERS: tuple[Operation, ...] = (
    Operation(
        "gaussian",
        "高斯滤波",
        "smooth",
        (
            Param("k", "核大小", "int", 5, 1, 99, 2),
            Param("sigma", "sigmaX", "float", 0.0, 0.0, 200.0, 1.0),
        ),
        hint="核大小须为正奇数。sigmaX 为 0 时按核大小自动算。",
    ),
    Operation(
        "mean",
        "均值滤波",
        "smooth",
        (Param("k", "核大小", "int", 5, 1, 99),),
        hint="核越大越糊。",
    ),
    Operation(
        "median",
        "中值滤波",
        "smooth",
        (Param("k", "核大小", "int", 5, 3, 99, 2),),
        hint="核大小须为大于 1 的奇数。",
    ),
    Operation(
        "bilateral",
        "双边滤波",
        "smooth",
        (
            Param("d", "直径 d", "int", 9, 1, 51),
            Param("sigma_color", "sigmaColor", "int", 75, 1, 300),
            Param("sigma_space", "sigmaSpace", "int", 75, 1, 300),
        ),
        hint="d 是邻域直径。sigmaColor 看颜色差，sigmaSpace 看空间距离。",
    ),
)

_FILTER_BY_ID = {item.id: item for item in FILTERS}


def filter_by_id(filter_id: str) -> Operation:
    return _FILTER_BY_ID[filter_id]


def filter_error(filter_id: str, params: dict) -> str | None:
    operation = filter_by_id(filter_id)
    values = _values(operation, params)
    if filter_id in ("gaussian", "median"):
        size = int(values["k"])
        if size % 2 == 0 or size < 1:
            return "核大小须为正奇数"
        if filter_id == "median" and size <= 1:
            return "核大小须为大于 1 的奇数"
    return None


def apply_filter(
    filter_id: str,
    image: np.ndarray,
    params: dict,
) -> tuple[np.ndarray | None, str]:
    """从当前图像算出滤波结果。不修改传入的数组。失败时结果为 None。"""
    error = filter_error(filter_id, params)
    if error:
        return None, error
    source = _single_channel(image)
    code = filter_code(filter_id, params, _channels_of(source))
    values = _values(filter_by_id(filter_id), params)
    try:
        if filter_id == "gaussian":
            size = int(values["k"])
            result = cv2.GaussianBlur(source, (size, size), float(values["sigma"]))
        elif filter_id == "mean":
            size = int(values["k"])
            result = cv2.blur(source, (size, size))
        elif filter_id == "median":
            result = cv2.medianBlur(source, int(values["k"]))
        else:
            work = source
            if _channels_of(source) == 4:
                work = cv2.cvtColor(source, cv2.COLOR_BGRA2BGR)
            elif _channels_of(source) not in (1, 3):
                return None, "双边滤波只接受灰度图或三通道图"
            result = cv2.bilateralFilter(
                work,
                int(values["d"]),
                float(values["sigma_color"]),
                float(values["sigma_space"]),
            )
    except cv2.error:
        return None, "这张图不能做这种滤波"
    return result, code


def filter_code(filter_id: str, params: dict, channels: int) -> str:
    operation = filter_by_id(filter_id)
    values = _values(operation, params)
    if filter_id == "gaussian":
        size = int(values["k"])
        sigma = _plain_number(float(values["sigma"]))
        return f"gauss = cv2.GaussianBlur(img, ({size}, {size}), {sigma})"
    if filter_id == "mean":
        size = int(values["k"])
        return f"mean = cv2.blur(img, ({size}, {size}))"
    if filter_id == "median":
        return f"median = cv2.medianBlur(img, {int(values['k'])})"
    d = int(values["d"])
    sigma_color = int(values["sigma_color"])
    sigma_space = int(values["sigma_space"])
    call = f"bilateral = cv2.bilateralFilter(img, {d}, {sigma_color}, {sigma_space})"
    if channels == 4:
        return "img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)\n" + call
    return call


def _plain_number(value: float) -> str:
    if float(value) == 0:
        return "0"
    return _float_text(value)


_STRUCTURING_ELEMENTS = (
    ("矩形", "rect"),
    ("椭圆", "ellipse"),
    ("十字", "cross"),
)
_SHAPE_CODE = {
    "rect": "cv2.MORPH_RECT",
    "ellipse": "cv2.MORPH_ELLIPSE",
    "cross": "cv2.MORPH_CROSS",
}
_SHAPE_FLAG = {
    "rect": cv2.MORPH_RECT,
    "ellipse": cv2.MORPH_ELLIPSE,
    "cross": cv2.MORPH_CROSS,
}
_MORPH_PARAMS = (
    Param("k", "核大小", "int", 5, 1, 99),
    Param("iterations", "迭代次数", "int", 1, 1, 20),
    Param("shape", "结构元素", "choice", "rect", choices=_STRUCTURING_ELEMENTS),
)
_MORPH_EX = {
    "open": ("opened", "cv2.MORPH_OPEN", cv2.MORPH_OPEN),
    "close": ("closed", "cv2.MORPH_CLOSE", cv2.MORPH_CLOSE),
    "gradient": ("gradient", "cv2.MORPH_GRADIENT", cv2.MORPH_GRADIENT),
    "tophat": ("tophat", "cv2.MORPH_TOPHAT", cv2.MORPH_TOPHAT),
    "blackhat": ("blackhat", "cv2.MORPH_BLACKHAT", cv2.MORPH_BLACKHAT),
}

MORPHOLOGY: tuple[Operation, ...] = (
    Operation(
        "erode",
        "腐蚀",
        "morph",
        _MORPH_PARAMS,
        hint="用结构元素扫描图像，如果结构元素覆盖的区域全是前景，则保留中心像素。",
    ),
    Operation(
        "dilate",
        "膨胀",
        "morph",
        _MORPH_PARAMS,
        hint="用结构元素扫描图像，如果结构元素覆盖的区域存在前景，则保留中心像素。",
    ),
    Operation("open", "开运算", "morph", _MORPH_PARAMS, hint="先腐蚀后膨胀。"),
    Operation("close", "闭运算", "morph", _MORPH_PARAMS, hint="先膨胀后腐蚀。"),
    Operation("gradient", "形态学梯度", "morph", _MORPH_PARAMS, hint="膨胀图减去腐蚀图。"),
    Operation("tophat", "顶帽运算", "morph", _MORPH_PARAMS, hint="原图减去开运算结果。"),
    Operation("blackhat", "黑帽运算", "morph", _MORPH_PARAMS, hint="闭运算结果减去原图。"),
)

_MORPH_BY_ID = {item.id: item for item in MORPHOLOGY}


def morphology_by_id(morph_id: str) -> Operation:
    return _MORPH_BY_ID[morph_id]


def morphology_error(morph_id: str, params: dict) -> str | None:
    operation = morphology_by_id(morph_id)
    values = _values(operation, params)
    if int(values["k"]) < 1:
        return "核大小须为正整数"
    if int(values["iterations"]) < 1:
        return "迭代次数须为正整数"
    if str(values["shape"]) not in _SHAPE_FLAG:
        return "结构元素须为矩形、椭圆或十字"
    return None


def apply_morphology(
    morph_id: str,
    image: np.ndarray,
    params: dict,
) -> tuple[np.ndarray | None, str]:
    """从当前图像算出形态学结果。不修改传入的数组。失败时结果为 None。"""
    error = morphology_error(morph_id, params)
    if error:
        return None, error
    source = _single_channel(image)
    code = morphology_code(morph_id, params, _channels_of(source))
    values = _values(morphology_by_id(morph_id), params)
    kernel = cv2.getStructuringElement(_SHAPE_FLAG[str(values["shape"])], (int(values["k"]), int(values["k"])))
    iterations = int(values["iterations"])
    try:
        if morph_id == "erode":
            result = cv2.erode(source, kernel, iterations=iterations)
        elif morph_id == "dilate":
            result = cv2.dilate(source, kernel, iterations=iterations)
        else:
            result = cv2.morphologyEx(source, _MORPH_EX[morph_id][2], kernel, iterations=iterations)
    except cv2.error:
        return None, "这张图不能做这种形态学处理"
    return result, code


def morphology_code(morph_id: str, params: dict, channels: int) -> str:
    del channels
    operation = morphology_by_id(morph_id)
    values = _values(operation, params)
    size = int(values["k"])
    iterations = int(values["iterations"])
    shape = _SHAPE_CODE[str(values["shape"])]
    kernel = f"kernel = cv2.getStructuringElement({shape}, ({size}, {size}))"
    if morph_id == "erode":
        call = f"eroded = cv2.erode(img, kernel, iterations={iterations})"
    elif morph_id == "dilate":
        call = f"dilated = cv2.dilate(img, kernel, iterations={iterations})"
    else:
        name, flag, _op = _MORPH_EX[morph_id]
        call = f"{name} = cv2.morphologyEx(img, {flag}, kernel, iterations={iterations})"
    return f"{kernel}\n{call}"


def operation_code(
    operation_id: str,
    params: dict,
    channels: int,
    shape: tuple[int, int] | None = None,
) -> str:
    if operation_id in _MORPH_BY_ID:
        return morphology_code(operation_id, params, channels)
    if operation_id in _BASIC_BY_ID:
        return basic_code(operation_id, params, shape or (0, 0), channels)
    if operation_id in _THRESHOLD_BY_ID:
        return threshold_code(operation_id, params, channels)
    if operation_id in _EDGE_BY_ID:
        return edge_code(operation_id, params, channels)
    if operation_id in _CONTOUR_BY_ID:
        return contour_code(operation_id, params, channels)
    return filter_code(operation_id, params, channels)


def operation_error(
    operation_id: str,
    params: dict,
    shape: tuple[int, int] | None = None,
) -> str | None:
    if operation_id in _MORPH_BY_ID:
        return morphology_error(operation_id, params)
    if operation_id in _BASIC_BY_ID:
        return basic_error(operation_id, params, shape)
    if operation_id in _THRESHOLD_BY_ID:
        return threshold_error(operation_id, params)
    if operation_id in _EDGE_BY_ID:
        return edge_error(operation_id, params)
    if operation_id in _CONTOUR_BY_ID:
        return contour_error(operation_id, params)
    return filter_error(operation_id, params)


_BASIC_PARAMS_ROI = (
    Param("y1", "上 y1", "int", 0, 0, 10000),
    Param("y2", "下 y2", "int", 0, 0, 10000),
    Param("x1", "左 x1", "int", 0, 0, 10000),
    Param("x2", "右 x2", "int", 0, 0, 10000),
)

BASICS: tuple[Operation, ...] = (
    Operation(
        "resize",
        "图像缩放",
        "basic",
        (
            Param("width", "宽度", "int", 200, 1, 8000),
            Param("height", "高度", "int", 200, 1, 8000),
        ),
        hint="调整图像大小。",
    ),
    Operation(
        "rotate",
        "图像旋转",
        "basic",
        (
            Param("angle", "角度", "int", 45, -360, 360),
            Param("scale", "缩放", "float", 1.0, 0.1, 5.0, 0.1),
        ),
        hint="旋转图像。",
    ),
    Operation(
        "translate",
        "图像平移",
        "basic",
        (
            Param("dx", "水平位移", "int", 100, -4000, 4000),
            Param("dy", "垂直位移", "int", 50, -4000, 4000),
        ),
        hint="平移图像。",
    ),
    Operation(
        "flip",
        "图像翻转",
        "basic",
        (Param("code", "方向", "choice", "1", choices=_FLIP),),
        hint="翻转图像。",
    ),
    Operation(
        "select_roi",
        "划定 ROI",
        "basic",
        _BASIC_PARAMS_ROI,
        hint="在当前图像上按下并拖动，框出区域。拖动边或角可以微调。",
    ),
    Operation(
        "to_gray",
        "转为灰度图",
        "basic",
        (),
        hint="彩色图按蓝、绿、红合成灰度。已经是灰度图则原样复制。",
    ),
)

_BASIC_BY_ID = {item.id: item for item in BASICS}
_FLIP_CODES = {0, 1, -1}


def basic_by_id(operation_id: str) -> Operation:
    return _BASIC_BY_ID[operation_id]


def basic_error(
    operation_id: str,
    params: dict,
    shape: tuple[int, int] | None = None,
) -> str | None:
    operation = basic_by_id(operation_id)
    values = _values(operation, params)
    if operation_id == "resize":
        if int(values["width"]) < 1 or int(values["height"]) < 1:
            return "宽度和高度须为正整数"
    if operation_id == "flip" and int(values["code"]) not in _FLIP_CODES:
        return "方向须为垂直、水平或双向"
    if operation_id == "select_roi":
        y1 = int(values["y1"])
        y2 = int(values["y2"])
        x1 = int(values["x1"])
        x2 = int(values["x2"])
        if y1 >= y2 or x1 >= x2:
            return "上沿要小于下沿，左沿要小于右沿"
        if shape is not None:
            height, width = shape
            if y1 < 0 or x1 < 0 or y2 > height or x2 > width:
                return "框要落在图像里面"
    return None


def basic_code(operation_id: str, params: dict, shape: tuple[int, int], channels: int = 1) -> str:
    operation = basic_by_id(operation_id)
    values = _values(operation, params)
    height, width = shape
    if operation_id == "resize":
        return f"resized = cv2.resize(img, ({int(values['width'])}, {int(values['height'])}))"
    if operation_id == "rotate":
        center = (width // 2, height // 2)
        angle = int(values["angle"])
        scale = _float_text(float(values["scale"]))
        return (
            f"h, w = img.shape[:2]\n"
            f"center = {center}\n"
            f"M = cv2.getRotationMatrix2D(center, {angle}, {scale})\n"
            f"rotated = cv2.warpAffine(img, M, ({width}, {height}))"
        )
    if operation_id == "translate":
        dx = int(values["dx"])
        dy = int(values["dy"])
        return (
            f"T = np.float32([[1, 0, {dx}], [0, 1, {dy}]])\n"
            f"translated = cv2.warpAffine(img, T, ({width}, {height}))"
        )
    if operation_id == "flip":
        return f"flipped = cv2.flip(img, {int(values['code'])})"
    if operation_id == "to_gray":
        if channels == 4:
            return "gray = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)"
        if channels != 1:
            return "gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)"
        return "gray = img.copy()"
    y1 = int(values["y1"])
    y2 = int(values["y2"])
    x1 = int(values["x1"])
    x2 = int(values["x2"])
    return f"roi = img[{y1}:{y2}, {x1}:{x2}].copy()"


def apply_basic(
    operation_id: str,
    image: np.ndarray,
    params: dict,
) -> tuple[np.ndarray | None, str]:
    """从当前图像算出基本处理的结果。不修改传入的数组。失败时结果为 None。"""
    shape = image.shape[:2]
    error = basic_error(operation_id, params, shape)
    if error:
        return None, error
    source = _single_channel(image)
    channels = _channels_of(source)
    code = basic_code(operation_id, params, source.shape[:2], channels)
    values = _values(basic_by_id(operation_id), params)
    try:
        if operation_id == "resize":
            result = cv2.resize(source, (int(values["width"]), int(values["height"])))
        elif operation_id == "rotate":
            height, width = source.shape[:2]
            center = (width // 2, height // 2)
            matrix = cv2.getRotationMatrix2D(center, int(values["angle"]), float(values["scale"]))
            result = cv2.warpAffine(source, matrix, (width, height))
        elif operation_id == "translate":
            height, width = source.shape[:2]
            matrix = np.float32([[1, 0, int(values["dx"])], [0, 1, int(values["dy"])]])
            result = cv2.warpAffine(source, matrix, (width, height))
        elif operation_id == "flip":
            result = cv2.flip(source, int(values["code"]))
        elif operation_id == "to_gray":
            if channels == 4:
                result = cv2.cvtColor(source, cv2.COLOR_BGRA2GRAY)
            elif channels != 1:
                result = cv2.cvtColor(source, cv2.COLOR_BGR2GRAY)
            elif source.ndim == 2:
                result = source.copy()
            else:
                result = source[:, :, 0].copy()
        else:
            y1 = int(values["y1"])
            y2 = int(values["y2"])
            x1 = int(values["x1"])
            x2 = int(values["x2"])
            result = source[y1:y2, x1:x2].copy()
    except cv2.error:
        return None, "这张图不能做这种处理"
    return result, code


_THRESHOLD_KINDS = (
    ("二值化", "binary"),
    ("反二值化", "binary_inv"),
    ("截断", "trunc"),
    ("归零", "tozero"),
    ("反归零", "tozero_inv"),
)
_ADAPTIVE_KINDS = (
    ("二值化", "binary"),
    ("反二值化", "binary_inv"),
)
_ADAPTIVE_METHODS = (
    ("邻域均值", "mean"),
    ("高斯加权", "gaussian"),
)
_THRESHOLD_FLAG = {
    "binary": cv2.THRESH_BINARY,
    "binary_inv": cv2.THRESH_BINARY_INV,
    "trunc": cv2.THRESH_TRUNC,
    "tozero": cv2.THRESH_TOZERO,
    "tozero_inv": cv2.THRESH_TOZERO_INV,
}
_THRESHOLD_CODE_NAME = {
    "binary": "cv2.THRESH_BINARY",
    "binary_inv": "cv2.THRESH_BINARY_INV",
    "trunc": "cv2.THRESH_TRUNC",
    "tozero": "cv2.THRESH_TOZERO",
    "tozero_inv": "cv2.THRESH_TOZERO_INV",
}
_ADAPTIVE_FLAG = {
    "mean": cv2.ADAPTIVE_THRESH_MEAN_C,
    "gaussian": cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
}
_ADAPTIVE_CODE_NAME = {
    "mean": "cv2.ADAPTIVE_THRESH_MEAN_C",
    "gaussian": "cv2.ADAPTIVE_THRESH_GAUSSIAN_C",
}

THRESHOLDS: tuple[Operation, ...] = (
    Operation(
        "simple",
        "简单阈值处理",
        "threshold",
        (
            Param("thresh", "阈值", "int", 127, 0, 255),
            Param("maxval", "最大值", "int", 255, 0, 255),
            Param("kind", "类型", "choice", "binary", choices=_THRESHOLD_KINDS),
        ),
        hint="大于阈值取最大值，否则取 0。类型可以反过来、截断或归零。",
    ),
    Operation(
        "adaptive",
        "自适应阈值处理",
        "threshold",
        (
            Param("maxval", "最大值", "int", 255, 0, 255),
            Param("method", "邻域", "choice", "mean", choices=_ADAPTIVE_METHODS),
            Param("kind", "类型", "choice", "binary", choices=_ADAPTIVE_KINDS),
            Param("block", "块大小", "int", 11, 3, 99, 2),
            Param("c", "常数 C", "int", 2, -50, 50),
        ),
        hint="每个像素的阈值来自邻域，再减去常数 C。块大小须为大于 1 的奇数。",
    ),
    Operation(
        "otsu",
        "Otsu 阈值处理",
        "threshold",
        (
            Param("maxval", "最大值", "int", 255, 0, 255),
            Param("kind", "类型", "choice", "binary", choices=_THRESHOLD_KINDS),
        ),
        hint="按直方图自动找阈值。适合有两个峰的图。",
    ),
    Operation(
        "histogram",
        "灰度直方图",
        "threshold",
        (),
        hint="统计每个灰度出现的次数。横轴是 0–255，纵轴是个数。",
    ),
    Operation(
        "equalize",
        "直方图均衡化",
        "threshold",
        (),
        hint="把灰度直方图拉平，提高对比。彩色图只均衡亮度，避免颜色偏掉。",
    ),
    Operation(
        "clahe",
        "自适应直方图均衡化",
        "threshold",
        (
            Param("clip", "对比度上限", "float", 2.0, 0.1, 40.0, 0.1),
            Param("tile", "块大小", "int", 8, 2, 32),
        ),
        hint="分块做均衡，再用对比度上限压住噪声。块越大，越接近整幅均衡。",
    ),
    Operation(
        "gray_value",
        "显示灰度值",
        "threshold",
        (),
        hint="悬停在像素上显示该点的灰度。再选一次关闭。",
    ),
)

_THRESHOLD_BY_ID = {item.id: item for item in THRESHOLDS}


def threshold_by_id(operation_id: str) -> Operation:
    return _THRESHOLD_BY_ID[operation_id]


def threshold_error(operation_id: str, params: dict) -> str | None:
    operation = threshold_by_id(operation_id)
    values = _values(operation, params)
    if operation_id != "adaptive":
        return None
    block = int(values["block"])
    if block % 2 == 0 or block <= 1:
        return "块大小须为大于 1 的奇数"
    return None


def gray_at(image: np.ndarray, x: int, y: int) -> int | None:
    """该像素的灰度。彩色像素先按蓝绿红换成灰度。"""
    if y < 0 or x < 0 or y >= int(image.shape[0]) or x >= int(image.shape[1]):
        return None
    if image.ndim == 2:
        return int(image[y, x])
    channels = int(image.shape[2])
    pixel = image[y : y + 1, x : x + 1]
    if channels == 3:
        gray = cv2.cvtColor(pixel, cv2.COLOR_BGR2GRAY)
    elif channels == 4:
        gray = cv2.cvtColor(pixel, cv2.COLOR_BGRA2GRAY)
    else:
        return None
    return int(gray[0, 0])


def _gray_source(
    image: np.ndarray,
    error: str = "阈值处理只接受灰度图或彩色图",
) -> tuple[np.ndarray | None, str]:
    source = _single_channel(image)
    channels = _channels_of(source)
    if channels == 1:
        return source, ""
    if channels == 3:
        return cv2.cvtColor(source, cv2.COLOR_BGR2GRAY), ""
    if channels == 4:
        return cv2.cvtColor(source, cv2.COLOR_BGRA2GRAY), ""
    return None, error


def threshold_code(operation_id: str, params: dict, channels: int) -> str:
    operation = threshold_by_id(operation_id)
    values = _values(operation, params)
    source_name = "img" if channels == 1 else "gray"
    if channels == 4:
        prefix = "gray = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)\n"
    elif channels != 1:
        prefix = "gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)\n"
    else:
        prefix = ""
    if operation_id == "simple":
        kind = _THRESHOLD_CODE_NAME[str(values["kind"])]
        call = (
            f"_, binary = cv2.threshold({source_name}, {int(values['thresh'])}, "
            f"{int(values['maxval'])}, {kind})"
        )
        return prefix + call
    if operation_id == "adaptive":
        method = _ADAPTIVE_CODE_NAME[str(values["method"])]
        kind = _THRESHOLD_CODE_NAME[str(values["kind"])]
        call = (
            f"adaptive = cv2.adaptiveThreshold({source_name}, {int(values['maxval'])}, "
            f"{method}, {kind}, {int(values['block'])}, {int(values['c'])})"
        )
        return prefix + call
    if operation_id == "otsu":
        kind = _THRESHOLD_CODE_NAME[str(values["kind"])]
        call = (
            f"otsu_val, otsu = cv2.threshold({source_name}, 0, {int(values['maxval'])}, "
            f"{kind} + cv2.THRESH_OTSU)"
        )
        return prefix + call
    if operation_id == "equalize":
        return _equalize_code(channels)
    if operation_id == "clahe":
        return _clahe_code(channels, float(values["clip"]), int(values["tile"]))
    return prefix + f"hist = cv2.calcHist([{source_name}], [0], None, [256], [0, 256])"


def _equalize_code(channels: int) -> str:
    if channels == 1:
        return "equalized = cv2.equalizeHist(img)"
    prefix = ""
    source = "img"
    if channels == 4:
        prefix = "bgr = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)\n"
        source = "bgr"
    return (
        f"{prefix}yuv = cv2.cvtColor({source}, cv2.COLOR_BGR2YUV)\n"
        "yuv[:, :, 0] = cv2.equalizeHist(yuv[:, :, 0])\n"
        "equalized = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)"
    )


def _clahe_code(channels: int, clip: float, tile: int) -> str:
    maker = f"clahe = cv2.createCLAHE(clipLimit={_float_text(clip)}, tileGridSize=({tile}, {tile}))"
    if channels == 1:
        return f"{maker}\nequalized = clahe.apply(img)"
    prefix = ""
    source = "img"
    if channels == 4:
        prefix = "bgr = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)\n"
        source = "bgr"
    return (
        f"{prefix}{maker}\n"
        f"yuv = cv2.cvtColor({source}, cv2.COLOR_BGR2YUV)\n"
        "yuv[:, :, 0] = clahe.apply(yuv[:, :, 0])\n"
        "equalized = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)"
    )


def _equalize_brightness(source: np.ndarray, apply) -> np.ndarray:
    if _channels_of(source) == 1:
        gray = source if source.ndim == 2 else source[:, :, 0]
        return apply(gray)
    bgr = source if _channels_of(source) == 3 else cv2.cvtColor(source, cv2.COLOR_BGRA2BGR)
    yuv = cv2.cvtColor(bgr, cv2.COLOR_BGR2YUV)
    yuv[:, :, 0] = apply(yuv[:, :, 0])
    return cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)


def _histogram_ticks(peak: int) -> tuple[int, list[int]]:
    if peak <= 0:
        return 1, [0]
    raw = max(peak / 4, 1)
    magnitude = 1
    while magnitude * 10 <= raw:
        magnitude *= 10
    step = 10 * magnitude
    for factor in (1, 2, 5, 10):
        if factor * magnitude >= raw:
            step = factor * magnitude
            break
    step = max(1, int(step))
    top = int((peak + step - 1) // step * step)
    return top, list(range(0, top + 1, step))


def _draw_text(canvas: np.ndarray, text: str, x: int, y: int, align: str) -> None:
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.45
    thickness = 1
    (width, height), baseline = cv2.getTextSize(text, font, scale, thickness)
    if align == "center":
        x -= width // 2
    elif align == "right":
        x -= width
    x = max(0, min(int(x), canvas.shape[1] - width))
    y = max(height, min(int(y), canvas.shape[0] - baseline))
    cv2.putText(canvas, text, (x, y), font, scale, 28, thickness, cv2.LINE_AA)


def _histogram_image(gray: np.ndarray) -> np.ndarray:
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).reshape(-1)
    left, top, right, bottom = 78, 18, 22, 36
    plot_width, plot_height = 512, 200
    height = top + plot_height + bottom
    width = left + plot_width + right
    canvas = np.full((height, width), 242, np.uint8)
    peak = int(round(float(hist.max())))
    y_max, y_ticks = _histogram_ticks(peak)
    axis_y = top + plot_height
    for index, count in enumerate(hist):
        bar_height = int(round(float(count) / y_max * plot_height))
        if bar_height <= 0:
            continue
        x0 = left + 1 + index * (plot_width - 2) // 256
        x1 = left + 1 + (index + 1) * (plot_width - 2) // 256 - 1
        cv2.rectangle(canvas, (x0, axis_y - bar_height), (max(x0, x1), axis_y - 1), 28, thickness=-1)
    cv2.line(canvas, (left, top), (left, axis_y), 28, 1)
    cv2.line(canvas, (left, axis_y), (left + plot_width, axis_y), 28, 1)
    for value in (0, 50, 100, 150, 200, 255):
        x = left + int(round(value * plot_width / 255))
        cv2.line(canvas, (x, axis_y), (x, axis_y + 5), 28, 1)
        _draw_text(canvas, str(value), x, axis_y + 18, "center")
    for value in y_ticks:
        y = axis_y - int(round(value * plot_height / y_max))
        cv2.line(canvas, (left - 5, y), (left, y), 28, 1)
        _draw_text(canvas, str(value), left - 8, y + 4, "right")
    return canvas


def apply_threshold(
    operation_id: str,
    image: np.ndarray,
    params: dict,
) -> tuple[np.ndarray | None, str]:
    """从当前图像算出阈值或直方图。不修改传入的数组。失败时结果为 None。"""
    error = threshold_error(operation_id, params)
    if error:
        return None, error
    gray, problem = _gray_source(image)
    if gray is None:
        return None, problem
    code = threshold_code(operation_id, params, _channels_of(_single_channel(image)))
    values = _values(threshold_by_id(operation_id), params)
    try:
        if operation_id == "equalize":
            source = _single_channel(image)
            if _channels_of(source) not in (1, 3, 4):
                return None, "阈值处理只接受灰度图或彩色图"
            result = _equalize_brightness(source, cv2.equalizeHist)
        elif operation_id == "clahe":
            source = _single_channel(image)
            if _channels_of(source) not in (1, 3, 4):
                return None, "阈值处理只接受灰度图或彩色图"
            clahe = cv2.createCLAHE(clipLimit=float(values["clip"]), tileGridSize=(int(values["tile"]), int(values["tile"])))
            result = _equalize_brightness(source, clahe.apply)
        elif operation_id == "simple":
            _retval, result = cv2.threshold(
                gray,
                int(values["thresh"]),
                int(values["maxval"]),
                _THRESHOLD_FLAG[str(values["kind"])],
            )
        elif operation_id == "adaptive":
            result = cv2.adaptiveThreshold(
                gray,
                int(values["maxval"]),
                _ADAPTIVE_FLAG[str(values["method"])],
                _THRESHOLD_FLAG[str(values["kind"])],
                int(values["block"]),
                int(values["c"]),
            )
        elif operation_id == "otsu":
            retval, result = cv2.threshold(
                gray,
                0,
                int(values["maxval"]),
                _THRESHOLD_FLAG[str(values["kind"])] + cv2.THRESH_OTSU,
            )
            code = f"{code}\n# Otsu 阈值: {int(round(float(retval)))}"
        else:
            result = _histogram_image(gray)
    except cv2.error:
        return None, "这张图不能做这种处理"
    return result, code


_EDGE_DIRECTIONS = (
    ("水平", "x"),
    ("垂直", "y"),
    ("幅值", "magnitude"),
)
_CANNY_NORMS = (
    ("绝对值之和", "l1"),
    ("平方和开方", "l2"),
)
_EDGE_KERNELS = (1, 3, 5, 7)
_CANNY_APERTURES = (3, 5, 7)

EDGES: tuple[Operation, ...] = (
    Operation(
        "sobel",
        "Sobel 算子",
        "edge",
        (
            Param("direction", "方向", "choice", "magnitude", choices=_EDGE_DIRECTIONS),
            Param("ksize", "核大小", "int", 3, 1, 7, 2),
        ),
        hint="用差分核估计水平或垂直方向的一阶导。负的一边用浮点保留，再取绝对值。",
    ),
    Operation(
        "scharr",
        "Scharr 算子",
        "edge",
        (Param("direction", "方向", "choice", "magnitude", choices=_EDGE_DIRECTIONS),),
        hint="和 Sobel 一样求一阶导，3×3 核更强调中心。",
    ),
    Operation(
        "laplacian",
        "Laplacian 算子",
        "edge",
        (Param("ksize", "核大小", "int", 3, 1, 7, 2),),
        hint="二阶导，各个方向的突变都会有响应。",
    ),
    Operation(
        "canny",
        "Canny 边缘检测",
        "edge",
        (
            Param("low", "低阈值", "int", 100, 0, 255),
            Param("high", "高阈值", "int", 200, 0, 255),
            Param("aperture", "孔径", "int", 3, 3, 7, 2),
            Param("norm", "梯度", "choice", "l1", choices=_CANNY_NORMS),
        ),
        hint="平滑之后求梯度，只保留连到强边缘上的弱边缘。低阈值不能高于高阈值。",
    ),
)

_EDGE_BY_ID = {item.id: item for item in EDGES}


def edge_by_id(operation_id: str) -> Operation:
    return _EDGE_BY_ID[operation_id]


def edge_error(operation_id: str, params: dict) -> str | None:
    values = _values(edge_by_id(operation_id), params)
    if operation_id in ("sobel", "laplacian") and int(values["ksize"]) not in _EDGE_KERNELS:
        return "核大小须为 1、3、5 或 7"
    if operation_id != "canny":
        return None
    if int(values["aperture"]) not in _CANNY_APERTURES:
        return "孔径须为 3、5 或 7"
    if int(values["low"]) > int(values["high"]):
        return "低阈值不能高于高阈值"
    return None


def _edge_names(channels: int) -> tuple[str, str]:
    if channels == 4:
        return "gray", "gray = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)\n"
    if channels != 1:
        return "gray", "gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)\n"
    return "img", ""


def _derivative_code(kind: str, source: str, direction: str, ksize: int | None) -> str:
    call = "cv2.Sobel" if kind == "sobel" else "cv2.Scharr"
    size = "" if ksize is None else f", ksize={ksize}"
    if direction == "magnitude":
        return (
            f"{kind}_x = {call}({source}, cv2.CV_64F, 1, 0{size})\n"
            f"{kind}_y = {call}({source}, cv2.CV_64F, 0, 1{size})\n"
            f"{kind} = cv2.convertScaleAbs(cv2.magnitude({kind}_x, {kind}_y))"
        )
    axis = "x" if direction == "x" else "y"
    orders = "1, 0" if direction == "x" else "0, 1"
    return (
        f"{kind}_{axis} = {call}({source}, cv2.CV_64F, {orders}{size})\n"
        f"{kind}_{axis} = cv2.convertScaleAbs({kind}_{axis})"
    )


def edge_code(operation_id: str, params: dict, channels: int) -> str:
    values = _values(edge_by_id(operation_id), params)
    source, prefix = _edge_names(channels)
    if operation_id == "sobel":
        body = _derivative_code("sobel", source, str(values["direction"]), int(values["ksize"]))
    elif operation_id == "scharr":
        body = _derivative_code("scharr", source, str(values["direction"]), None)
    elif operation_id == "laplacian":
        body = (
            f"lap = cv2.Laplacian({source}, cv2.CV_64F, ksize={int(values['ksize'])})\n"
            "lap = cv2.convertScaleAbs(lap)"
        )
    else:
        norm = "True" if str(values["norm"]) == "l2" else "False"
        body = (
            f"edges = cv2.Canny({source}, {int(values['low'])}, {int(values['high'])}, "
            f"apertureSize={int(values['aperture'])}, L2gradient={norm})"
        )
    return prefix + body


def _derivative(kind: str, gray: np.ndarray, direction: str, ksize: int | None) -> np.ndarray:
    kwargs = {} if ksize is None else {"ksize": ksize}
    operator = cv2.Sobel if kind == "sobel" else cv2.Scharr
    if direction == "magnitude":
        gx = operator(gray, cv2.CV_64F, 1, 0, **kwargs)
        gy = operator(gray, cv2.CV_64F, 0, 1, **kwargs)
        return cv2.convertScaleAbs(cv2.magnitude(gx, gy))
    dx, dy = (1, 0) if direction == "x" else (0, 1)
    return cv2.convertScaleAbs(operator(gray, cv2.CV_64F, dx, dy, **kwargs))


def apply_edge(
    operation_id: str,
    image: np.ndarray,
    params: dict,
) -> tuple[np.ndarray | None, str]:
    """从当前图像算出边缘。不修改传入的数组。失败时结果为 None。"""
    error = edge_error(operation_id, params)
    if error:
        return None, error
    gray, problem = _gray_source(image, "边缘检测只接受灰度图或彩色图")
    if gray is None:
        return None, problem
    code = edge_code(operation_id, params, _channels_of(_single_channel(image)))
    values = _values(edge_by_id(operation_id), params)
    try:
        if operation_id == "sobel":
            result = _derivative("sobel", gray, str(values["direction"]), int(values["ksize"]))
        elif operation_id == "scharr":
            result = _derivative("scharr", gray, str(values["direction"]), None)
        elif operation_id == "laplacian":
            result = cv2.convertScaleAbs(cv2.Laplacian(gray, cv2.CV_64F, ksize=int(values["ksize"])))
        else:
            result = cv2.Canny(
                gray,
                int(values["low"]),
                int(values["high"]),
                apertureSize=int(values["aperture"]),
                L2gradient=str(values["norm"]) == "l2",
            )
    except cv2.error:
        return None, "这张图不能做这种处理"
    return result, code


_CONTOUR_MODES = (
    ("最外层", "external"),
    ("全部", "list"),
    ("两层", "ccomp"),
    ("整棵树", "tree"),
)
_CONTOUR_METHODS = (
    ("只留端点", "simple"),
    ("保留每个点", "none"),
)
_CONTOUR_MODE_CODE = {
    "external": "cv2.RETR_EXTERNAL",
    "list": "cv2.RETR_LIST",
    "ccomp": "cv2.RETR_CCOMP",
    "tree": "cv2.RETR_TREE",
}
_CONTOUR_MODE_FLAG = {
    "external": cv2.RETR_EXTERNAL,
    "list": cv2.RETR_LIST,
    "ccomp": cv2.RETR_CCOMP,
    "tree": cv2.RETR_TREE,
}
_CONTOUR_METHOD_CODE = {
    "simple": "cv2.CHAIN_APPROX_SIMPLE",
    "none": "cv2.CHAIN_APPROX_NONE",
}
_CONTOUR_METHOD_FLAG = {
    "simple": cv2.CHAIN_APPROX_SIMPLE,
    "none": cv2.CHAIN_APPROX_NONE,
}

CONTOURS: tuple[Operation, ...] = (
    Operation(
        "find",
        "查找轮廓",
        "contour",
        (
            Param("thresh", "阈值", "int", 127, 0, 255),
            Param("mode", "检索", "choice", "external", choices=_CONTOUR_MODES),
            Param("method", "近似", "choice", "simple", choices=_CONTOUR_METHODS),
            Param("thickness", "线宽", "int", 2, 1, 20),
            Param("min_area", "最小面积", "int", 0, 0, 1_000_000),
        ),
        hint="先按阈值分成黑白，白的区域是前景。绿线画出轮廓。最小面积可以丢掉小碎块。",
    ),
)

_CONTOUR_BY_ID = {item.id: item for item in CONTOURS}


def contour_by_id(operation_id: str) -> Operation:
    return _CONTOUR_BY_ID[operation_id]


def contour_error(operation_id: str, params: dict) -> str | None:
    del operation_id, params
    return None


def _contour_lines(channels: int) -> tuple[str, str, str]:
    if channels == 4:
        return (
            "gray",
            "gray = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)\n",
            "drawn = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)",
        )
    if channels != 1:
        return "gray", "gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)\n", "drawn = img.copy()"
    return "img", "", "drawn = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)"


def contour_code(operation_id: str, params: dict, channels: int) -> str:
    del operation_id
    values = _values(contour_by_id("find"), params)
    source, prefix, canvas = _contour_lines(channels)
    mode = _CONTOUR_MODE_CODE[str(values["mode"])]
    method = _CONTOUR_METHOD_CODE[str(values["method"])]
    lines = []
    if prefix:
        lines.append(prefix.rstrip("\n"))
    lines.append(f"_, binary = cv2.threshold({source}, {int(values['thresh'])}, 255, cv2.THRESH_BINARY)")
    lines.append(f"contours, hierarchy = cv2.findContours(binary, {mode}, {method})")
    min_area = int(values["min_area"])
    if min_area > 0:
        lines.append(f"contours = [contour for contour in contours if cv2.contourArea(contour) >= {min_area}]")
    lines.append(canvas)
    lines.append(f"cv2.drawContours(drawn, contours, -1, (0, 255, 0), {int(values['thickness'])})")
    return "\n".join(lines)


def apply_contour(
    operation_id: str,
    image: np.ndarray,
    params: dict,
) -> tuple[np.ndarray | None, str]:
    """从当前图像找出轮廓并画成绿线。不修改传入的数组。失败时结果为 None。"""
    del operation_id
    error = contour_error("find", params)
    if error:
        return None, error
    source = _single_channel(image)
    channels = _channels_of(source)
    gray, problem = _gray_source(image, "轮廓检测只接受灰度图或彩色图")
    if gray is None:
        return None, problem
    code = contour_code("find", params, channels)
    values = _values(contour_by_id("find"), params)
    try:
        _retval, binary = cv2.threshold(gray, int(values["thresh"]), 255, cv2.THRESH_BINARY)
        contours, _hierarchy = cv2.findContours(
            binary,
            _CONTOUR_MODE_FLAG[str(values["mode"])],
            _CONTOUR_METHOD_FLAG[str(values["method"])],
        )
        min_area = int(values["min_area"])
        if min_area > 0:
            contours = [contour for contour in contours if cv2.contourArea(contour) >= min_area]
        if channels == 1:
            drawn = cv2.cvtColor(source, cv2.COLOR_GRAY2BGR)
        elif channels == 4:
            drawn = cv2.cvtColor(source, cv2.COLOR_BGRA2BGR)
        else:
            drawn = source.copy()
        cv2.drawContours(drawn, contours, -1, (0, 255, 0), int(values["thickness"]))
    except cv2.error:
        return None, "这张图不能做这种处理"
    return drawn, f"{code}\n# 轮廓数: {len(contours)}"
