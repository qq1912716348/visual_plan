"""当前图像上的框：像素坐标和画面坐标的换算。"""


def view_to_image(
    x: int,
    y: int,
    origin_x: int,
    origin_y: int,
    pixmap_width: int,
    pixmap_height: int,
    image_width: int,
    image_height: int,
) -> tuple[int, int] | None:
    local_x = x - origin_x
    local_y = y - origin_y
    if local_x < 0 or local_y < 0 or local_x >= pixmap_width or local_y >= pixmap_height:
        return None
    return (
        min(image_width - 1, local_x * image_width // pixmap_width),
        min(image_height - 1, local_y * image_height // pixmap_height),
    )


def view_to_image_clamped(
    x: int,
    y: int,
    origin_x: int,
    origin_y: int,
    pixmap_width: int,
    pixmap_height: int,
    image_width: int,
    image_height: int,
) -> tuple[int, int]:
    local_x = min(max(0, x - origin_x), pixmap_width - 1)
    local_y = min(max(0, y - origin_y), pixmap_height - 1)
    return (
        min(image_width - 1, local_x * image_width // pixmap_width),
        min(image_height - 1, local_y * image_height // pixmap_height),
    )


def image_to_view(
    index_x: int,
    index_y: int,
    origin_x: int,
    origin_y: int,
    pixmap_width: int,
    pixmap_height: int,
    image_width: int,
    image_height: int,
) -> tuple[int, int]:
    return (
        origin_x + index_x * pixmap_width // image_width,
        origin_y + index_y * pixmap_height // image_height,
    )


def normalize_box(
    x0: int,
    y0: int,
    x1: int,
    y1: int,
    width: int,
    height: int,
) -> tuple[int, int, int, int]:
    """把两次点击收成半开区间 [x1, x2) × [y1, y2)，终点那一格算在框内。"""
    left, right = sorted((x0, x1))
    top, bottom = sorted((y0, y1))
    x1_out = max(0, min(left, width - 1))
    y1_out = max(0, min(top, height - 1))
    x2_out = min(width, max(x1_out + 1, right + 1))
    y2_out = min(height, max(y1_out + 1, bottom + 1))
    return x1_out, y1_out, x2_out, y2_out


def move_box(
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    dx: int,
    dy: int,
    width: int,
    height: int,
) -> tuple[int, int, int, int]:
    box_width = x2 - x1
    box_height = y2 - y1
    next_x = min(max(0, x1 + dx), max(0, width - box_width))
    next_y = min(max(0, y1 + dy), max(0, height - box_height))
    return next_x, next_y, next_x + box_width, next_y + box_height


def resize_box(
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    handle: str,
    x: int,
    y: int,
    width: int,
    height: int,
) -> tuple[int, int, int, int]:
    if "w" in handle:
        x1 = min(max(0, x), x2 - 1)
    if "e" in handle:
        x2 = max(x1 + 1, min(width, x + 1))
    if "n" in handle:
        y1 = min(max(0, y), y2 - 1)
    if "s" in handle:
        y2 = max(y1 + 1, min(height, y + 1))
    return x1, y1, x2, y2


def hit_handle(x: int, y: int, left: int, top: int, right: int, bottom: int, tolerance: int = 6) -> str:
    near_left = abs(x - left) <= tolerance
    near_right = abs(x - right) <= tolerance
    near_top = abs(y - top) <= tolerance
    near_bottom = abs(y - bottom) <= tolerance
    spans_x = left - tolerance <= x <= right + tolerance
    spans_y = top - tolerance <= y <= bottom + tolerance
    if near_left and near_top and spans_x and spans_y:
        return "nw"
    if near_right and near_top and spans_x and spans_y:
        return "ne"
    if near_left and near_bottom and spans_x and spans_y:
        return "sw"
    if near_right and near_bottom and spans_x and spans_y:
        return "se"
    if near_top and spans_x:
        return "n"
    if near_bottom and spans_x:
        return "s"
    if near_left and spans_y:
        return "w"
    if near_right and spans_y:
        return "e"
    if left <= x <= right and top <= y <= bottom:
        return "move"
    return ""
