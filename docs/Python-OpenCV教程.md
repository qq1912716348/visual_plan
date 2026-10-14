# Python OpenCV 教程

按菜鸟教程「Python OpenCV」全部章节整理的离线笔记：简介、安装，以及从入门实例到滤镜的 18 节。讲解和示例按同一套知识点重写，读这份文档即可，不必再点进各页。

图像在内存里是 NumPy 数组。彩色图默认是 BGR，通道顺序是蓝、绿、红，不是 RGB。灰度图是二维数组。像素值常用 `uint8`，范围 0 到 255。

下文示例里的 `image.jpg`、`video.mp4` 请换成自己的文件。窗口里按 `q` 或 `Esc` 退出；`waitKey(0)` 会一直等到按键。

---

## 1. OpenCV 是什么

OpenCV（Open Source Computer Vision Library）是开源的计算机视觉库，用 C/C++ 实现，并提供 Python、Java、MATLAB 等接口。1999 年由英特尔研究院启动，2000 年发布首个版本，2009 年的 2.0 开始较好支持 Python，2015 年的 3.0 加强深度学习，2018 年起的 4.x 继续优化性能，并支持 CUDA、OpenCL 等加速。

常用能力：

- 图像：滤波、颜色转换、形态学、边缘、特征。
- 视频：采集、运动分析、检测与跟踪。
- 几何：相机标定、立体视觉、三维重建。
- 学习：传统机器学习，以及用 DNN 模块跑预训练模型。

它跨 Windows、Linux、macOS、Android、iOS，算法经过优化，适合实时处理。典型场景包括人脸与物体检测、监控与跟踪、车道与标志识别、医学图像、机器人视觉、增强现实。

---

## 2. 安装（Python）

需要 Python 3 和 pip。Python 3.4 及以上自带 pip。

```bash
python -V
pip -V
pip install opencv-python
```

需要扩展模块（背景减除的旧接口 `bgsegm`、部分额外算法）时安装：

```bash
pip install opencv-contrib-python
```

`opencv-python` 与 `opencv-contrib-python` 不要同时装，二者都会提供 `cv2`，一起装会冲突。扩展包已包含主库功能。下载慢时可用镜像：

```bash
pip install opencv-python -i https://pypi.tuna.tsinghua.edu.cn/simple
```

用 Anaconda 时：

```bash
conda create -n opencv_env python=3.11
conda activate opencv_env
conda install -c conda-forge opencv
```

验证：

```python
import cv2
print(cv2.__version__)
```

能打印出版本号（例如 `4.10.0`）即安装成功。

---

## 3. 入门：读、显示、保存

```python
import cv2

image = cv2.imread("./bird.jpg")
if image is None:
    print("无法加载图像，请检查路径。")
    raise SystemExit

cv2.imshow("Display Image", image)
key = cv2.waitKey(0)  # 0 表示一直等待

if key == ord("s"):
    cv2.imwrite("saved_image.jpg", image)
    print("图像已保存为 saved_image.jpg")
elif key == ord("q"):
    print("退出。")
else:
    print("未保存。")

cv2.destroyAllWindows()
```

`ord("s")` 是字符 `s` 的编码。窗口标题、保存格式（`.jpg` / `.png`）都可以改。`imread` 路径错误时返回 `None`，不要对 `None` 调用 `imshow`。

---

## 4. 基础模块

Python 里这些能力都在 `cv2` 这一个包下，按功能可以分成这些模块：

| 模块 | 做什么 | 常见接口 |
| --- | --- | --- |
| core | 数组、矩阵、绘图 | `line`、`circle`、`rectangle`、`putText` |
| imgproc | 滤波、几何、颜色、阈值、边缘 | `blur`、`resize`、`warpAffine`、`cvtColor`、`threshold`、`Canny` |
| highgui | 显示与交互、视频读写 | `imshow`、`waitKey`、`VideoCapture`、`setMouseCallback` |
| video | 运动分析 | `createBackgroundSubtractorMOG2`、`calcOpticalFlowPyrLK` |
| features2d | 特征点 | `SIFT_create`、`ORB_create`、`BFMatcher`、`drawKeypoints` |
| calib3d | 标定与三维 | `calibrateCamera`、`findChessboardCorners`、`solvePnP`、`findHomography` |
| objdetect | 目标检测 | `CascadeClassifier`（人脸）、HOG 行人检测 |
| ml | 传统机器学习 | `ml.SVM_create`、`kmeans` |
| dnn | 加载预训练网络 | `dnn.readNet`、`net.forward` |
| 其他 | 近邻、修复、拼接、形状 | `flann`、`photo`、`Stitcher`、`shape` |

绘图示例：

```python
import cv2
import numpy as np

canvas = np.zeros((300, 400, 3), dtype=np.uint8)
cv2.line(canvas, (20, 40), (200, 40), (255, 0, 0), 2)
cv2.rectangle(canvas, (20, 80), (160, 180), (0, 255, 0), 2)
cv2.circle(canvas, (280, 140), 50, (0, 0, 255), -1)
cv2.putText(canvas, "OpenCV", (20, 250), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
cv2.imshow("draw", canvas)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

颜色参数是 BGR。线宽为 `-1` 时，矩形和圆会被填充。SURF 在较新的主库中不再附带，特征练习用 SIFT 或 ORB。

---

## 5. 图像处理基础

一张图就是数组：

- `img.shape`：灰度是 `(高, 宽)`，彩色是 `(高, 宽, 通道数)`。
- 下标是 `img[y, x]`，先行后列，不要写成 `(x, y)`。
- `cv2.imread(path)` 失败返回 `None`。第二参数 `cv2.IMREAD_GRAYSCALE`（也写作 `0`）直接读成灰度。
- `cv2.imwrite(path, img)` 写文件。
- `cv2.imshow` + `cv2.waitKey` + `cv2.destroyAllWindows` 负责显示。不调用 `waitKey`，窗口往往不刷新。

后面各节会展开像素、几何、算术、阈值、平滑、形态学、边缘、轮廓和视频。这里先记住颜色转换和裁剪：

```python
import cv2

img = cv2.imread("image.jpg")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)

h, w = img.shape[:2]
resized = cv2.resize(img, (w // 2, h // 2))
cropped = img[40:200, 60:260]  # y1:y2, x1:x2
```

---

## 6. 图像基本操作

| 操作 | 写法 |
| --- | --- |
| 读改像素 | `image[y, x]`，彩色再加通道 `image[y, x, c]`，`c` 为 0 蓝、1 绿、2 红 |
| ROI | `image[y1:y2, x1:x2]` |
| 拆合通道 | `cv2.split` / `cv2.merge` |
| 缩放 | `cv2.resize` |
| 旋转 | `cv2.getRotationMatrix2D` + `cv2.warpAffine` |
| 平移 | 2×3 矩阵 + `cv2.warpAffine` |
| 翻转 | `cv2.flip`，`0` 垂直，`1` 水平，`-1` 双向 |

```python
import cv2
import numpy as np

img = cv2.imread("image.jpg")
print(img[100, 150])                 # (x=150, y=100) 的 BGR
img[100, 150] = [255, 255, 255]

roi = img[0:100, 0:100].copy()
roi[:, :] = [0, 255, 0]
img[0:100, 0:100] = roi

b, g, r = cv2.split(img)
merged = cv2.merge([b, g, r])

resized = cv2.resize(img, (200, 200))

h, w = img.shape[:2]
center = (w // 2, h // 2)
M = cv2.getRotationMatrix2D(center, 45, 1.0)  # 角度、缩放
rotated = cv2.warpAffine(img, M, (w, h))

T = np.float32([[1, 0, 100], [0, 1, 50]])     # 右移 100，下移 50
translated = cv2.warpAffine(img, T, (w, h))

flipped = cv2.flip(img, 1)
```

切片得到的 ROI 与原图共享内存，改 ROI 会改原图。需要独立副本时用 `.copy()`。

---

## 7. 图像算术与位运算

两张图必须尺寸和类型一致。OpenCV 算术会饱和：相加超过 255 变成 255，相减小于 0 变成 0。NumPy 的 `+` 按 `uint8` 回绕（250+10 变成 4），和 `cv2.add` 不是一回事。

| 函数 | 作用 | 注意 |
| --- | --- | --- |
| `cv2.add` | 像素相加 | 超过 255 截为 255 |
| `cv2.subtract` | 像素相减 | 小于 0 截为 0 |
| `cv2.multiply` | 像素相乘 | 超过 255 截为 255 |
| `cv2.divide` | 像素相除 | 除数为 0 时结果为 0 |
| `cv2.addWeighted` | 按权重混合 | `结果 = img1*α + img2*β + γ` |
| `cv2.bitwise_and` | 按位与 | 掩码、抠图 |
| `cv2.bitwise_or` | 按位或 | 叠加 |
| `cv2.bitwise_not` | 按位取反 | 黑白颠倒 |
| `cv2.bitwise_xor` | 按位异或 | 找两图差异 |

```python
import cv2

img1 = cv2.imread("image1.jpg")
img2 = cv2.imread("image2.jpg")

added = cv2.add(img1, img2)
subbed = cv2.subtract(img1, img2)
mul = cv2.multiply(img1, img2)
div = cv2.divide(img1, img2)

blend = cv2.addWeighted(img1, 0.7, img2, 0.3, 0)

and_img = cv2.bitwise_and(img1, img2)
or_img = cv2.bitwise_or(img1, img2)
not_img = cv2.bitwise_not(img1)
xor_img = cv2.bitwise_xor(img1, img2)
```

用掩码把一张图的局部嵌进另一张图时，先做一张单通道掩码，再用 `bitwise_and` 分别保留背景和前景，最后 `add`。

---

## 8. 阈值

阈值把灰度图分成两类像素。输入一般是单通道。

`retval, dst = cv2.threshold(src, thresh, maxval, type)`

| type | 规则 |
| --- | --- |
| `THRESH_BINARY` | 大于阈值取 `maxval`，否则 0 |
| `THRESH_BINARY_INV` | 与上面相反 |
| `THRESH_TRUNC` | 大于阈值的截成阈值，其余不变 |
| `THRESH_TOZERO` | 大于阈值的保留，其余变 0 |
| `THRESH_TOZERO_INV` | 大于阈值的变 0，其余保留 |

光照不均时用自适应阈值。每个像素的阈值来自邻域：

`cv2.adaptiveThreshold(src, maxValue, adaptiveMethod, thresholdType, blockSize, C)`

- `ADAPTIVE_THRESH_MEAN_C`：邻域均值减常数 `C`。
- `ADAPTIVE_THRESH_GAUSSIAN_C`：邻域高斯加权均值减 `C`。
- `blockSize` 必须是奇数。

Otsu 适合直方图有两个峰的图。它自己找阈值，调用时把 `thresh` 设为 0，类型加上 `THRESH_OTSU`。返回值 `retval` 就是它选出的阈值。

```python
import cv2

img = cv2.imread("image.jpg", cv2.IMREAD_GRAYSCALE)

_, binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

adaptive = cv2.adaptiveThreshold(
    img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2
)
adaptive_g = cv2.adaptiveThreshold(
    img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
)

otsu_val, otsu = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
print("Otsu 阈值:", otsu_val)
```

---

## 9. 平滑

平滑用邻域像素替换中心像素，用来压噪声，也会抹掉细节。

| 方法 | 函数 | 特点 |
| --- | --- | --- |
| 均值 | `cv2.blur(img, (k, k))` | 邻域平均，边缘一起糊 |
| 高斯 | `cv2.GaussianBlur(img, (k, k), sigmaX)` | 中心权重大；`sigmaX=0` 时按核大小自动算 |
| 中值 | `cv2.medianBlur(img, k)` | `k` 为奇数；椒盐噪声效果好 |
| 双边 | `cv2.bilateralFilter(img, d, sigmaColor, sigmaSpace)` | 同时看距离和颜色差，更容易保住边缘 |

```python
import cv2

image = cv2.imread("image.jpg")
mean = cv2.blur(image, (5, 5))
gauss = cv2.GaussianBlur(image, (5, 5), 0)
median = cv2.medianBlur(image, 5)
bilateral = cv2.bilateralFilter(image, 9, 75, 75)
```

核越大越糊。高斯核的宽高必须是正奇数。双边滤波里，`d` 是邻域直径，两个 `75` 分别是颜色差和空间距离的尺度。

---

## 10. 形态学

形态学主要处理二值图（白前景、黑背景）。结构元素可用 `np.ones((5, 5), np.uint8)`，或 `cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))`。形状还有 `MORPH_ELLIPSE`、`MORPH_CROSS`。

| 操作 | 写法 | 效果 |
| --- | --- | --- |
| 腐蚀 | `cv2.erode(src, kernel, iterations=1)` | 前景变瘦，去掉小白点、断开粘连 |
| 膨胀 | `cv2.dilate(src, kernel, iterations=1)` | 前景变胖，填小洞、接上断裂 |
| 开运算 | `morphologyEx(..., MORPH_OPEN, kernel)` | 先腐蚀后膨胀，去小物体 |
| 闭运算 | `morphologyEx(..., MORPH_CLOSE, kernel)` | 先膨胀后腐蚀，填小孔 |
| 梯度 | `MORPH_GRADIENT` | 膨胀减腐蚀，得到边缘 |
| 顶帽 | `MORPH_TOPHAT` | 原图减开运算，留下比背景亮的小物体 |
| 黑帽 | `MORPH_BLACKHAT` | 闭运算减原图，留下比背景暗的小物体 |

```python
import cv2
import numpy as np

image = cv2.imread("image.png", cv2.IMREAD_GRAYSCALE)
kernel = np.ones((5, 5), np.uint8)

eroded = cv2.erode(image, kernel, iterations=1)
dilated = cv2.dilate(image, kernel, iterations=1)
opened = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
closed = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
gradient = cv2.morphologyEx(image, cv2.MORPH_GRADIENT, kernel)
tophat = cv2.morphologyEx(image, cv2.MORPH_TOPHAT, kernel)
blackhat = cv2.morphologyEx(image, cv2.MORPH_BLACKHAT, kernel)
```

`iterations` 越大，腐蚀和膨胀越狠。开运算去噪、闭运算补洞，是二值图里最常用的两步。

---

## 11. 边缘检测

边缘是亮度变化剧烈的地方。

| 函数 | 算法 | 适用 |
| --- | --- | --- |
| `cv2.Canny` | 高斯平滑、梯度、非极大值抑制、双阈值连接 | 通用，噪声抑制较好 |
| `cv2.Sobel` | 一阶导数，可分水平和垂直 | 看某个方向的边缘 |
| `cv2.Scharr` | Sobel 的 3×3 加强版 | 更细的边缘 |
| `cv2.Laplacian` | 二阶导数 | 边缘和角点，对噪声敏感 |

Canny 的步骤：高斯去噪，Sobel 求梯度幅值和方向，沿梯度方向只留局部最大，再用低阈值、高阈值把边缘分成弱边缘和强边缘，只保留连到强边缘上的弱边缘。

```python
edges = cv2.Canny(gray, threshold1, threshold2, apertureSize=3, L2gradient=False)
```

`threshold1` 是低阈值，`threshold2` 是高阈值。`apertureSize` 是内部 Sobel 核大小。`L2gradient=True` 时用平方和开方，否则用绝对值之和。

Sobel 的 3×3 核：

```text
水平 Gx          垂直 Gy
-1  0  1        -1 -2 -1
-2  0  2         0  0  0
-1  0  1         1  2  1
```

梯度幅值 `G = sqrt(Gx^2 + Gy^2)`。输出深度用 `cv2.CV_64F`，否则负梯度会被截成 0。显示前可取绝对值再转成 `uint8`。

Laplacian 常用核：

```text
 0  1  0
 1 -4  1
 0  1  0
```

```python
import cv2
import numpy as np

image = cv2.imread("image.jpg", cv2.IMREAD_GRAYSCALE)

edges = cv2.Canny(image, 100, 200)

sobel_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
sobel_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
sobel = cv2.magnitude(sobel_x, sobel_y)

scharr_x = cv2.Scharr(image, cv2.CV_64F, 1, 0)
scharr_y = cv2.Scharr(image, cv2.CV_64F, 0, 1)

lap = cv2.Laplacian(image, cv2.CV_64F)

def show_float(name, mat):
    vis = cv2.convertScaleAbs(mat)
    cv2.imshow(name, vis)

show_float("canny", edges)
show_float("sobel", sobel)
show_float("laplacian", lap)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

`ksize=-1` 传给 `Sobel` 时，实际走的就是 Scharr。

---

## 12. 轮廓

轮廓是二值图里同色区域的边界点集。流程：灰度、二值化（或 Canny），再 `findContours`。白像素被视为前景。

```python
contours, hierarchy = cv2.findContours(binary, mode, method)
```

| mode | 含义 |
| --- | --- |
| `RETR_EXTERNAL` | 只要最外层 |
| `RETR_LIST` | 全部轮廓，不建层次 |
| `RETR_TREE` | 全部轮廓，并记录谁包含谁 |
| `RETR_CCOMP` | 两层：外轮廓和洞 |

| method | 含义 |
| --- | --- |
| `CHAIN_APPROX_NONE` | 保存边界上每个点 |
| `CHAIN_APPROX_SIMPLE` | 水平、垂直、对角线只留端点 |

`drawContours(image, contours, index, color, thickness)` 中，`index=-1` 画全部；`thickness<0` 会填充。

| 函数 | 结果 |
| --- | --- |
| `contourArea(c)` | 面积。`oriented=True` 时带符号 |
| `arcLength(c, closed)` | 周长。闭合轮廓把 `closed` 设为 `True` |
| `boundingRect(c)` | 正外接矩形 `(x, y, w, h)` |
| `minAreaRect(c)` | 可旋转的最小矩形：中心、宽高、角度 |
| `minEnclosingCircle(c)` | 圆心和半径 |
| `approxPolyDP(c, epsilon, closed)` | 多边形近似。`epsilon` 越小越贴原轮廓，常用周长的 1% 到 3% |

```python
import cv2
import numpy as np

image = cv2.imread("image.png")
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
_, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
cv2.drawContours(image, contours, -1, (0, 255, 0), 2)

for contour in contours:
    area = cv2.contourArea(contour)
    if area < 100:
        continue
    perimeter = cv2.arcLength(contour, True)

    x, y, w, h = cv2.boundingRect(contour)
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

    rect = cv2.minAreaRect(contour)
    box = cv2.boxPoints(rect).astype(np.int32)
    cv2.drawContours(image, [box], 0, (0, 0, 255), 2)

    (cx, cy), radius = cv2.minEnclosingCircle(contour)
    cv2.circle(image, (int(cx), int(cy)), int(radius), (255, 0, 0), 2)

    epsilon = 0.01 * perimeter
    approx = cv2.approxPolyDP(contour, epsilon, True)
    cv2.drawContours(image, [approx], 0, (0, 255, 255), 2)
    print(f"面积 {area:.1f}  周长 {perimeter:.1f}  近似顶点数 {len(approx)}")

cv2.imshow("Contours", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

顶点数可以粗分形状：三角形 3、矩形 4、圆则远大于 4。面积和周长还可用来丢掉噪点轮廓。

---

## 13. 直方图

直方图统计每个亮度出现了多少次。横轴是像素值 0–255，纵轴是个数。直方图挤在左边说明偏暗，铺得比较开说明对比度更高。彩色图按 B、G、R 三个通道分别统计。

```python
hist = cv2.calcHist(images, channels, mask, histSize, ranges)
```

- `images`：图像列表，例如 `[img]`。
- `channels`：灰度用 `[0]`；彩色 BGR 里 0 是蓝、1 是绿、2 是红。
- `mask`：只统计掩码里的像素；不需要就传 `None`。
- `histSize`：通常 `[256]`。
- `ranges`：通常 `[0, 256]`，右端不包含。
- `accumulate=True` 时在已有直方图上累加，而不是清零重算。

`cv2.equalizeHist` 把灰度直方图拉匀，用来提高对比度。彩色图不要直接对 BGR 做，应拆开通道各自均衡，或转到 YUV / HSV 后只均衡亮度通道，否则颜色会偏。

`cv2.compareHist(h1, h2, method)` 比较两张直方图。`HISTCMP_CORREL` 是相关，越接近 1 越像。另外还有卡方 `HISTCMP_CHISQR`、相交 `HISTCMP_INTERSECT`、巴氏距离 `HISTCMP_BHATTACHARYYA`。

```python
import cv2
import matplotlib.pyplot as plt

gray = cv2.imread("image.jpg", cv2.IMREAD_GRAYSCALE)
hist = cv2.calcHist([gray], [0], None, [256], [0, 256])

plt.plot(hist)
plt.title("Grayscale Histogram")
plt.xlabel("Pixel Value")
plt.ylabel("Count")
plt.show()

equalized = cv2.equalizeHist(gray)

color = cv2.imread("image.jpg")
for i, name in enumerate(("b", "g", "r")):
    ch = cv2.calcHist([color], [i], None, [256], [0, 256])
    plt.plot(ch, color=name)
plt.title("Color Histogram")
plt.show()

b, g, r = cv2.split(color)
merged = cv2.merge([cv2.equalizeHist(b), cv2.equalizeHist(g), cv2.equalizeHist(r)])

hist2 = cv2.calcHist([equalized], [0], None, [256], [0, 256])
similarity = cv2.compareHist(hist, hist2, cv2.HISTCMP_CORREL)
print("相关度", similarity)
```

---

## 14. 视频

视频是一串帧。`VideoCapture` 打开文件或摄像头，`read` 每次拿一帧。`VideoWriter` 把帧写回去。

```python
cap = cv2.VideoCapture(source)          # 文件路径，或摄像头编号 0
ok, frame = cap.read()                  # ok 为 False 表示结束或失败
cap.get(cv2.CAP_PROP_FPS)
cap.get(cv2.CAP_PROP_FRAME_WIDTH)
cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
cap.release()
```

```python
import cv2

cap = cv2.VideoCapture("example.mp4")  # 摄像头则写成 0
if not cap.isOpened():
    raise SystemExit("打不开视频")

fps = cap.get(cv2.CAP_PROP_FPS) or 25
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fourcc = cv2.VideoWriter_fourcc(*"XVID")
out = cv2.VideoWriter("output.avi", fourcc, fps, (w, h))

while True:
    ok, frame = cap.read()
    if not ok:
        break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # VideoWriter 要三通道，灰度要转回 BGR
    out.write(cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR))
    cv2.imshow("Video", gray)
    if cv2.waitKey(25) & 0xFF == ord("q"):
        break

cap.release()
out.release()
cv2.destroyAllWindows()
```

`waitKey` 的参数是毫秒。25 毫秒大约对应 40 帧每秒的刷新上限；要按原速播放，可改成 `int(1000 / fps)`。

相邻帧做差，是最简单的运动检测：

```python
import cv2

cap = cv2.VideoCapture("example.mp4")
ok, prev = cap.read()
prev_gray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)

while True:
    ok, frame = cap.read()
    if not ok:
        break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    diff = cv2.absdiff(prev_gray, gray)
    _, mask = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)
    cv2.imshow("motion", mask)
    prev_gray = gray
    if cv2.waitKey(25) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
```

摄像头画面上写字：

```python
cv2.putText(frame, "Hello", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
```

`org` 是文字左下角。字体常用 `FONT_HERSHEY_SIMPLEX`。

---

## 15. MeanShift 与 CamShift

两者都用颜色直方图做跟踪。先在第一帧框住目标，把 ROI 转到 HSV，只取色调 H 的直方图，再用反向投影得到「每个像素有多像目标」的图，然后移动窗口。

MeanShift：窗口大小固定，反复把窗口中心移到反向投影的质心。计算少，目标大小变化时框不会跟着变，快速移动和遮挡也容易丢。

CamShift：在 MeanShift 上继续调整窗口的宽高和角度，能跟上缩放和旋转，计算稍多。

终止条件 `TERM_CRITERIA_EPS | TERM_CRITERIA_COUNT` 表示：移动小于 1 像素，或迭代满 10 次，就停。

下面用鼠标在第一帧上拖一个矩形。不拖的话，把 `track_window` 改成已知的 `(x, y, w, h)`。

```python
import cv2
import numpy as np

cap = cv2.VideoCapture("video.mp4")
ok, frame = cap.read()
if not ok:
    raise SystemExit("读不到第一帧")

x, y, w, h = cv2.selectROI("pick", frame, fromCenter=False, showCrosshair=True)
cv2.destroyWindow("pick")
track_window = (x, y, w, h)

roi = frame[y:y + h, x:x + w]
hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
mask = cv2.inRange(hsv_roi, np.array((0, 60, 32)), np.array((180, 255, 255)))
roi_hist = cv2.calcHist([hsv_roi], [0], mask, [180], [0, 180])
cv2.normalize(roi_hist, roi_hist, 0, 255, cv2.NORM_MINMAX)

term = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)

while True:
    ok, frame = cap.read()
    if not ok:
        break
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    back = cv2.calcBackProject([hsv], [0], roi_hist, [0, 180], 1)

    # MeanShift：窗口大小不变
    _, track_window = cv2.meanShift(back, track_window, term)
    x, y, w, h = track_window
    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # 若改用 CamShift，用下面三行替换上面的矩形：
    # rotated, track_window = cv2.CamShift(back, track_window, term)
    # box = cv2.boxPoints(rotated).astype(np.int32)
    # cv2.polylines(frame, [box], True, (0, 255, 0), 2)

    cv2.imshow("track", frame)
    if cv2.waitKey(30) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
```

HSV 里 H 的范围是 0–180（OpenCV 把 0–360 度折半存储）。`inRange` 去掉低饱和、低亮度的像素，避免灰背景污染直方图。目标和背景颜色接近时，这两种方法都会跟错。

| | MeanShift | CamShift |
| --- | --- | --- |
| 窗口 | 固定 | 大小和方向会变 |
| 场景 | 目标尺度稳定 | 有缩放、旋转 |
| 开销 | 更低 | 稍高 |

---

## 16. 背景减除（MOG、MOG2）

背景减除为每个像素建立背景模型，当前帧对不上模型的地方就是前景。背景会变（光照、树叶），所以模型要逐帧更新。

MOG 用固定个数的高斯分布描述一个像素。MOG2 会增减高斯个数，背景变化更快时更合适，计算也更多。前景掩码里，白色是前景，黑色是背景，灰色阴影有时会被单独标出。

MOG2 在主库里：

```python
import cv2

cap = cv2.VideoCapture("video.mp4")
mog2 = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=16, detectShadows=True)

while True:
    ok, frame = cap.read()
    if not ok:
        break
    mask = mog2.apply(frame)
    cv2.imshow("frame", frame)
    cv2.imshow("mask", mask)
    if cv2.waitKey(30) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
```

`history` 是用来建模的帧数。`varThreshold` 越大，越不容易把像素判成前景。`detectShadows=True` 时阴影标成灰色（值 127），而不是纯白前景。

旧的 MOG 在扩展模块 `cv2.bgsegm` 里，需要 `opencv-contrib-python`：

```python
mog = cv2.bgsegm.createBackgroundSubtractorMOG()
mask = mog.apply(frame)
```

| | MOG | MOG2 |
| --- | --- | --- |
| 高斯个数 | 固定 | 动态增减 |
| 适应变化 | 较慢 | 较快 |
| 计算 | 较少 | 较多 |
| 所在包 | contrib 的 `bgsegm` | 主库 |

掩码上还可以接形态学开运算，去掉零散噪点，再交给轮廓去框出运动物体。

---

## 17. 人脸检测（Haar）

Haar 分类器由 Viola 和 Jones 在 2001 年提出。它用相邻矩形的像素和之差描述边缘、线条，再用 AdaBoost 串成级联分类器：大部分背景在前几级就被丢掉。OpenCV 自带训练好的 XML，正面人脸用 `haarcascade_frontalface_default.xml`。同目录还有眼睛、微笑等模型。

检测在灰度图上进行。`detectMultiScale` 返回 `(x, y, w, h)` 列表。

- `scaleFactor`：图像金字塔每次缩小的比例，1.1 表示缩小 10%。越小越慢、越不容易漏。
- `minNeighbors`：候选框周围要有多少个邻居才保留。越大越严，误检更少。
- `minSize`：小于该尺寸的脸忽略。

```python
import cv2

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)
if face_cascade.empty():
    raise SystemExit("分类器没有加载成功")

image = cv2.imread("image.jpg")
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

for (x, y, w, h) in faces:
    cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)

cv2.imshow("faces", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

视频里把上面的检测放进 `VideoCapture` 循环即可，每一帧转灰度、检测、画框。侧脸、太小的脸、强逆光时，这个模型会漏。它只回答「像不像正面人脸」，不做身份识别。

---

## 18. 物体识别：模板匹配

模板匹配在大图上滑动一块模板，逐位置算相似度。物体的大小和朝向要和模板差不多，否则对不上。

`result = cv2.matchTemplate(image, template, method)`

结果比原图小：高为 `H - h + 1`，宽为 `W - w + 1`。

| 方法 | 最好的匹配 |
| --- | --- |
| `TM_SQDIFF`、`TM_SQDIFF_NORMED` | 值越小越好，看 `minMaxLoc` 的最小值 |
| `TM_CCORR`、`TM_CCORR_NORMED` | 值越大越好 |
| `TM_CCOEFF`、`TM_CCOEFF_NORMED` | 值越大越好，归一化相关系数最常用 |

只找一处时用 `cv2.minMaxLoc`。允许多次出现时，用阈值筛 `result`，再画所有矩形。

```python
import cv2
import numpy as np

img = cv2.imread("target_image.jpg", cv2.IMREAD_GRAYSCALE)
template = cv2.imread("template_image.jpg", cv2.IMREAD_GRAYSCALE)
h, w = template.shape[:2]

result = cv2.matchTemplate(img, template, cv2.TM_CCOEFF_NORMED)
threshold = 0.8
ys, xs = np.where(result >= threshold)

vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
for x, y in zip(xs, ys):
    cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 255, 0), 2)

min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
print("最佳位置", max_loc, "得分", max_val)

cv2.imshow("match", vis)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

相邻像素常会同时超过阈值，矩形会叠成一堆。需要单个框时，只取 `max_loc`，或对候选框做非极大值抑制。

---

## 19. 图像拼接

有重叠的两张图可以拼成一张更大的图。手工流程是：检测特征、匹配、求单应性矩阵、透视变换后贴到一起。

1. 读入两图，转灰度。
2. `SIFT_create().detectAndCompute` 得到关键点和描述子。SIFT 在 OpenCV 4.4 及之后的主库里可用。
3. `BFMatcher.knnMatch(..., k=2)` 为每个点找两个最近邻。
4. Lowe 比率测试：最近距离 `< 0.75 * 次近距离` 才留下。
5. 好匹配多于 4 个（实用里常要求多于 10 个）时，`findHomography(..., RANSAC, 5.0)` 求 3×3 单应性。`5.0` 是内点重投影误差阈值，单位像素。
6. `warpPerspective` 把第一张图变到第二张图的坐标系，再把第二张图贴回去。

直接贴会有一条硬缝。要更自然，可以在重叠区按距离做加权融合。OpenCV 也提供 `cv2.Stitcher_create()`，内部包含特征、曝光补偿和融合，两张图重叠足够时更省事。

```python
import cv2
import numpy as np

image1 = cv2.imread("image1.jpg")
image2 = cv2.imread("image2.jpg")
gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

sift = cv2.SIFT_create()
kp1, des1 = sift.detectAndCompute(gray1, None)
kp2, des2 = sift.detectAndCompute(gray2, None)

matcher = cv2.BFMatcher()
pairs = matcher.knnMatch(des1, des2, k=2)
good = [m for m, n in pairs if m.distance < 0.75 * n.distance]

if len(good) <= 10:
    raise SystemExit("有效匹配太少")

src = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
dst = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
H, status = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)

h1, w1 = image1.shape[:2]
h2, w2 = image2.shape[:2]
canvas = cv2.warpPerspective(image1, H, (w1 + w2, max(h1, h2)))
canvas[0:h2, 0:w2] = image2

cv2.imshow("stitched", canvas)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

这段假设第二张图落在画布左上角，适合大致左右排列、重叠充分的照片。上下排列或重叠方式不同时，要先用单应性变换四个角点，算出能装下两张图的画布，再加一个平移，避免坐标变成负数。

一行接口：

```python
stitcher = cv2.Stitcher_create()
status, pano = stitcher.stitch([image1, image2])
# status == cv2.Stitcher_OK 时 pano 可用
```

---

## 20. 简单滤镜

| 效果 | 做法 |
| --- | --- |
| 灰度 | `cvtColor(..., COLOR_BGR2GRAY)` |
| 怀旧 | 按固定系数混合 BGR，压低蓝色 |
| 浮雕 | 灰度图上用差分卷积核 |
| 模糊 | `GaussianBlur`，核越大越糊 |
| 锐化 | 中心为正、四周为负的卷积核 |
| 边缘 | 灰度图上 `Canny` |

卷积用 `cv2.filter2D(src, ddepth, kernel)`。`ddepth=-1` 表示输出深度与输入相同。

怀旧（sepia）必须用原始的 B、G、R 来算三个新通道。若先改写 `r` 再用它算 `g` 和 `b`，颜色会错。

```python
import cv2
import numpy as np

image = cv2.imread("input.jpg")

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

b, g, r = cv2.split(image)
b = b.astype(np.float32)
g = g.astype(np.float32)
r = r.astype(np.float32)
new_r = np.clip(0.393 * r + 0.769 * g + 0.189 * b, 0, 255)
new_g = np.clip(0.349 * r + 0.686 * g + 0.168 * b, 0, 255)
new_b = np.clip(0.272 * r + 0.534 * g + 0.131 * b, 0, 255)
vintage = cv2.merge([new_b.astype(np.uint8), new_g.astype(np.uint8), new_r.astype(np.uint8)])

emboss_kernel = np.array([[-2, -1, 0],
                          [-1,  1, 1],
                          [ 0,  1, 2]], dtype=np.float32)
emboss = cv2.filter2D(gray, -1, emboss_kernel)

blurred = cv2.GaussianBlur(image, (15, 15), 0)

sharpen_kernel = np.array([[ 0, -1,  0],
                           [-1,  5, -1],
                           [ 0, -1,  0]], dtype=np.float32)
sharp = cv2.filter2D(image, -1, sharpen_kernel)

edges = cv2.Canny(gray, 100, 200)

cv2.imshow("gray", gray)
cv2.imshow("vintage", vintage)
cv2.imshow("emboss", emboss)
cv2.imshow("blur", blurred)
cv2.imshow("sharp", sharp)
cv2.imshow("edges", edges)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

锐化核的元素相加为 1，整体亮度大致不变，中心被加强。浮雕核沿一个方向做差分，所以亮暗会出现在边缘两侧。

---

## 章节对照

| 笔记 | 对应内容 |
| --- | --- |
| 1–2 | 简介、Python 安装 |
| 3 | 入门实例：读、显示、按键保存 |
| 4 | 基础模块 |
| 5 | 图像处理基础 |
| 6 | 像素、ROI、通道、缩放旋转平移翻转 |
| 7 | 加减乘除、位运算、加权混合 |
| 8 | 固定阈值、自适应阈值、Otsu |
| 9 | 均值、高斯、中值、双边 |
| 10 | 腐蚀、膨胀、开闭、梯度、顶帽、黑帽 |
| 11 | Canny、Sobel、Scharr、Laplacian |
| 12 | 轮廓查找、绘制与几何量 |
| 13 | 直方图、均衡化、比较 |
| 14 | 视频读取、写入、帧差 |
| 15 | MeanShift、CamShift |
| 16 | MOG、MOG2 |
| 17 | Haar 人脸检测 |
| 18 | 模板匹配 |
| 19 | 特征匹配与拼接 |
| 20 | 灰度、怀旧、浮雕、模糊、锐化、边缘 |
