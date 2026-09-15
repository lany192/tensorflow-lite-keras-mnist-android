#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""训练一个识别**中文简体汉字**的 TFLite 模型，供 Android 应用使用。

与 keras_mnist_tflite.py 的关系：两者共享同一条工程约定 —— **Python 侧的预处理必须复刻
Android 侧的预处理**。数字脚本为此写了 _normalize_like_android 并让 INK_THRESHOLD 与
InkSegmenter.inkThreshold 对齐；本脚本沿用这套做法，但预处理契约**不同**，见 preprocess_gray。

## 与数字链路的三个关键差异

1. **单字，不切分。** 数字链路靠 InkSegmenter 把画布切成一串字形。汉字不能复用那套几何：
   splitAspectFactor=1.35 会把「一/二/三/川」切开，dropShortGroups 会把「三」的三横当噪点
   整组删掉，而聚组只看 X 交叠会把左右结构的字（氵+青、亻+尔）拆成两半。所以汉字直接取
   **整画布的紧包围盒**，一个画布一个字。

2. **输入 64x64 / 内容区 56px**，而不是 28x28 / 20px。汉字笔画密度远高于数字。

3. **包围盒居中，而不是质心居中。** 数字脚本用质心居中；汉字改用 bbox 居中，因为质心居中
   会把左右结构的字（如「们」）整体拉偏 —— 氵比「尔」重，质心明显偏左。
   **这条差异是刻意的，不要为了"和数字脚本统一"而改回去。**

## 两条数据路径

- `--data synthetic`：用系统字体渲染 + 强形变增强。零数据集依赖，用于**验证链路正确性**。
  **警告：印刷体与真人手指书写差距显著，这条路训出的模型在真实手写上没有可用准确率。**
  真实效果必须走 hwdb。
- `--data hwdb`：CASIA-HWDB1.1（3755 类真实手写，学术免费，需自行下载）。
  下载：http://www.nlpr.ia.ac.cn/databases/handwriting/Download.html

## 本机约束

开发机是 Intel i5-10500 纯 CPU，全量 100 万样本单 epoch 要数小时。所以本脚本规模全可配：
小配置验链路，同一份脚本放大参数即可上 GPU 正式训练。

```bash
# 本机冒烟
python chinese_character_recognition.py --data synthetic --chars 300 \
    --samples-per-class 40 --epochs 3
# 正式训练
python chinese_character_recognition.py --data hwdb --hwdb-dir ~/HWDB1.1 --chars 3755 --epochs 40
# 只校验字符集
python chinese_character_recognition.py --dry-run
# 验证 .gnt 解析（拿到数据后先跑这个）
python chinese_character_recognition.py --inspect-gnt ~/HWDB1.1/HWDB1.1trn_gnt/001-a.gnt
```
"""

import argparse
import struct
import sys
from pathlib import Path

import numpy as np
import tensorflow as tf
from PIL import Image, ImageDraw, ImageFont
from tensorflow.keras import layers, models

# ============================================================================
# 常量
# ============================================================================

# 与 Android 侧 SegmentConfig.inkThreshold 一致。它**只**用来决定紧包围盒的边界，
# 预处理内部不再做第二次阈值处理 —— 与 CLAUDE.md "墨迹阈值只存在于 SegmentConfig.inkThreshold"
# 这条约定同源。若 Android 侧改了阈值，这里必须同步。
INK_THRESHOLD = 32

# GB2312-80 一级汉字（16~55 区）恰好 3755 个，与 CASIA-HWDB1.1 的类别数一致。
GB2312_LEVEL1_COUNT = 3755
GB2312_LEVEL1_FIRST = "啊"
GB2312_LEVEL1_LAST = "座"

# 内容区占输入画布的比例。64 -> 56。与 Android 侧 ChinesePreprocessor 必须一致。
CONTENT_RATIO = 0.875

# 合成数据的渲染分辨率。先在高分辨率上画字再做形变，最后才走 preprocess_gray 归一化，
# 这样形变导致的包围盒变化会被如实还原（与数字脚本用 RENDER_SIZE 的理由相同）。
RENDER_SIZE = 256

# 合成渲染后 InkSegmenter 式的合理性下限：墨迹太少说明字体缺字（渲染出空白或豆腐块），
# 墨迹太多说明字形糊成一团。两头都拒绝，避免把垃圾样本喂进训练集。
SYNTHETIC_MIN_INK_RATIO = 0.02
SYNTHETIC_MAX_INK_RATIO = 0.60

# BatchNorm 动量。**这是本脚本最容易踩且最难发现的一个坑，不要改回 Keras 默认的 0.99。**
#
# BN 在训练时用**批次统计**归一化，推理时用**滑动平均**。默认 momentum=0.99 意味着滑动
# 平均的更新极慢：第 n 步后初值仍占 0.99^n，要上万步才收敛。
#
# 后果是灾难性的、而且不报任何错：小规模训练时模型在 training=True 下能拟合到 0.85，
# 但切到 training=False 立刻掉回随机水平（实测：60 类时 0.8567 -> 0.0167）。而
# **TFLite 推理永远只用冻结的滑动统计**，所以"训练看着收敛了"的模型导出后完全无效。
#
# 0.9 是 BN 原论文的推荐值，几百步内就收敛，对大规模训练也无害。
# 数字脚本 keras_mnist_tflite.py 没有这个问题，因为它不用 BatchNorm。
BN_MOMENTUM = 0.9


# ============================================================================
# 字符集
# ============================================================================

def build_charset():
    """按 GB2312 区位顺序生成一级汉字表（3755 个）。

    类别下标 = 列表下标 = GB2312 区位顺序，与 CASIA-HWDB1.1 的标签顺序一致，
    所以 HWDB 的 tagcode 可以直接用 charset.index(ch) 反查类别。

    第 55 区末尾（0xD7F9~0xD7FE）是 5 个空位，用 UnicodeDecodeError 自然跳过 ——
    不要改成硬编码 3755 次循环，那样会把空位当成字。
    """
    chars = []
    for hi in range(0xB0, 0xD8):          # 16~55 区
        for lo in range(0xA1, 0xFF):
            try:
                chars.append(bytes([hi, lo]).decode("gb2312"))
            except UnicodeDecodeError:
                continue
    if len(chars) != GB2312_LEVEL1_COUNT:
        raise RuntimeError(
            f"GB2312 一级汉字应为 {GB2312_LEVEL1_COUNT} 个，实际得到 {len(chars)} 个。"
            f"字符集错位会让模型输出与 chars.txt 全部对不上，必须在训练前解决。"
        )
    return chars


# ============================================================================
# 预处理：Python 与 Android 的唯一契约
# ============================================================================

def preprocess_gray(gray, size):
    """把「白底黑笔」的灰度图归一化成模型输入，返回 float32[size, size, 1] 或 None。

    这是与 Android 侧 ChinesePreprocessor 的契约，六步：

      1. ink = 255 - gray            （Android 导出的是白底黑笔，墨迹是低灰度）
      2. 紧包围盒                     （单字路径，不切分）
      3. 等比缩放到最长边 CONTENT_SIZE（保持宽高比，绝不拉伸）
      4. **包围盒居中**放到 size×size
      5. 除以本字最大 ink             （逐字独立，与数字链路一致）
      6. 输出 墨迹=1.0 / 背景=0.0

    与数字链路的两处差异（输入尺寸、居中方式）见模块 docstring，是刻意的。

    无墨时返回 None —— 调用方必须处理，不要用零图顶替（零图是合法的"空画布"输入，
    会让模型对空画布的响应被训练成随机类别）。
    """
    ink = 255.0 - np.asarray(gray, dtype=np.float32)
    mask = ink > INK_THRESHOLD
    if not mask.any():
        return None

    rows = np.nonzero(mask.any(axis=1))[0]
    cols = np.nonzero(mask.any(axis=0))[0]
    crop = ink[rows[0]:rows[-1] + 1, cols[0]:cols[-1] + 1]

    content = int(round(size * CONTENT_RATIO))
    height, width = crop.shape
    scale = content / max(height, width)
    out_h = max(1, int(round(height * scale)))
    out_w = max(1, int(round(width * scale)))

    # PIL 双线性，对应 Android 的 Bitmap.createScaledBitmap(..., filter=true)。
    # 两者实现不同、不可能逐位相同，但它们都是"双线性"这一档；
    # 数字脚本同样用 tf.image.resize 近似 Android 的双线性，这里沿用同一取舍。
    resized = np.asarray(
        Image.fromarray(crop.astype(np.uint8)).resize((out_w, out_h), Image.BILINEAR),
        dtype=np.float32,
    )

    canvas = np.zeros((size, size), dtype=np.float32)
    top = (size - out_h) // 2
    left = (size - out_w) // 2
    canvas[top:top + out_h, left:left + out_w] = resized

    peak = float(canvas.max())
    if peak > 0.0:
        canvas /= peak
    return canvas[..., np.newaxis]


# ============================================================================
# 数据路径 A：合成数据
# ============================================================================

# 系统字体候选。本机已确认可用的三个；缺字会在渲染后被 ink 比例检查挡掉。
FONT_CANDIDATES = [
    "/System/Library/Fonts/STHeiti Light.ttc",
    "/System/Library/Fonts/STHeiti Medium.ttc",
    "/System/Library/Fonts/Supplemental/Songti.ttc",
    "/System/Library/Fonts/PingFang.ttc",
    "/usr/share/fonts/truetype/arphic/uming.ttc",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
]


def discover_fonts(explicit=None, probe_char="永"):
    """挑出真正能渲染汉字的字体。

    必须实渲染验证：字体文件存在不代表含汉字字形，缺字时 PIL 会画出空白或豆腐块，
    而那会变成"合法但错误"的训练样本 —— 比报错更糟。
    """
    paths = [explicit] if explicit else FONT_CANDIDATES
    usable = []
    for path in paths:
        if not Path(path).exists():
            continue
        try:
            font = ImageFont.truetype(path, 64)
            probe = Image.new("L", (96, 96), 255)
            ImageDraw.Draw(probe).text((48, 48), probe_char, font=font, fill=0, anchor="mm")
            arr = np.asarray(probe, dtype=np.float32)
            if (255.0 - arr > INK_THRESHOLD).mean() > SYNTHETIC_MIN_INK_RATIO:
                usable.append(path)
        except OSError:
            continue
    return usable


def elastic_warp(gray, rng, alpha, grid):
    """弹性形变：手写笔画不会像印刷体那样笔直，用低频位移场模拟。

    位移场在粗网格上随机生成再放大到全图，因此形变是**低频**的（整笔弯曲），
    而不是逐像素噪声。用 PIL 做放大插值，避免为一个形变引入 scipy 依赖。
    """
    height, width = gray.shape
    coarse_h = max(2, height // grid)
    coarse_w = max(2, width // grid)

    dx = rng.uniform(-alpha, alpha, (coarse_h, coarse_w)).astype(np.float32)
    dy = rng.uniform(-alpha, alpha, (coarse_h, coarse_w)).astype(np.float32)
    # float32 数组交给 PIL 会自然得到 'F' 模式，不要显式传 mode=（Pillow 13 起已废弃）
    dx = np.asarray(Image.fromarray(dx).resize((width, height), Image.BILINEAR))
    dy = np.asarray(Image.fromarray(dy).resize((width, height), Image.BILINEAR))

    yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)
    sx = np.clip(xx + dx, 0, width - 1)
    sy = np.clip(yy + dy, 0, height - 1)

    x0 = np.floor(sx).astype(np.int32)
    y0 = np.floor(sy).astype(np.int32)
    x1 = np.minimum(x0 + 1, width - 1)
    y1 = np.minimum(y0 + 1, height - 1)
    wx = sx - x0
    wy = sy - y0

    out = (gray[y0, x0] * (1 - wx) * (1 - wy) + gray[y0, x1] * wx * (1 - wy)
           + gray[y1, x0] * (1 - wx) * wy + gray[y1, x1] * wx * wy)
    return out.astype(np.uint8)


def render_synthetic(ch, fonts, rng):
    """渲染一个汉字的合成样本，返回「白底黑笔」的灰度图（尚未归一化）。"""
    font_path = fonts[rng.integers(len(fonts))]
    font_size = int(RENDER_SIZE * rng.uniform(0.70, 0.90))
    font = ImageFont.truetype(font_path, font_size)

    image = Image.new("L", (RENDER_SIZE, RENDER_SIZE), 255)
    draw = ImageDraw.Draw(image)

    # 笔画粗细扰动：stroke_width 相当于给字形描边，能覆盖"写得轻"和"写得重"两端。
    stroke = int(rng.integers(0, max(1, font_size // 45)))
    jitter = 0.03 * RENDER_SIZE
    draw.text(
        (RENDER_SIZE / 2 + rng.uniform(-jitter, jitter),
         RENDER_SIZE / 2 + rng.uniform(-jitter, jitter)),
        ch, font=font, fill=0, anchor="mm",
        stroke_width=stroke, stroke_fill=0,
    )

    if rng.random() < 0.7:
        image = image.rotate(
            rng.uniform(-8.0, 8.0), resample=Image.BICUBIC, fillcolor=255
        )
    if rng.random() < 0.5:
        shear = rng.uniform(-0.18, 0.18)
        image = image.transform(
            image.size, Image.AFFINE, (1.0, shear, 0.0, 0.0, 1.0, 0.0),
            resample=Image.BICUBIC, fillcolor=255,
        )

    gray = np.asarray(image, dtype=np.uint8)
    if rng.random() < 0.8:
        gray = elastic_warp(
            gray, rng,
            alpha=rng.uniform(2.0, 6.0),
            grid=int(rng.integers(16, 40)),
        )
    return gray


def make_synthetic_split(chars, per_class, fonts, size, seed, desc):
    """为每个字生成 per_class 个合成样本。图像以 uint8 存储，与 HWDB 路径一致。"""
    rng = np.random.default_rng(seed)
    total = len(chars) * per_class
    images = np.zeros((total, size, size, 1), dtype=np.uint8)
    labels = np.zeros(total, dtype=np.int64)

    written = 0
    rejected = 0
    for class_index, ch in enumerate(chars):
        produced = 0
        attempts = 0
        while produced < per_class:
            attempts += 1
            if attempts > per_class * 12:
                # 字体覆盖不住这个字，或形变把它推出了合理区间。宁可少样本也不要垃圾样本。
                print(f"  警告：字符「{ch}」只生成 {produced}/{per_class} 个样本，跳过其余")
                break
            gray = render_synthetic(ch, fonts, rng)
            processed = preprocess_gray(gray, size)
            if processed is None:
                rejected += 1
                continue
            ratio = float((processed > 0.5).mean())
            if not (SYNTHETIC_MIN_INK_RATIO <= ratio <= SYNTHETIC_MAX_INK_RATIO):
                # 墨迹比例越界 = 字体缺字画出豆腐块，或形变把字形糊死
                rejected += 1
                continue
            images[written] = (processed * 255.0).round().astype(np.uint8)
            labels[written] = class_index
            written += 1
            produced += 1

    print(f"  {desc}: 生成 {written} 个样本，拒绝 {rejected} 个")
    return images[:written], labels[:written]


# ============================================================================
# 数据路径 B：CASIA-HWDB1.1 (.gnt)
# ============================================================================

# .gnt 是私有二进制格式，公开资料对头部布局的说法**互相矛盾**（样本长度字段有 uint16/uint32
# 两说，tagcode 有大小端两说）。本机没有数据可以验证，所以不能凭记忆赌一种 ——
# 这里枚举候选布局，用全文不变量自动判别。
#
# 每条记录的布局：
#   [长度字段 len_bytes][tagcode 2B][width 2B LE][height 2B LE][pixels w*h]
# 判别依据是三条不变量同时成立：
#   (a) 长度字段 == w*h + len_extra
#   (b) 顺序解析恰好走到文件末尾（offset == 文件长度）
#   (c) tagcode 按该字节序解出的字符能解码为 GB2312，且落在汉字区
_GNT_LAYOUTS = [
    # (名称, 长度字段字节数, 长度字段格式, 长度相对 w*h 多出的字节数, tagcode 格式)
    ("len16+tagBE", 2, "<H", 4, ">H"),
    ("len16+tagLE", 2, "<H", 4, "<H"),
    ("len32+tagBE", 4, "<I", 8, ">H"),
    ("len32+tagLE", 4, "<I", 8, "<H"),
    ("len32self+tagBE", 4, "<I", 4, ">H"),
    ("len32self+tagLE", 4, "<I", 4, "<H"),
]

_GNT_MAX_SIDE = 2048  # HWDB 单字远小于此；越界说明布局判断错了，防止误读导致天文数字的 w*h


def _gb2312_char(code):
    """区位码 -> 字符，非法返回 None。"""
    try:
        return bytes([(code >> 8) & 0xFF, code & 0xFF]).decode("gb2312")
    except UnicodeDecodeError:
        return None


def _parse_gnt_with(data, layout, keep=0):
    """按给定布局解析。返回 (records, 说明)；records 为 None 表示该布局不成立。

    records 元素：(字符, width, height, 灰度 ndarray 或 None)。
    `keep` 控制**保留多少条像素数据**：判别布局时传 0（只校验、不留数据，
    避免为一个 2MB 的 per-writer 文件把几十 MB 像素全拷进内存），
    真正加载时传 -1 表示全留。

    tagcode 的字节序由布局决定：文件里那 2 个字节按 tag_fmt 解出来的值就是 GB2312 区位码，
    所以 `struct.unpack_from(tag_fmt, ...)` 本身就完成了"该按大端还是小端读"的判别。
    """
    _, len_bytes, len_fmt, len_extra, tag_fmt = layout
    total = len(data)
    offset = 0
    records = []
    cjk_hits = 0

    while offset < total:
        if offset + len_bytes + 2 + 4 > total:
            return None, "记录头越界"
        sample_size = struct.unpack_from(len_fmt, data, offset)[0]
        code = struct.unpack_from(tag_fmt, data, offset + len_bytes)[0]
        char = _gb2312_char(code)
        if char is None:
            return None, f"tagcode 0x{code:04x} 不是合法 GB2312"

        p = offset + len_bytes + 2
        width, height = struct.unpack_from("<HH", data, p)
        p += 4
        if width == 0 or height == 0 or width > _GNT_MAX_SIDE or height > _GNT_MAX_SIDE:
            return None, f"尺寸越界 {width}x{height}"
        if sample_size != width * height + len_extra:
            return None, (f"长度字段 {sample_size} != w*h+{len_extra} "
                          f"({width * height + len_extra})")
        if p + width * height > total:
            return None, "像素数据越界"

        pixels = None
        if keep < 0 or len(records) < keep:
            pixels = np.frombuffer(data, dtype=np.uint8,
                                   count=width * height, offset=p)
            pixels = pixels.reshape(height, width).copy()
        records.append((char, width, height, pixels))

        # HWDB1.1 全是汉字，区位码高位落在 0xB0~0xF7（GB2312 一二级汉字区）才算合理。
        # 这是区分大小端的有力佐证：读错字节序时高位会落到非汉字区。
        if 0xB0 <= (code >> 8) <= 0xF7:
            cjk_hits += 1
        offset = p + width * height

    if offset != total:
        return None, f"解析在 {offset} 处结束，文件长度 {total}，未恰好走到末尾"

    ratio = cjk_hits / max(1, len(records))
    return records, f"{len(records)} 条，汉字占比 {ratio:.3f}"


def sniff_gnt_layout(data, verbose=True):
    """自动判别 .gnt 头部布局。返回 (layout, records, 说明)。

    "顺序解析恰好走到文件末尾"是最强的判别条件 —— 它同时钉死了长度字段宽度、
    长度语义和记录边界。多个布局都通过时（理论上可能），选汉字占比最高的那个。
    """
    results = []
    for layout in _GNT_LAYOUTS:
        records, note = _parse_gnt_with(data, layout, keep=0)
        if verbose:
            status = "通过" if records is not None else "不成立"
            print(f"    {layout[0]:<18} {status:<6} {note}")
        if records is not None:
            results.append((layout, records, note))

    if not results:
        raise RuntimeError(
            "无法识别 .gnt 布局：所有候选都未通过全文校验。\n"
            "这多半意味着数据不是原始的 CASIA-HWDB1.1，或者文件被截断/还带着压缩头。\n"
            "请确认解压完整，并把上面每个候选的失败原因贴出来。"
        )

    if len(results) > 1:
        def cjk_ratio(item):
            try:
                return float(item[2].split("汉字占比")[1].strip())
            except (IndexError, ValueError):
                return 0.0

        results.sort(key=cjk_ratio, reverse=True)
        if verbose:
            print(f"    多个布局通过，按汉字占比选用 {results[0][0][0]}")

    return results[0]


def find_hwdb_split(root, *keywords):
    """在 root 下按关键字找一个子目录（大小写不敏感的子串匹配）。"""
    if not root.is_dir():
        return None
    for candidate in sorted(root.iterdir()):
        if not candidate.is_dir():
            continue
        lowered = candidate.name.lower()
        if any(k in lowered for k in keywords):
            return candidate
    return None


def load_hwdb(hwdb_dir, charset, per_class, size):
    """加载 HWDB1.1，返回 (x_tr, y_tr, x_te, y_te)，图像以 **uint8** 存储。

    内存是这里最要紧的约束，所以两条：先读先处理（不缓存原始灰度图），
    并一律以 uint8 落内存，float32 推迟到 tf.data 里按批转换。
    以 size=64 计每样本 4096 字节：3755 类 × 200/类 ≈ 750k 样本 ≈ 3GB，
    若改成 float32 直接翻四倍到 12GB。
    """
    root = Path(hwdb_dir).expanduser()
    if not root.exists():
        raise SystemExit(f"HWDB 目录不存在：{root}")

    trn = find_hwdb_split(root, "trn_gnt", "trn")
    tst = find_hwdb_split(root, "tst_gnt", "tst")
    if trn is None:
        raise SystemExit(
            f"{root} 下没找到训练集目录（形如 HWDB1.1trn_gnt）。\n"
            f"官网下载解压后应是 HWDB1.1trn_gnt / HWDB1.1tst_gnt 两层结构。"
        )

    char_to_index = {ch: i for i, ch in enumerate(charset)}

    def collect(split_dir, cap_per_class, desc):
        files = sorted(split_dir.glob("*.gnt"))
        if not files:
            raise SystemExit(f"{split_dir} 下没有 .gnt 文件")
        print(f"  {desc}: {len(files)} 个文件，每类上限 {cap_per_class}")

        capacity = len(charset) * cap_per_class
        images = np.zeros((capacity, size, size, 1), dtype=np.uint8)
        labels = np.zeros(capacity, dtype=np.int64)
        counts = np.zeros(len(charset), dtype=np.int64)
        written = 0
        skipped = 0
        layout = None

        for path in files:
            data = path.read_bytes()
            if layout is None:
                print(f"  判别 .gnt 布局（用 {path.name}）:")
                layout, _records, note = sniff_gnt_layout(data)
                print(f"    -> 采用 {layout[0]}，{note}")

            records, problem = _parse_gnt_with(data, layout, keep=-1)
            if records is None:
                # 单个文件坏掉不该让整个训练停摆，但必须说出来
                print(f"    跳过 {path.name}：{problem}")
                skipped += 1
                continue

            for char, _w, _h, gray in records:
                index = char_to_index.get(char)
                if index is None or counts[index] >= cap_per_class:
                    continue
                processed = preprocess_gray(gray, size)
                if processed is None:
                    continue
                # 存 uint8，省四分之三内存；float32 在 tf.data 里转
                images[written] = (processed * 255.0).round().astype(np.uint8)
                labels[written] = index
                written += 1
                counts[index] += 1

            if written >= capacity:
                break

        empty = int((counts == 0).sum())
        print(f"    {desc}: {written} 个样本"
              + (f"，{empty} 个字符无样本" if empty else "")
              + (f"，跳过 {skipped} 个文件" if skipped else ""))
        return images[:written], labels[:written]

    print("加载 CASIA-HWDB1.1:")
    x_tr, y_tr = collect(trn, per_class, "训练集")
    # 测试集按训练集的四分之一取，避免训练集小、测试集反而更大
    test_cap = max(1, per_class // 4)
    if tst is None:
        print("  未找到测试集目录，跳过测试集")
        x_te, y_te = (np.zeros((0, size, size, 1), dtype=np.uint8),
                      np.zeros(0, dtype=np.int64))
    else:
        x_te, y_te = collect(tst, test_cap, "测试集")
    return x_tr, y_tr, x_te, y_te


# ============================================================================
# 模型
# ============================================================================

def build_model(size, num_classes):
    """轻量 CNN，输出 logits。

    用 GlobalAveragePooling 而不是 Flatten：3755 类下 Flatten 会先接一个
    8*8*256=16384 维的向量，再乘 3755 类 = 6100 万参数（约 235MB），模型直接失控。
    GAP 把每个通道压成一个数，分类头降到 256*3755 ≈ 96 万参数。

    BatchNorm 的动量固定在 BN_MOMENTUM，**不要改回 Keras 默认的 0.99**。
    理由见该常量的注释 —— 用默认值会让小规模训练导出一个完全无效的模型。
    """
    def conv_block(filters, repeats):
        block = []
        for _ in range(repeats):
            block += [
                layers.Conv2D(filters, 3, padding="same", use_bias=False),
                layers.BatchNormalization(momentum=BN_MOMENTUM),
                layers.ReLU(),
            ]
        return block

    return models.Sequential([
        layers.Input(shape=(size, size, 1)),
        *conv_block(32, 2), layers.MaxPooling2D(),
        *conv_block(64, 2), layers.MaxPooling2D(),
        *conv_block(128, 2), layers.MaxPooling2D(),
        *conv_block(256, 1),
        layers.GlobalAveragePooling2D(),
        layers.Dropout(0.4),
        layers.Dense(num_classes),
    ])


def make_augmentation():
    """只作用于训练集。fill_mode='constant' 补 0（背景），与数字脚本一致。"""
    return tf.keras.Sequential([
        layers.RandomTranslation(0.06, 0.06, fill_mode="constant"),
        layers.RandomRotation(0.04, fill_mode="constant"),
        layers.RandomZoom(0.08, fill_mode="constant"),
    ])


def build_datasets(x_train, y_train, x_test, y_test, batch, seed):
    """uint8 -> float32 在这里做，并把增强挂上。

    图像以 uint8 落内存（见 load_hwdb），所以归一化和类型转换都推迟到这里按批进行。

    增强层**只创建一次**：写进 .map() 的 lambda 里会让每个 batch 都新建一套层，
    既慢又会在 tf.function 追踪期反复建层。
    """
    augmentation = make_augmentation()

    def augment(x, y):
        return augmentation(tf.cast(x, tf.float32) / 255.0, training=True), y

    def normalize(x, y):
        return tf.cast(x, tf.float32) / 255.0, y

    train_ds = (
        tf.data.Dataset.from_tensor_slices((x_train, y_train))
        .shuffle(min(len(x_train), 100_000), seed=seed)
        .batch(batch)
        .map(augment, num_parallel_calls=tf.data.AUTOTUNE)
        .prefetch(tf.data.AUTOTUNE)
    )

    test_ds = None
    if len(x_test) > 0:
        test_ds = (
            tf.data.Dataset.from_tensor_slices((x_test, y_test))
            .batch(batch)
            .map(normalize, num_parallel_calls=tf.data.AUTOTUNE)
            .prefetch(tf.data.AUTOTUNE)
        )
    return train_ds, test_ds


# ============================================================================
# 评估与导出
# ============================================================================

def evaluate_tflite(tflite_model, images, labels, num_classes, batch=256, top_k=5):
    """用 TFLite 解释器而非 Keras 模型复算指标。

    必须这样：报的必须是**转换后**模型的真实表现。量化会掉点，用 Keras 的
    val_accuracy 汇报会让 Android 侧的实际体验与报告不符。
    """
    if len(images) == 0:
        return None

    interpreter = tf.lite.Interpreter(model_content=tflite_model)
    input_detail = interpreter.get_input_details()[0]
    output_detail = interpreter.get_output_details()[0]

    correct_1 = 0
    correct_k = 0
    total = 0

    for start in range(0, len(images), batch):
        chunk = images[start:start + batch]
        n = chunk.shape[0]
        interpreter.resize_tensor_input(input_detail["index"], [n] + list(chunk.shape[1:]))
        interpreter.allocate_tensors()
        interpreter.set_tensor(input_detail["index"],
                               chunk.astype(np.float32) / 255.0)
        interpreter.invoke()
        logits = interpreter.get_tensor(output_detail["index"])
        truth = labels[start:start + batch]

        order = np.argsort(-logits, axis=1)
        correct_1 += int((order[:, 0] == truth).sum())
        correct_k += int((order[:, :top_k] == truth[:, None]).any(axis=1).sum())
        total += n

    return {
        "total": total,
        "top1": correct_1 / total,
        f"top{top_k}": correct_k / total,
        "input_shape": list(input_detail["shape"]),
        "output_shape": list(output_detail["shape"]),
    }


def write_charset(path, charset):
    """写字符表：每行一个字符，UTF-8，行号即类别下标。

    Android 侧靠它把 argmax 下标还原成汉字。行数与模型类别数不一致会静默乱码，
    所以这里强制校验。
    """
    text = "\n".join(charset) + "\n"
    if len(text.rstrip("\n").split("\n")) != len(charset):
        raise RuntimeError("字符表写出后行数不符，多半是字符本身含换行")
    path.write_text(text, encoding="utf-8")


# 预览用的探针字：刻意挑结构上的极端情况 —— 独立横笔画（一/三）、竖向并列（川）、
# 左右结构（们/好）、笔画极密（疆/赢）、易混对（日/曰、未/末）、包围结构（国）。
# 这几类正是数字链路的切分器会误杀的形状，用来看预处理是否把它们完整保留。
PREVIEW_CHARS = "一三川们好疆赢日曰未末口品国我"


def write_preview_montage(path, chars, fonts, size, seed, cols=6, cell=112):
    """把「渲染结果」和「模型实际看到的输入」拼成对照图。

    上半张是渲染出的灰度图，下半张是 preprocess_gray 的输出再放大回同样大小 ——
    归一化只要有一处写错（缩放比例、居中方式、阈值），下半张会立刻看出来：
    字形贴在边上、被裁掉一角、或者糊成一团。
    """
    from PIL import Image as _Image

    font = ImageFont.truetype(fonts[0], 20)
    label_h = 26
    rows = (len(chars) + cols - 1) // cols
    canvas = _Image.new("L", (cols * cell, rows * 2 * (cell + label_h)), 255)
    draw = ImageDraw.Draw(canvas)

    rng = np.random.default_rng(seed)
    for i, ch in enumerate(chars):
        col = i % cols
        row = i // cols

        raw = render_synthetic(ch, fonts, rng)
        processed = preprocess_gray(raw, size)

        raw_cell = _Image.fromarray(raw).convert("L").resize((cell, cell), _Image.BILINEAR)
        raw_y = row * 2 * (cell + label_h)
        canvas.paste(raw_cell, (col * cell, raw_y))
        draw.text((col * cell + 4, raw_y + cell + 2), f"{ch} 渲染", font=font, fill=0)

        input_y = raw_y + cell + label_h
        if processed is None:
            draw.text((col * cell + 4, input_y + cell // 2), "无墨", font=font, fill=0)
        else:
            ink = (255.0 - processed[..., 0] * 255.0).clip(0, 255).astype(np.uint8)
            input_cell = _Image.fromarray(ink).resize((cell, cell), _Image.NEAREST)
            canvas.paste(input_cell, (col * cell, input_y))
        draw.text((col * cell + 4, input_y + cell + 2), f"{ch} 输入", font=font, fill=0)

    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)
    return path


# ============================================================================
# gnt 检视
# ============================================================================

def inspect_gnt(path, preview_dir=None, limit=12):
    """独立验证 .gnt 解析。拿到数据集后先跑这个，别直接开始训练。"""
    file_path = Path(path).expanduser()
    if not file_path.exists():
        raise SystemExit(f"文件不存在：{file_path}")

    data = file_path.read_bytes()
    print(f"文件: {file_path.name}  大小: {len(data)} 字节")
    print("候选布局校验:")
    layout, records, note = sniff_gnt_layout(data)
    print(f"\n采用布局: {layout[0]}  ({note})")

    print(f"\n前 {min(limit, len(records))} 个样本:")
    for char, width, height, gray in records[:limit]:
        print(f"  「{char}」 {width}x{height}  灰度 [{gray.min()}, {gray.max()}]")

    if preview_dir:
        out = Path(preview_dir).expanduser()
        out.mkdir(parents=True, exist_ok=True)
        for i, (char, _w, _h, gray) in enumerate(records[:limit]):
            Image.fromarray(gray).save(out / f"{i:02d}_{char}.png")
        print(f"\n预览已写入 {out}/ —— 请打开确认字形可辨认、不是噪点")


# ============================================================================
# 主流程
# ============================================================================

def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="训练中文简体汉字识别模型并导出 TFLite",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data", choices=["synthetic", "hwdb"], default="synthetic",
                        help="synthetic=字体渲染（仅验证链路）；hwdb=CASIA-HWDB1.1 真实手写")
    parser.add_argument("--chars", type=int, default=0,
                        help="使用字符集前 N 个字（按 GB2312 顺序），0=全部 3755")
    parser.add_argument("--samples-per-class", type=int, default=0,
                        help="每类样本数上限，0=不限（synthetic 默认 40，hwdb 默认 200）")
    parser.add_argument("--size", type=int, default=64, help="输入边长")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch", type=int, default=128)
    parser.add_argument("--hwdb-dir", type=str, default=None,
                        help="包含 HWDB1.1trn_gnt / HWDB1.1tst_gnt 的目录")
    parser.add_argument("--font", type=str, default=None, help="指定合成数据用的字体文件")
    parser.add_argument("--out-dir", type=str, default="result", help="产物输出目录")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry-run", action="store_true",
                        help="只校验字符集与配置，不训练")
    parser.add_argument("--preview", action="store_true",
                        help="渲染「原图 / 模型输入」对照图后退出，用于肉眼检查预处理")
    parser.add_argument("--inspect-gnt", type=str, default=None,
                        help="解析并检视一个 .gnt 文件后退出")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)

    if args.inspect_gnt:
        inspect_gnt(args.inspect_gnt, preview_dir=args.out_dir)
        return 0

    tf.random.set_seed(args.seed)
    np.random.seed(args.seed)

    charset = build_charset()
    print(f"字符集: GB2312 一级汉字 {len(charset)} 个（{charset[0]} … {charset[-1]}）")

    if args.chars > 0:
        if args.chars > len(charset):
            raise SystemExit(f"--chars {args.chars} 超过字符集大小 {len(charset)}")
        charset = charset[:args.chars]
        print(f"按 --chars 取前 {len(charset)} 个（{charset[0]} … {charset[-1]}）")

    if args.preview:
        fonts = discover_fonts(args.font)
        if not fonts:
            raise SystemExit(f"没有找到可渲染汉字的字体，用 --font 指定一个：{FONT_CANDIDATES}")
        out_path = write_preview_montage(
            Path(args.out_dir) / "preview_synthetic.png",
            PREVIEW_CHARS, fonts, args.size, args.seed)
        print(f"对照图已写入 {out_path}")
        print("上排是字体渲染结果，下排是模型实际看到的输入。")
        print("检查：字形居中、不贴边、没被裁角、笔画没糊在一起。")
        return 0

    if args.dry_run:
        print(f"\n配置: data={args.data} size={args.size} 类别数={len(charset)}")
        print(f"内容区: {int(round(args.size * CONTENT_RATIO))}px "
              f"(比例 {CONTENT_RATIO})")
        if args.data == "synthetic":
            fonts = discover_fonts(args.font)
            print(f"可用字体: {len(fonts)}")
            for f in fonts:
                print(f"  {f}")
        else:
            print(f"HWDB 目录: {args.hwdb_dir}")
        print("\n--dry-run：未训练。")
        return 0

    num_classes = len(charset)

    # ---- 数据 ----
    if args.data == "synthetic":
        per_class = args.samples_per_class or 40
        fonts = discover_fonts(args.font)
        if not fonts:
            raise SystemExit(
                "没有找到可渲染汉字的字体。用 --font 指定一个中文字体文件。\n"
                f"已尝试: {FONT_CANDIDATES}"
            )
        print(f"合成数据: {len(fonts)} 个字体，每类 {per_class} 个样本")
        x_train, y_train = make_synthetic_split(
            charset, per_class, fonts, args.size, args.seed, "训练集")
        x_test, y_test = make_synthetic_split(
            charset, max(1, per_class // 5), fonts, args.size, args.seed + 1, "测试集")
    else:
        if not args.hwdb_dir:
            raise SystemExit("--data hwdb 需要 --hwdb-dir")
        per_class = args.samples_per_class or 200
        x_train, y_train, x_test, y_test = load_hwdb(
            args.hwdb_dir, charset, per_class, args.size)

    if len(x_train) == 0:
        raise SystemExit("训练集为空。检查数据路径与每类样本数。")
    if len(x_test) == 0:
        print("警告：测试集为空，将跳过验证（只会在小规模调试时发生）")

    print(f"\n训练集 {x_train.shape}  测试集 {x_test.shape}  "
          f"内存约 {(x_train.nbytes + x_test.nbytes) / 1e6:.0f} MB")

    # ---- 模型 ----
    model = build_model(args.size, num_classes)
    model.summary()
    print(f"参数量: {model.count_params():,}")

    train_ds, test_ds = build_datasets(
        x_train, y_train, x_test, y_test, args.batch, args.seed)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )

    model.fit(
        train_ds,
        epochs=args.epochs,
        validation_data=test_ds,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(
                monitor="val_accuracy" if test_ds else "accuracy",
                patience=5, restore_best_weights=True),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss" if test_ds else "loss",
                factor=0.5, patience=2, min_lr=1e-5),
        ],
    )

    # ---- 导出 ----
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()
    (out_dir / "model.tflite").write_bytes(tflite_model)

    write_charset(out_dir / "chars.txt", charset)

    # 产物自检：Android 侧能否正确映射，全看这几个数是否自洽
    metrics = evaluate_tflite(tflite_model, x_test, y_test, num_classes)
    lines = [
        "中文简体汉字识别模型 训练报告",
        "=" * 40,
        f"数据路径      : {args.data}",
        f"字符数        : {num_classes}",
        f"输入尺寸      : {args.size}x{args.size}",
        f"内容区        : {int(round(args.size * CONTENT_RATIO))}px",
        f"训练样本      : {len(x_train)}",
        f"测试样本      : {len(x_test)}",
        f"参数量        : {model.count_params():,}",
        f"TFLite 体积   : {len(tflite_model) / 1e6:.2f} MB",
    ]
    if metrics:
        lines += [
            "",
            "TFLite 模型指标（转换后复算，非 Keras 指标）",
            f"  top-1       : {metrics['top1']:.4f}",
            f"  top-5       : {metrics['top5']:.4f}",
        ]
        print(f"\nTFLite 模型 top-1: {metrics['top1']:.4f}  top-5: {metrics['top5']:.4f}")
    lines += [
        "",
        f"输入张量      : {metrics['input_shape'] if metrics else 'n/a'}",
        f"输出张量      : {metrics['output_shape'] if metrics else 'n/a'}",
        "",
        "Android 接入：",
        "  model.tflite -> app/src/main/assets/model_cn.tflite",
        "  chars.txt    -> app/src/main/assets/chars.txt",
        "  输出维度必须等于 chars.txt 行数，否则下标会静默错位。",
    ]
    if args.data == "synthetic":
        lines += [
            "",
            "警告：本模型由字体渲染的合成数据训练，仅用于验证链路。",
            "印刷体与真实手指书写差距显著，不要用它评估真实准确率。",
        ]

    report = "\n".join(lines)
    (out_dir / "training_report.txt").write_text(report, encoding="utf-8")
    print("\n" + report)

    # 真正的自检：三个数字必须自洽
    if metrics:
        expected_output = metrics["output_shape"][-1]
        if expected_output != num_classes:
            raise RuntimeError(
                f"模型输出维度 {expected_output} != 字符数 {num_classes}，"
                f"chars.txt 与模型对不上")
        written = len((out_dir / "chars.txt").read_text(encoding="utf-8").rstrip("\n").split("\n"))
        if written != num_classes:
            raise RuntimeError(f"chars.txt 写了 {written} 行，应为 {num_classes} 行")

    print(f"\n产物已写入 {out_dir}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
