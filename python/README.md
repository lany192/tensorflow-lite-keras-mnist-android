本目录有两个训练脚本，产物都交给 Android 应用离线加载。

- `keras_mnist_tflite.py` —— 手写数字（0~9 + 小数点，11 类，28×28）
- `chinese_character_recognition.py` —— 中文简体汉字（GB2312 一级字库，3755 类，64×64）

环境准备：

```bash
pip install -r requirements.txt -i https://mirrors.aliyun.com/pypi/simple/
python -m pip install --upgrade pip -i https://mirrors.aliyun.com/pypi/simple/
```

> `pip freeze > requirements.txt` 的编码取决于当前 shell，历史上这个文件曾被写成
> UTF-16，导致 pip 无法解析。重新生成后请用 `file requirements.txt` 确认是 ASCII/UTF-8。

---

## 数字识别（keras_mnist_tflite.py）

训练脚本会在 MNIST 的 0~9 之外合成小数点样本，生成输出维度为 11 的 `model.tflite`，
并分别打印混合测试集准确率、小数点召回率，以及数字被误判成小数点的数量。

小数点样本的形态由 Android 侧的两段处理共同决定，生成器复现了这两层：

1. `InkSegmenter` 判定"点候选"的几何门槛——紧包围盒宽高比 0.65~1.55、填充率 ≥ 0.5。
   不满足的形状根本到不了模型，所以生成时按同一组阈值拒绝采样。
2. `MnistPreprocessor` 的归一化——包围盒最长边缩放到 20px 再质心居中，因此点到达
   模型时长轴恒为 20px，形态只由固有长短轴比、朝向和边缘软硬决定。

关键后果是**斜向拖尾**：它的包围盒接近方形能通过第 1 步，归一化后的固有长短轴比却
可以到 2.5（斜向的椭圆则因四个角太空、填充率不足而被挡掉）。改这里的生成参数前，
先确认 Android 侧这两段行为没有变，否则模型会重新退化到"只认正圆"。

训练并同步到 Android assets：

```bash
python keras_mnist_tflite.py
cp model.tflite ../app/src/main/assets/model.tflite
```

---

## 中文简体汉字识别（chinese_character_recognition.py）

### 数据从哪来

| 模式 | 命令 | 说明 |
|---|---|---|
| 合成 | `--data synthetic` | 用系统字体渲染 + 强形变增强。**零数据集依赖，但只用于验证链路** |
| 真实 | `--data hwdb --hwdb-dir ~/HWDB1.1` | CASIA-HWDB1.1，3755 类真实手写 |

> **合成数据训出的模型不能用于真实手写。** 字体渲染的印刷体与真人手指书写差距显著，
> 它唯一的用途是让「训练 → 导出 → Android 加载」这条链路跑通、验证预处理与产物正确。
> 要真实效果必须走 hwdb。脚本在合成模式下会把这条警告写进 `training_report.txt`。

CASIA-HWDB1.1 从 <http://www.nlpr.ia.ac.cn/databases/handwriting/Download.html> 获取
（学术免费，2020 年起无需申请表，约 1.2GB），解压后应是
`HWDB1.1trn_gnt/` 与 `HWDB1.1tst_gnt/` 两层结构。

`.gnt` 是私有二进制格式，而公开资料对头部布局的说法互相矛盾（样本长度字段有 uint16 /
uint32 两说，tagcode 有大小端两说）。脚本**不赌任何一种**，而是枚举候选布局、用
「顺序解析恰好走到文件末尾」这条不变量自动判别，并把每个候选的成败打印出来。
拿到数据后先单独验证：

```bash
python chinese_character_recognition.py --inspect-gnt ~/HWDB1.1/HWDB1.1trn_gnt/001-a.gnt
```

它会把采用哪种布局、前若干样本的字符与灰度范围列出来，并在 `result/` 下写 PNG 预览。
**先打开预览确认字形可辨认，再开始训练。**

### 预处理契约

`preprocess_gray()` 是 Python 与 Android 的唯一契约，六步：取墨 → 紧包围盒 → 等比缩放
（最长边 → 内容区 56px）→ **包围盒居中** → 除以本字最大 ink → 输出墨迹=1.0 / 背景=0.0。

两处与数字链路**刻意不同**，不要为了"统一"而改回去：

1. **单字，不切分。** 数字链路靠 `InkSegmenter` 切成一串字形；汉字不能复用那套几何 ——
   `splitAspectFactor=1.35` 会把「一/二/三/川」切开，`dropShortGroups` 会把「三」的三横
   当噪点整组删掉，聚组只看 X 交叠会把左右结构的字（氵+青、亻+尔）拆成两半。
   所以汉字直接取整画布的紧包围盒。
2. **包围盒居中，而不是质心居中。** 质心居中会把左右结构的字（如「们」）整体拉偏 ——
   氵比「尔」重，质心明显偏左。

唯一沿用数字链路的是 `INK_THRESHOLD = 32`（对应 `SegmentConfig.inkThreshold`），
且它**只**用来确定包围盒边界，预处理内部不再做第二次阈值。

改预处理后跑一次对照图，肉眼确认字形居中、不贴边、没被裁角：

```bash
python chinese_character_recognition.py --preview
# -> result/preview_synthetic.png，上排是渲染图，下排是模型实际看到的输入
```

### 训练

本机（Intel i5-10500，纯 CPU）**跑不动全量**：100 万样本单 epoch 要数小时。
所以规模全可配，本机用小配置验链路，正式训练放到 GPU 机器上用同一份脚本放大参数。

```bash
# 本机冒烟
python chinese_character_recognition.py --data synthetic \
    --chars 200 --samples-per-class 24 --epochs 3
# 正式训练
python chinese_character_recognition.py --data hwdb --hwdb-dir ~/HWDB1.1 \
    --chars 3755 --samples-per-class 200 --epochs 40
```

`--samples-per-class 0` 表示用该模式的默认值（合成 40 / HWDB 200）。

产物写入 `--out-dir`（默认 `result/`，已在 `.gitignore` 中），**不会覆盖数字模型**：

| 文件 | 说明 |
|---|---|
| `model.tflite` | 动态范围量化 |
| `chars.txt` | 每行一个字符，UTF-8，**行号即类别下标** |
| `training_report.txt` | 配置、参数量、top-1 / top-5、张量形状 |

报告里的指标是用 TFLite 解释器在**转换后**的模型上复算的，不是 Keras 的 `val_accuracy` ——
量化会掉点，用 Keras 指标汇报会与 Android 侧的实际体验不符。脚本在写出前会强制校验
「模型输出维度 == 字符数 == chars.txt 行数」，三者不一致直接报错，因为那会让 Android
侧的 argmax 下标静默错位成别的字。

### 接入 Android

本次只交付 Python 侧，Android 侧改动按下面清单实施。

**1. assets**

```bash
cp result/model.tflite ../app/src/main/assets/model_cn.tflite
cp result/chars.txt ../app/src/main/assets/chars.txt
```

带 `_cn` 后缀，不覆盖数字模型。构建配置不用改：`KerasTFLite.loadModelFile` 走
`assetManager.open()` 流式拷贝到 `filesDir`，`AssetManager` 会自动解压，几 MB 的模型
同样适用（会因压缩而崩的是 `openFd`/mmap 那条路径，这里没走）。

**2. 要新建的类**

- `recognize/ChineseTFLite.kt`（Android 依赖）—— 仿 `KerasTFLite`，加 `chars.txt` 加载做
  下标→字符映射。**新建而不是改造 `KerasTFLite`**：后者的 `MODEL_SIZE` / 类别数 / 小数点
  类别全是 private 硬编码，`run()` 直接把下标当数字字符用，泛化它会扰动已在跑的数字链路。
  输入构造建议从嵌套 `Array` 换成 `ByteBuffer.allocateDirect`（64×64 的嵌套 Array 会产生
  4096 个装箱 `Float`）。
- `recognize/ChinesePreprocessor.kt`（Android 依赖）—— 复刻 `preprocess_gray` 六步。
  **不要**复用 `MnistPreprocessor`，它的 20px/28px + 质心居中与汉字契约不同。
- 单字取框（纯 Kotlin）—— 在 `InkSegmenter.kt` 加一个公开入口返回整画布墨迹紧包围盒，
  绕开颈部切分/小数点候选/`dropShortGroups` 等全部数字专用守卫。保持该文件零
  `android.*` import，并在 `InkSegmenterTest.kt` 补 JVM 测试（必含「一」「三」这类会被
  现有守卫误杀的用例）。
- 新的识别页（建议独立成页，而非改造识别页）—— `MainContract` / `MainViewModel` /
  `DigitInput` 整条链路是 `List<Int>` + `BigDecimal` 数值语义，改造会污染数字链路并牵连
  练习页的整数答案约束。新页面按项目 MVI 约定配 Contract + ViewModel，两者**不得**
  `import android.`。

**3. 必须同步登记 `PureKotlinBoundaryTest`**

该测试的 `everySourceFile_isClassified` 断言 `src/main/kotlin/.../mnist` 下每个 `.kt` 恰好
落在 `PURE_KOTLIN_FILES` 或 `ANDROID_DEPENDENT_FILES` 之一，**双向**校验：漏登记新文件、
或清单里留着已删除的文件，都会让 `./gradlew test` 直接失败。按**相对路径**索引
（`recognize/Xxx.kt` 形式），且测试的工作目录是 `app/`。凡是几何/映射/状态机逻辑都该进
纯 Kotlin 那一侧（这样才有 JVM 覆盖），只有碰 `Bitmap`/`Context`/`Interpreter` 的才进另一侧。

**4. 两个大模型特有的运行期风险**

现在的模型只有 0.4MB，所以下面两条至今没暴露，模型涨到几 MB 后才会：

- `MainActivity.kt` 在**主线程**构造 `KerasTFLite`，即主线程做「assets 解压 + 拷到
  filesDir + `Interpreter` 构造」。大模型下需要挪到后台并加加载态，否则有 ANR 风险。
- `MainActivity` 与 `MathPracticeActivity` **各自持有一个 Interpreter**，大模型下是两份
  内存，建议改为共享单例。

**5. APK 体积**

当前 debug APK 约 22.4 MiB，其中 4 个 ABI 的 `libtensorflowlite_jni.so` 合计约 15.2 MB，
模型只有 0.4 MB —— **膨胀主因是 native lib，不是模型**。要控体积应做 ABI 过滤/splits
（x86/x86_64 合计 9.3 MB，只有模拟器需要），而不是折腾模型压缩。另外建议显式加
`androidResources { noCompress += "tflite" }`：这不是正确性需要，而是避免每次冷启动
白付一遍解压开销。

**6. 范围**

只支持**单字**（一个画布一个字）。多字识别需要行切分或 CRNN+CTC 序列模型，Android 侧
「逐字形」结构要整个换掉，不在本次范围内。
