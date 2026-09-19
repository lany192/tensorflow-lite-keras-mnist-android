# DEMAND.md — 项目需求与验收基线

> **文档定位**：本文根据当前实现反推并固化产品需求，是后续功能开发、重构和缺陷修复的第一依据。
> 当前基线日期：2026-09-19。对应的主要事实为：Android `versionCode 1` / `versionName 1.0`、
> Room schema `v1`、11 类（`0~9` + 小数点）单字形 TFLite 模型。
>
> **变更规则**：
> 1. 本文中的“必须”是兼容契约或明确验收条件；改变既有“必须”行为时，必须同时修改实现、测试和本文。
> 2. 新增功能应先补充或调整本文中的需求与验收口径，再进入实现。
> 3. `AGENTS.md` 中的安全、架构和工具链约束继续有效；若实现与本文冲突，应先确认哪一边是目标状态，
>    不允许代码、测试和需求长期三套说法。
> 4. 本文描述的是“当前产品基线”，不是未来路线图；未写入的需求默认不属于当前承诺。

## 目录

- [1. 产品定义](#1-产品定义)
- [2. 通用业务规则](#2-通用业务规则)
- [3. 手写数值识别需求](#3-手写数值识别需求)
- [4. 图像预处理兼容契约](#4-图像预处理兼容契约)
- [5. 数学练习需求](#5-数学练习需求)
- [6. 学习记录与错题本需求](#6-学习记录与错题本需求)
- [7. 持久化与数据模式](#7-持久化与数据模式)
- [8. 架构与状态管理需求](#8-架构与状态管理需求)
- [9. 界面与可访问性质量要求](#9-界面与可访问性质量要求)
- [10. 测试、构建与发布基线](#10-测试构建与发布基线)
- [11. 变更影响清单](#11-变更影响清单)
- [12. 验收追踪表](#12-验收追踪表)

---

## 1. 产品定义

### 1.1 产品目标

本项目是一个面向小学生手写场景的离线 Android 应用，包含三条主能力：

1. **手写数值识别**：在画布上书写数字和小数点，应用给出识别结果及逐字形拆分。
2. **按年级数学练习**：系统出题，学生在画布上手写答案，应用识别后判分并结算。
3. **学习记录与错题本**：保存完整练习结果，展示累计正确率、历史练习和当前错题；支持错题重做。

核心价值是：用手写输入代替键盘输入，把识别、练习、判分和订正串成一个闭环。

### 1.2 目标用户与场景

- 目标用户：小学一至六年级学生；监护人或教师可查看学习记录。
- 典型场景：离线环境中完成口算、小数、分数和百分数练习。
- 交互要求：界面以中文文案为主，操作路径简单，主要动作不超过“写—提交—确认/重写—下一题”。
- 识别认知：手写模型不可能保证每题 100% 正确，因此练习流程必须提供“识别结果核对”步骤，避免直接
  把识别错误当成学生答错。

### 1.3 当前范围

- Android 单模块应用，功能包按 `recognize/`、`practice/`、`history/`、`data/` 划分。
- Python 训练链路负责生成 `model.tflite`，当前 App 实际加载的是 MNIST 数字 + 小数点模型。
- 数据仅保存在本地 SQLite 数据库中，不提供登录、账号、云同步、跨设备同步或网络接口。
- 仓库内的中文汉字训练脚本属于独立训练实验，不是当前 Android 产品的端上功能。

### 1.4 明确不做 / 当前不支持

- 不识别整数之外的完整算式，只识别“学生写的答案数值”。
- 不识别负号、分数符号、百分号、汉字或字母；练习中的分数/百分数只出现在题面，答案仍是整数。
- 当前答案约束为整数 `1..9999`，不生成答案 `0`。
- 不支持小数答案落库；`Problem.answer`、`correctAnswer` 和判定基线当前均为 `Int`。
- 不保证画布笔画跨配置变更、进程重建或页面退出后恢复。
- 不做用户自定义题库、教师后台、排行榜、成就系统或在线内容更新。

---

## 2. 通用业务规则

### GEN-01 数值表示

- 识别输入由一个或多个数字以及至多一个小数点组成。
- 合法小数点必须位于两个数字之间，不能出现在开头或末尾。
- 数字位允许前导零，展示和存储必须保留学生实际写出的形式，例如 `068`、`68.0`。
- 数值判定必须按精确数值比较，不能按字符串比较，也不能使用 `Double`；`"068"`、`"68.0"` 与 `68`
  必须等价。
- 单次识别最多保留 4 位数字；小数点不占该上限。超过 4 位时保留前 4 位并明确提示，不能静默丢弃。

### GEN-02 本地与隐私

- 识别、推理、出题、判分和历史统计必须能在无网络情况下完成。
- 应用不提供主动上传练习数据的接口。
- 练习历史保存在应用本地数据库 `practice.db`；卸载应用会清除应用数据。
- Android 系统级备份由平台和设备设置决定；manifest 当前保留 `allowBackup=true`，产品不得把它描述为
  应用自有的云同步能力。

### GEN-03 平台基线

- `minSdk = 24`，`targetSdk = 36`，`compileSdk = 37.1`。
- 应用包名/namespace/applicationId 为 `com.github.lany192.mnist`。
- 所有界面文案放在资源文件中；年级名称顺序必须与 `Grade` 枚举顺序一致。
- 视觉实现遵循 `DESIGN.md` 的 Calm UI 令牌。新增控件不得在布局中临时写死颜色、字号或非标准间距。
- 主要触控目标不得小于 `48dp`。

---

## 3. 手写数值识别需求

### REC-01 画布输入

- 识别页必须提供可手写画布、`清除` 和 `识别` 两个操作。
- 提示必须明确表达：从左到右书写、数字尽量写大并留空隙、小数点在右下角且位于两个数字之间、
  最多 4 位数字。
- 画布导出识别输入时必须使用原始分辨率、纯白背景；不得把视图的显示背景色带入导出位图。
- 导出后必须保留墨迹宽高比，不得为了显示或性能先缩放整张画布。

### REC-02 单字形识别模型

- 模型必须输出 11 类：类别 `0..9` 为数字，类别 `10` 为小数点。
- 输入固定为 `[1, 28, 28, 1]` 的 float 张量，取值范围 `0..1`，墨迹为 `1.0`、背景为 `0.0`。
- Android 包装层接受扁平的 784 个 float；输入为空或长度不为 784 时返回无效结果，不得越界。
- 模型资产位于 `app/src/main/assets/model.tflite`；模型文件必须在每次创建解释器时从 assets 覆盖到
  `filesDir`，不能因旧文件存在而跳过，否则 APK 升级后可能继续加载旧模型。
- 每个页面可以持有独立解释器，但必须在 Activity `onDestroy` 释放；不能在 `onPause` 提前释放。

### REC-03 切分流程

识别链路必须按以下顺序工作：

1. 将画布按亮度转换为墨迹灰度，白底为 `0`、黑笔为最大墨迹值。
2. 使用 8 连通域分离笔画；不能使用 4 连通，否则抗锯齿斜笔画会被打碎。
3. 按水平交叠把属于同一数字的多笔合并；交叠判据只关注 X 方向，以支持断开的数字笔画。
4. 过滤孤立小噪点、过短组和误触实心圆点。
5. 识别小数点位候选，再做守卫式颈部切分处理粘连数字。
6. 最终按包围盒中心 X 从左到右排序，逐字形送入模型。

### REC-04 小数点候选

一个区域只有在同时满足“小而实心、形状接近方形、靠近基线、左右都存在数字”时，才允许保留为
小数点候选：

- 宽和高不得超过参考字高的 `0.38`。
- 包围盒宽高比必须在 `0.65..1.55`。
- 紧包围盒内墨迹填充率不得低于 `0.50`。
- 内部孔洞面积不得达到噪点面积阈值，以免把写得较小但带孔的数字误删。
- 中心必须位于数字基线附近的允许带内。
- 与左右数字的水平空隙不得超过参考字高的 `0.75`。

几何位置只是“候选”条件，最终类别必须由 11 类模型确认。位置不成立的候选应回到普通数字过滤链路，
不能被直接丢弃。

### REC-05 小数点结果规则

- 点必须出现在两个数字之间，最终结果最多包含一个小数点。
- 小数点不增加数字位数上限；4 位整数加一个小数点仍属于合法形式。
- 识别结果返回数字列表和小数点索引，展示时保留实际写法。
- 任何非法形式（多个点、点在开头、点在末尾）都必须被拒绝，并提示学生写在两个数字之间后重写。
- 被拒绝的识别不得覆盖上一次有效结果。

### REC-06 识别页状态与反馈

| 场景 | 行为 |
|---|---|
| 初始状态 | 无结果文本，画布可写 |
| 空画布点击识别 | 提示“请先写一个数字”，保留上一有效结果 |
| 有笔迹但切不出数字 | 提示“没检测到有效笔画，请写大一点”，保留上一有效结果 |
| 识别超过 4 位数字 | 使用前 4 位作为结果，同时提示位数过多 |
| 小数点非法 | 提示位置错误，保留上一有效结果 |
| 识别成功 | 显示完整数值，并显示包括小数点在内的字形数量和逐字形拆分 |
| 点击清除 | 清空画布并清空当前识别状态 |

### REC-07 识别页导航

- 识别页是应用唯一 launcher 入口。
- 页面底部必须提供“数学练习”和“学习记录”两个入口。
- 进入二级页面后，使用 ActionBar 左上角返回箭头退出；行为与系统返回键一致。

### REC-08 生命周期行为

- 识别结果属于 `MainState`，必须能跨 Activity 配置变更保留。
- 画布像素、绘画路径和 TFLite Interpreter 不属于可序列化 State；旋转或 Activity 重建后不要求恢复笔迹。
- 当前实现不承诺进程被杀后的页面状态恢复；只有已归档的练习数据由数据库恢复。

---

## 4. 图像预处理兼容契约

本节是实现识别效果的基础设施需求。除非完成真机识别回归测量，否则不得“顺手优化”。

### PRE-01 输入缩放

`MnistPreprocessor.preprocessRegion` 对切分器给出的紧包围盒执行：

1. 按包围盒裁剪。
2. 对尺寸过大的字形多级减半，避免一步双线性缩放跳过细笔画。
3. 保持宽高比，将最长边缩放至 `20px`。
4. 按该字形自身最大亮度归一化，不能跨字共享全局最亮值。
5. 先按包围盒居中，再按墨迹质心平移到 `28x28` 画布中心。
6. 输出 784 个 float，墨迹 `1.0`、背景 `0.0`。

必须继续使用 Android `Bitmap.createScaledBitmap` 的实际行为；不得替换为自定义缩放器或“更标准”的
双线性算法。数字与小数点共用同一预处理流程。

### PRE-02 阈值唯一性

墨迹/背景阈值 `32` 只能存在于 `SegmentConfig.inkThreshold`。切分器负责判断哪些像素属于墨迹；
`MnistPreprocessor` 不得再做第二套阈值判断。Python 训练脚本中的对应阈值为 `32 / 255.0`，必须与
Android 端保持一致。

### PRE-03 紧包围盒

传给预处理的 `Box` 必须是该字形的紧墨迹包围盒。不得为了“保险”额外加 padding；padding 会改变
缩放分母和质心位置，破坏与模型训练分布的一致性。

### PRE-04 模型训练分布

`python/keras_mnist_tflite.py` 必须保持以下训练契约：

- 使用 MNIST `0..9` 和合成的第 11 类小数点样本。
- 小数点生成前必须通过与 `InkSegmenter` 相同的点候选形状门槛。
- 合成点必须复现 Android 预处理：紧包围盒、最长边 `20px`、按质心居中。
- 训练集可做平移、旋转、缩放增强；测试集不得做训练增强。
- 转换后必须单独报告整体准确率、小数点召回率，以及数字被误判成小数点的数量。
- 生成的新模型必须复制到 `app/src/main/assets/model.tflite` 才能被 App 使用。

### PRE-05 参数红线

- 笔宽 `32f` 是识别精度参数，不是外观参数。安全区按“笔宽 / 手写数字高度 ≤ 0.19”评估；
  `0.20~0.25` 开始退化，`≥0.30` 会显著失效。
- 不得在未重新测量多位数识别率的情况下把笔宽改回接近 `64f`。
- 不得随意降低颈部切分的宽高比守卫。默认 `W < 1.35 * H` 时，正常 0–9 手写不应被切开；该值是比例
  约束而非绝对结构保证，改动必须由 `InkSegmenterTest.singleDigit_*` 验证。
- 修改切分参数、预处理、模型或笔宽时，至少要同时运行 JVM 切分测试和设备端完整识别测试，并记录
  多位数字场景的准确率变化。

---

## 5. 数学练习需求

### PRA-01 练习初始化与年级

- 从首页进入时，默认年级为一年级，生成 10 道题。
- 年级选择器必须提供一年级到六年级，顺序与 `Grade` 枚举和 `grade_names` 数组完全一致。
- 切换年级会用新年级重新生成一组 10 道题，并清空当前进度、作答记录和画布。
- 选择与当前相同的年级必须是 no-op；`render()` 只在 Spinner 选择与状态不一致时回写，避免
  `render → dispatch → render` 回环。
- 重做错题模式下年级选择器必须不可见且不可操作，但保留最后一次年级值用于归档来源信息。

### PRA-02 出题语义

所有题目必须满足：

- 题面只包含等号左侧表达式；界面统一追加 ` = ?`。
- 题面求值结果必须与 `Problem.answer` 精确一致。
- 答案必须是 `1..9999` 的整数，不能为负数，当前不能为 `0`。
- 减法的数学结果不得为负；除法必须整除。
- 同一组题内应避免重复题面；生成必须在有限重试后终止，不能因题库较小而死循环。
- 题型按固定顺序轮转，不能每题独立随机抽题型造成整组偏科。
- 随机源必须可注入固定 seed 以便测试复现；生产环境使用随机源。

各年级题型要求如下：

| 年级 | 题型 | 关键边界 |
|---|---|---|
| 一年级 | 20 以内加法、20 以内减法 | 两数题面均不含 0；加法和不小于 2；减法差为正 |
| 二年级 | 100 以内加减、表内乘法、表内除法 | 加法两数均 ≥2；减法差 ≥2；乘除因数/除数在 `2..9`；除法必须整除 |
| 三年级 | 三位数加减、两位数乘一位数、两位数除一位数 | 加法两数均为三位且和 ≤999；减法差 ≥1；除法商和被除数边界合法 |
| 四年级 | 万以内加减、三位数乘两位数、三位数除两位数 | 加法和在 1000–9999；减法被减数为 4 位、差为正；乘法答案 ≤9999；除法整除且被除数为三位数 |
| 五年级 | 一位/两位小数加减乘除 | 每条题面都含小数点；只构造数学上精确、答案为整数的组合 |
| 六年级 | 分数加减乘除、百分数 | 题面必须含 `/` 或 `的`；分数必须按约分后展示；答案必须为整数 |

### PRA-03 难度与识别上限一致

- 数学答案最多 4 位，和识别器的 `MAX_RECOGNIZED_DIGITS` 必须一致。
- 生成器、识别状态机、提示文案和测试都不得出现不同的位数上限。
- 四年级必须稳定覆盖 4 位答案，以保留真实的高难度识别场景。

### PRA-04 单题状态机

练习题必须且只能处于以下四种阶段之一：

1. `Answering`：等待书写和提交。
2. `Confirming`：已经识别出结果，等待学生确认或重写。
3. `Judged`：答案已判错，等待学生查看正确答案后进入下一题。
4. `Finished`：整组题已完成并进入结算。

答对时直接从 `Confirming` 进入下一题；只有答错时才停留在 `Judged`。最后一题答对时直接进入
`Finished`，最后一题答错时必须在点击“下一题”后进入结算。

### PRA-05 提交与异常输入

| 输入场景 | 要求 |
|---|---|
| 空画布提交 | 保持 `Answering`，提示先写答案，不清空已有笔迹 |
| 有笔迹但识别不出数字 | 保持 `Answering`，提示写大一点，不清空已有笔迹 |
| 小数点位置非法 | 保持 `Answering`，提示重新书写合法小数 |
| 超过 4 位数字 | 截断到前 4 位后进入 `Confirming`，同时提示只识别前 4 位 |
| 合法识别结果 | 进入 `Confirming`，画布冻结并展示识别数字及逐字形拆分 |

### PRA-06 两阶段确认

- `Confirming` 必须同时展示“识别结果”和逐字形拆分，让学生能发现是哪一位识别错误。
- 此阶段画布必须冻结，学生不能继续写入影响待确认结果。
- “重写”必须回到 `Answering` 并清空画布。
- “确认”才对当前识别值进行判定；确认后不能再次重复判定同一道题。
- 画面冻结必须由阶段状态推导，不能做成一次性 Effect；旋转后阶段仍为 `Confirming` 时，新画布也必须保持
  不可写。

### PRA-07 判定规则

- 判定使用精确十进制数值比较；必须支持前导零和末尾小数零。
- `Confirm` 只在 `Confirming` 状态生效；其他状态重复点击必须被忽略。
- `Next` 只在 `Judged` 状态生效；其他状态重复点击必须被忽略。
- 每次确认必须生成一条作答记录，记录题号、题目、学生写出的数字与小数点位置、判定结果。
- 答错的判定态必须显示正确答案和学生写出的答案。

### PRA-08 操作按钮与阶段

- `Answering`：显示“清除”“提交”，画布可写。
- `Confirming`：显示“重写”“确认”，画布冻结。
- `Judged`：显示“下一题”。
- `Finished`：隐藏题目与作答区，显示结算区和“再来一组”。
- 按钮行高度必须恒定，不能因某个按钮显隐改变画布高度。

### PRA-09 结算

- 结算显示答对数和总题数。
- 普通练习全部答对时可显示鼓励文案和正确色；有错题时使用中性摘要。
- 错题重做结算只复述本次正确数/题数，不使用“太棒了”等庆祝文案。
- 有错题时显示“错题再练一遍”按钮；无错题时隐藏该入口。
- 错题回顾列表只包含本次答错的题，每行显示题号、题面、正确答案和学生答案。

### PRA-10 错题重做模式

- 从学习记录进入时，接收最近最多 10 道错题并进入重做模式。
- 从结算页“错题再练一遍”进入时，只重做本次练错的题。
- 重做模式隐藏并禁用年级选择器，任何 `SelectGrade` 回调都必须被忽略，不能退化为按年级新出题。
- 重做完成后“再来一组”不得改回按年级出题，必须重跑同一批重做题。
- 空题目集合必须被状态层拒绝，不能让空列表进入 `problems[index]` 导致越界。
- 重做题目可能跨年级；归档时记录发起重做时保存的年级，但来源必须标记为 `REVIEW`。

### PRA-11 练习归档时机

- 只有整组题完成后才归档，一次练习必须在同一事务中写入 session 和全部 answer 记录。
- 中途退出不写入本组任何记录；这是当前已知取舍。
- 归档必须记录每道题，包括答对的题，不能只记录错题。
- 归档交付必须非阻塞、不能抛异常到页面流程；真正 I/O 运行在应用级作用域，不能挂在 Activity 的
  `lifecycleScope` 上。
- 归档失败只记录日志，不得破坏已经完成的结算页面。
- 重做记录必须写库，否则错题本无法通过“最新记录”自愈；但不得计入累计正确率。

### PRA-12 页面状态与布局硬约束

- `MathPracticeActivity` 不声明 `configChanges`；旋转后必须重建 Activity，但题目、年级、进度、作答记录和
  阶段由 ViewModel 保留。
- 练习画布占据 `weight=1` 的剩余空间；任何改变画布高度的布局变化都会重建 `drawingBitmap` 并擦掉笔迹。
- 练习页所有位于画布周围、会切换可见性的文本行必须使用 `INVISIBLE` 或预留固定高度，不得用 `GONE`。
- 重做模式下年级标签和 Spinner 必须使用 `INVISIBLE + isEnabled=false`，不能使用 `GONE`。
- 检查提示 `textCheckHint` 必须使用 `invisible`，不能改成 `gone`。
- `recyclerReview` 必须始终位于作答态为 `GONE` 的 `groupResult` 内，并保持 `0dp + weight=1`；否则会
  参与画布高度计算。
- 布局 XML 中控件的初始可见性必须与 `render()` 对 `Answering` 的结果逐项一致。
- 作答态用空字符串清空 `textFeedback` / `textDetail`，不得改成 `GONE`，否则结果出现时仍会改变画布高度。
- `FingerPaintView.clear()` 必须为空安全：首次布局前调用不能访问尚不存在的 bitmap。
- 冻结画布只能使用 `inputEnabled=false`；`setEnabled(false)` 对该 View 无效。

---

## 6. 学习记录与错题本需求

### HIS-01 页面结构

- 学习记录页必须使用一个整页 `RecyclerView` 呈现所有内容；不得退回多个可滚动容器或按分区堆叠静态容器。
- 行顺序固定为：累计正确率 → 各年级统计 → “历史练习”标题及记录 → “错题本”标题、超限提示及错题。
- 没有任何统计时只显示空态：“还没有练习记录，做完一组题就会出现在这里”。
- 没有内容的 section 不显示标题，避免出现空分区。
- 错题超过一次重做上限时，在错题条目之前显示“另有 N 道错题未纳入一次重做”。

### HIS-02 累计正确率

- 累计正确率只统计 `source = GRADE` 的作答。
- `REVIEW` 重做记录必须入库，但不得进入正确率分子或分母。
- 统计必须按年级聚合，并按 `Grade` 枚举从低到高排序。
- 总体正确率由各年级正确数、总题数求和后计算。
- 没有记录时不得显示误导性的 `0%`，应显示空态。
- 百分比按整数除法计算，当前不保留小数。

### HIS-03 历史练习列表

- 历史列表同时展示普通练习和错题重做。
- 每行显示时间、年级、答对数和总题数；重做记录必须带“错题重做”标签。
- 时间格式当前为 `MM-dd HH:mm`，使用 `SimpleDateFormat`；因 `minSdk 24` 不使用 `java.time`。
- 列表按 `createdAt` 倒序、相同时按自增 `id` 倒序，最新记录在前。
- 每次练习的 `total` / `correct` 必须从 `answer_record` 实时聚合，不能在 session 表中再存一份。

### HIS-04 错题本定义

错题本条目按 `(expression, correctAnswer)` 唯一分组：

- 每个题目的最新一条作答记录如果答错，则该题当前是错题。
- 如果同一道题后来答对，必须自动从错题本消失。
- 同一道题重复答错只显示一条，并显示累计错误次数。
- 分组不能只看 `expression`，因为不同年级或不同答案可能复用相同题面。
- 记录必须按自增 `id` 判断新旧；不能按 `createdAt` 排序，因为同一组题共享同一时间戳。
- 错题本当前读取最近 1000 条作答记录，最多展示 100 道错题；更老的数据不参与当前展示。

### HIS-05 错题重做入口

- 页面底部“重做错题”按钮在错题本为空时必须不可用，且点击不得发出跳转事件。
- 重做时取错题本中最近的 10 道题，避免一次重做过多。
- 题目通过两个平行数组 `expressions` / `answers` 跨页传递，不能让 `Problem` 依赖 `Parcelable`。
- 两个数组长度不一致时必须终止进入练习页，不能构造残缺题目集。
- 完成后按正常练习流程逐题填答，并按 `PRA-10`、`PRA-11` 处理。

### HIS-06 列表实现约束

- 全项目列表只允许在“学习记录页”和“练习结算错题回顾”两处使用 `RecyclerView`。
- 必须使用 `ListAdapter` / `DiffUtil`；每个行型必须是稳定的 `data class` 或 `data object`，不能使用
  identity equality 的普通类。
- `DiffUtil` 的 item 身份必须先判断行型；错题行身份为 `(expression, correctAnswer)`，练习行身份为 session id。
- 行模型只放原始数据，不得放 `SimpleDateFormat` 结果或依赖 `Context` 的文本。
- 行间距由 item 根布局的 `layout_marginTop` 提供，不使用 `DividerItemDecoration`。
- 适配器在 `init` 中设置 `stateRestorationPolicy = PREVENT_WHEN_EMPTY`；该设置当前不是滚动恢复的唯一保障，
  但保留用于 render 时机变化后的防御。

---

## 7. 持久化与数据模式

### DATA-01 存储边界

- 本地数据库名为 `practice.db`，Room schema 当前版本为 `1`。
- `PracticeRepository` / `PracticeRecorder` 是持久化边界；更换 Room 或存储实现时，业务状态机和界面不应改动。
- 读取使用 Room `Flow`，由 View 收集后通过 `Loaded` intent 推入历史 ViewModel。
- 写入由练习 ViewModel 注入 `PracticeRecorder` 完成；ViewModel 自己不直接依赖 Room 实现。
- 数据库单例只持有 `applicationContext`，不得持有 Activity。

### DATA-02 表结构

`practice_session`：

| 字段 | 含义 | 约束 |
|---|---|---|
| `id` | 自增主键 | `Long` |
| `grade` | 年级 | 存 `Grade.name` |
| `source` | 练习来源 | 存 `GRADE` 或 `REVIEW` |
| `createdAt` | 创建时间 | epoch 毫秒 |

`answer_record`：

| 字段 | 含义 | 约束 |
|---|---|---|
| `id` | 自增主键 | `Long` |
| `sessionId` | 所属练习 | 外键指向 session，删除级联 |
| `expression` | 题面 | 不含 `= ?` |
| `correctAnswer` | 正确答案 | 当前为整数 |
| `written` | 学生写出的文本 | 非空；无识别文本时用空串，不存 NULL |
| `correct` | 判定结果 | 由 ViewModel 原样写入，不能用 SQL 重算 |

- `sessionId` 和 `expression` 必须有索引。
- `practice_session` 不得存 `total` / `correct`；它们必须从 `answer_record` 聚合得到。
- 枚举字段存 `name`，不能存 `ordinal`。
- `correct` 的判定必须保持与 PRA-07 一致；禁止 SQL `CAST(written AS INTEGER)` 形成第二套判定逻辑。

### DATA-03 查询规则

- 累计正确率查询只包含 `source = 'GRADE'`。
- 历史列表包含两种来源，按 session 聚合题数和正确数，按时间倒序。
- 错题查询只允许按 `id DESC` 取原始作答行；“每个题目的最新记录是否答错”的策略放在纯 Kotlin 中，
  不得通过带 `WHERE correct = 0` 的 SQL 子查询实现。
- `minSdk 24` 下不能使用窗口函数或 `RETURNING`；时间格式化不能使用 `java.time`。

### DATA-04 Schema 与迁移

- `PracticeDatabase` 必须保持 `exportSchema = true`，KSP 参数仍是 `room.schemaLocation`。
- schema 快照位于 `app/schemas/com.github.lany192.mnist.data.PracticeDatabase/`，必须纳入 git。
- Room 3 使用 `androidx.room3:*` 坐标和包名；不要改回 Room 2，也不要删除现有 schema。
- 任何数据库字段、索引、外键或语义变化都必须提升 schema 版本、提供迁移、更新 schema 文件并增加迁移验证。
- 修改实体后必须确认 schema 文件确实被重写；如果导出目录缺失，需要先 clean 再构建。
- 搬动 `@Database` 包名时必须同步移动 schema 目录；Room 不会替开发者保住旧路径。

---

## 8. 架构与状态管理需求

### ARCH-01 MVI 页面

- `MainActivity`、`MathPracticeActivity`、`HistoryActivity` 均采用 MVI：每页有 `XxxContract` 和 `XxxViewModel`。
- Activity 只负责：把用户输入翻译成 Intent、订阅 State 并渲染、执行一次性 Effect。
- 所有状态转移必须收敛到唯一的 `private fun reduce(intent)`；辅助方法只能返回 Transition，不能直接写
  `_state.value`。
- `dispatch()` 返回时状态必须已经就位；状态机保持同步，不能为了异步而改成需要 `Dispatchers.Main` 的
  `viewModelScope.launch`。
- Effect 只承载一次性事件。任何 Activity 重建后应能由状态恢复的内容都必须进入 State。
- `render()` 必须幂等，不得调用 `dispatch()`；每次 `onStart` 允许重放当前状态。
- State 中不得放 `Bitmap`、`IntArray` 或其他 identity-equals 对象。

### ARCH-02 纯 Kotlin 边界

以下文件不得出现 `import android.`：

- `recognize/DigitInput.kt`
- `recognize/InkSegmenter.kt`
- `recognize/MainContract.kt`
- `recognize/MainViewModel.kt`
- `practice/MathProblem.kt`
- `practice/MathProblemGenerator.kt`
- `practice/MathPracticeContract.kt`
- `practice/MathPracticeViewModel.kt`
- `history/HistorySummary.kt`
- `history/HistoryContract.kt`
- `history/HistoryRows.kt`
- `history/HistoryViewModel.kt`
- `data/PracticeRepository.kt`
- `data/PracticeEntities.kt`

边界由 `PureKotlinBoundaryTest` 强制。新增 `src/main/kotlin` 文件必须明确归入纯 Kotlin 或 Android 依赖
清单；漏归类应直接让 `./gradlew test` 失败。

### ARCH-03 识别与 ViewModel 分工

- `MnistRecognizer` 依赖 `Bitmap`，识别编排必须留在 Activity。
- Activity 只把导出画布和识别结果翻译成 `DigitInput`：`CanvasEmpty`、`NotRecognized` 或 `Digits`。
- 空画布、无法识别、位数过多和非法小数点的处理策略必须位于纯 Kotlin ViewModel，便于 JVM 测试。
- `Problem` 不得实现 `Parcelable`；跨页传题使用平行数组和 `problemsOf`。

### ARCH-04 写入与读取边界

- 写入不可重推：由 ViewModel 注入 `PracticeRecorder`，在 `dispatch()` 的同步返回路径上交付；真正的 I/O
  交给应用级作用域。
- 读取可重推：Room `Flow` 由 View 收集并通过 `Loaded` 推入 `HistoryViewModel`，因此历史 ViewModel 可以
  无参、无协程、纯 reducer。
- 不得让 Activity 使用 `lifecycleScope` 直接写数据库；结算瞬间旋转可能取消协程而静默丢数据。

### ARCH-05 ViewModel 构造兼容

- `MainViewModel`、`HistoryViewModel` 和 `MathPracticeViewModel` 必须保持可被 `by viewModels()` 或显式工厂
  创建。
- `MathPracticeViewModel` 的每个主构造参数必须有默认值；删除默认值会把错误推迟到运行时。
- `MathPracticeViewModel.factory(recorder)` 必须显式注入 recorder，不能从全局偷偷取实例，以免忘记初始化时
  静默不写数据。

---

## 9. 界面与可访问性质量要求

- 视觉方向为 Calm UI：暖中性背景、白色 soft surface、低饱和强调色、大圆角、去阴影、去边框。
- 颜色、字号、间距、按钮和状态色必须来自 `colors.xml`、`dimens.xml`、`styles.xml`。
- 布局必须显式使用 `MaterialButton`，不要用裸 `Button`。
- 二级页返回入口固定在 ActionBar 左上角，不往内容区增加返回按钮。
- 答对使用 `correct` 语义色，答错使用 `wrong` 语义色；识别待确认结果使用中性色。
- 页面反馈优先在页面内展示；Toast 只用于短暂错误提示，不作为业务状态的唯一载体。
- 文案必须准确描述动作。位数超限提示必须包含实际上限，小数点错误提示必须说明“两个数字之间”。
- 任何新增的列表都必须使用 `RecyclerView + ListAdapter/DiffUtil`，不得引入同类第二套列表方案。

---

## 10. 测试、构建与发布基线

### QA-01 JVM 测试

`./gradlew test` 必须保持全绿，至少覆盖：

- `InkSegmenterTest`：连通域、聚组、噪点过滤、小数点位置、颈部切分和排序。
- `DigitInputTest`：输入形态、前导零、小数格式、非法小数点和 4 位上限一致性。
- `MainViewModelTest`：识别页状态与 Effect。
- `MathProblemGeneratorTest`：六个年级题面求值、答案范围、唯一性、题型覆盖、无负数和整除。
- `MathPracticeViewModelTest` / `MathPracticeViewModelReviewTest`：练习状态机、两阶段确认、结算、重做和归档。
- `HistorySummaryTest`：累计正确率、错题本最新记录规则、跨年级复用题面和重做取题。
- `HistoryRowsTest`：学习记录页状态到列表行的映射。
- `HistoryViewModelTest`：历史页 reducer 和空错题防跳转。
- `PracticeLayoutConstraintTest`：练习页错题列表必须位于隐藏结果组内且保持固定权重高度。
- `PureKotlinBoundaryTest`：纯 Kotlin 边界和源码分类完整性。

### QA-02 设备测试

`./gradlew connectedAndroidTest` 必须在设备/模拟器上覆盖：

- TFLite 能加载 assets 模型并完成空画布、单笔画、多笔数字、分离数字和小数点识别。
- Room 能建库、事务写入、回填 sessionId、正确 JOIN、按时限读取并在内存数据库场景清理。
- 切分算法已有 JVM 覆盖的部分仍以 JVM 测试为主，设备测试只补 Bitmap 和 TFLite 两段。

### QA-03 构建与产物

- Android 构建使用 Gradle Wrapper，daemon JVM toolchain 固定为 JDK 17。
- `./gradlew assembleDebug` 成功，APK 输出为 `app/build/outputs/apk/debug/app-debug.apk`。
- `settings.gradle.kts` 中的阿里云 Maven 镜像为中国大陆构建环境所需，不能移除。
- Python 依赖通过 `python/requirements.txt` 安装；训练输出 `model.tflite` 后必须同步到 assets。

### QA-04 发布前人工检查

- 至少覆盖一至六年级各一组题。
- 验证 1–4 位整数、前导零、带小数点答案、非法小数点、空画布和无法识别输入。
- 验证旋转、多窗口、字体缩放和折叠屏重建后：状态机进度保留、画布笔迹不要求保留、冻结状态仍正确。
- 验证中途退出不写记录，完整完成后历史记录、正确率、错题本和重做入口一致。

---

## 11. 变更影响清单

以下改动最容易产生“无编译错误但功能静默错误”，提交前必须逐项检查：

- **改识别上限**：同步 `MAX_RECOGNIZED_DIGITS`、生题答案上限、UI 提示和测试。
- **支持小数/分数答案**：必须同时修改出题器、`Problem.answer`、Attempt、跨页 extras、Room schema 和迁移；
  只改其中一部分不算完成。
- **改切分或预处理**：重新跑 JVM、设备端识别回归，并记录位数、小数点召回和数字误判为点的变化。
- **改笔宽**：先验证“笔宽 / 数字高度”比例，再验证 1–4 位和跨多位数字的整串准确率。
- **改练习页布局**：检查所有 `GONE/VISIBLE` 切换对画布高度的影响，运行 `PracticeLayoutConstraintTest`。
- **改年级枚举或文案顺序**：同步 `Grade`、`grade_names`、历史数据 name 解析和重做模式拦截。
- **改数据库字段**：提升版本、写迁移、导出 schema、补迁移测试，并检查 SQLite minSdk 24 能力。
- **搬动源码包**：同步 `AndroidManifest.xml` 相对类名、布局中的全限定自定义 View、Room schema 目录路径，
  并给子包文件显式导入 `com.github.lany192.mnist.R`。
- **新增纯 Kotlin 文件**：加入 `PureKotlinBoundaryTest` 白名单并补 JVM 测试。
- **新增列表**：只能使用 `RecyclerView + ListAdapter/DiffUtil`，行模型放原始数据，格式化留在适配器。
- **修改错题规则**：保留 `(expression, correctAnswer)` 分组、按最新记录判错、REVIEW 不计正确率、
  MAX(id) 语义不提前过滤错题等核心行为。

---

## 12. 验收追踪表

| 需求组 | 主要验收来源 |
|---|---|
| REC-01、REC-06、REC-07 | `MainViewModelTest`、`MainActivity`、`activity_main.xml` |
| REC-02、REC-05、PRE-01～PRE-04 | `MnistRecognizerTest`、`KerasTFLite`、`MnistPreprocessor`、Python 训练脚本 |
| REC-03、REC-04 | `InkSegmenterTest`、`InkSegmenter.kt` |
| PRA-01～PRA-09 | `MathProblemGeneratorTest`、`MathPracticeViewModelTest`、`MathPracticeActivity` |
| PRA-10～PRA-12 | `MathPracticeViewModelReviewTest`、`PracticeLayoutConstraintTest`、`activity_math_practice.xml` |
| HIS-01～HIS-06 | `HistorySummaryTest`、`HistoryRowsTest`、`HistoryViewModelTest`、`HistoryActivity`、适配器 |
| DATA-01～DATA-04 | `PracticeEntities.kt`、`PracticeDatabase.kt`、`PracticeDaoTest`、`app/schemas/**` |
| ARCH-01～ARCH-05 | `PureKotlinBoundaryTest`、各 Contract/ViewModel、各 Activity |
| QA-01～QA-04 | `./gradlew test`、`./gradlew connectedAndroidTest`、`./gradlew assembleDebug` |

当实现、测试与本文出现冲突时，处理顺序必须是：确认产品目标 → 更新本文 → 修改测试 → 修改实现，
不得通过删除测试或关闭 schema 导出来让流程变绿。
