# CLAUDE.md

本文件为 AI 代理在本仓库中处理代码时提供指导。

## 项目概览

这是一个用于手写数字（MNIST）识别的两部分项目：

- `python/` — Keras 训练脚本（`keras_mnist_tflite.py`），负责训练模型并转换为 TensorFlow Lite。
- `app/` — 单模块 Android 应用（Kotlin、ViewBinding），通过 TFLite 识别在 `FingerPaintView` 中绘制的多位数值。画布会被切分成单个字形（数字和小数点），每个字形单独分类，然后从左到右拼接结果。

模型仍然只分类单个字符，多字符能力完全位于 Android 侧。

源码按**功能模块**分了子包：`recognize/`（识别页与整条识别链路）、`practice/`（数学练习页）、
`history/`（学习记录页）、`data/`（Room 持久化）。测试源码镜像同样的结构。
`R` 与 ViewBinding 生成类仍在 `com.github.lany192.mnist` 根包 —— 那是由 `namespace` 决定的，
与源码放在哪个子包无关。

## 工作流

- 训练模型：`cd python && python keras_mnist_tflite.py` → 在当前工作目录生成 `model.tflite` → 复制到 `app/src/main/assets/` 供应用加载。
- Python 依赖：`pip install -r python/requirements.txt`（使用阿里云镜像：`https://mirrors.aliyun.com/pypi/simple/`）。
- Android：使用标准 Gradle（`./gradlew assembleDebug`、`./gradlew test`）。`./gradlew test` 是 JVM 基线测试，覆盖切分、题目生成、三个状态机、学习记录页的行映射（`HistoryRowsTest`）和练习页的列表布局约束（`PracticeLayoutConstraintTest`），必须保持全绿；它覆盖 `PureKotlinBoundaryTest` 白名单中的所有内容。该命令**不会**执行 Room 相关测试（见“持久化”）。数据库测试需要连接设备：`./gradlew connectedAndroidTest`。APK 输出路径：`app/build/outputs/apk/debug/app-debug.apk`。

## 容易踩坑

- 模型是 11 类 CNN（0–9 加小数点），输入形状为 [1, 28, 28, 1]（float，0–1；墨迹 = 1.0，背景 = 0.0）；`KerasTFLite.kt` 包装了扁平的 784 个 float 数组，并期望长度为 11 的输出。`python/keras_mnist_tflite.py` 在与 Android 相同的归一化 28×28 分布上合成小数点样本。Android 侧 `MnistPreprocessor.preprocessRegion` 必须保持 MNIST 风格：bbox 裁剪 → 缩放到 20px → 亮度归一化 → 按质心居中。`FingerPaintView.exportDrawingBitmap` 必须以原生分辨率导出纯白背景，绝不能把视图背景色画进导出图，否则墨迹阈值和宽高比都会被破坏。
- 多字符支持由三个文件分担，并且存在硬边界：
  - `InkSegmenter.kt` — 纯 Kotlin，**文件中任何位置都不得导入 `android.*`**。连通域切分返回 `InkGlyph`：只有同时满足“小、实心、靠近基线、位于两个数字组之间”的小数点候选才会被保留，其他内容仍走原有噪声过滤。这是整条链路中唯一可以在 JVM 上做单元测试的部分（`app/src/test/kotlin/.../InkSegmenterTest.kt`），因此所有几何逻辑都放在这里，Bitmap 操作一律排除在外。
  - `MnistPreprocessor.kt` — Android 实现。这里的数值行为必须与历史单数字版本的 `preprocess()` 保持逐字节一致；唯一变化是裁剪框现在由切分器提供，而不是内部自行扫描 bbox。不要“优化”缩放步骤，也不要替换 `Bitmap.createScaledBitmap`——模型就是基于它私有的双线性实现调校的。
  - `MainActivity.kt` — 只负责编排。
- 数学练习（`MathPracticeActivity` + `MathProblem.kt` + `MathProblemGenerator.kt`）建立在识别链路**之上**，不会修改识别链路。`MathProblemGenerator.kt` 与 `InkSegmenter.kt` 遵守同一条硬边界：纯 Kotlin，**文件中任何位置都不得导入 `android.*`**，由 JVM 单元测试覆盖（`MathProblemGeneratorTest.kt`）。在这里加入 Android import 会静默破坏 `./gradlew test`。
- **生成的答案仍然是 1..9999 的整数。** 识别现在已经支持小数点（`Problem.answer` 仍是 `Int`；`68.0` 会按数值判定为 68），但如果要支持小数答案，必须同时修改出题器、跨页 extras、Room schema 和迁移，不能只改其中一部分。4 位上限来自限制 `MAX_DIGITS` 的“笔画宽 / 数字高”比例。整数答案约束由 `MathProblemGeneratorTest` 断言固定，其中包括五六年级的三个数学折中方案。修改任何出题规则前，先阅读这些测试。
- 答案 0 被刻意排除——画布上单独的 “0” 是闭合环，在某些切分路径上很难与“什么都没写”区分。
- `FingerPaintView.inputEnabled` 是冻结画布的唯一方式。`setEnabled(false)` 在这里不起作用：该 View 无条件覆写了 `onTouchEvent`，并且始终返回 true。
- `FingerPaintView.clear()` 必须保持空安全。`drawingBitmap` 在 `onSizeChanged` 中创建，因此在首次布局前清空画布——`MathPracticeActivity.onCreate` 会这样做以展示新题——会命中空 bitmap 并导致应用崩溃。清空尚未布局的画布是合法操作；只是此时还没有 bitmap 需要重建。
- 练习画布周围任何在 `GONE` 和 `VISIBLE` 之间切换的控件都会改变画布尺寸：画布占用 `weight=1`，`onSizeChanged` 会重建 `drawingBitmap`，从而静默擦掉学生刚写的内容——他们会认为“提交吃掉了我的答案”。检查提示使用 `INVISIBLE` 来保留其行高。新增的任何同级控件也要做同样处理，或者预留固定高度。
- **练习页结算区的错题回顾列表（`recyclerReview`）必须留在 `groupResult` 内，并保持 `0dp` + `weight=1`。** 它安全的原因不是“它是个 RecyclerView”，而是整个结算区在作答态是 `GONE`，且 `weight` 给的 `EXACTLY` 规格让 RecyclerView 不测量子项——高度与行数无关。挪出 `groupResult` 或改成 `wrap_content`，它就变成又一个能挤动画布、擦掉笔迹的兄弟控件。`PracticeLayoutConstraintTest` 直接解析布局 XML 把这条钉住；`app/build.gradle.kts` 里已把 `src/main/res/layout` 声明为单测任务的输入，否则改完布局 `./gradlew test` 会直接 UP-TO-DATE 跳过，那条测试就停在旧结论上。
- 判定答案必须按**数值**比较，绝不能按字符串比较——学生写 “068” 表示 68，而且手写中的前导零很常见。当前实现使用 `decimalValueOf(...)` 和 `BigDecimal.compareTo` 做精确比较，因此 `"068"`、`"68.0"` 与 68 等价；不要改回 `Double` 或字符串比较。
- 上一条的推论：**控件的 XML 默认可见性必须与 `render()` 为初始状态生成的可见性一致。** `render()` 首次在 `onCreate` 中运行，之后每次 `onStart`（`repeatOnLifecycle`）都会再次运行；如果默认值不一致，第一次 render 就会翻转兄弟控件的可见性，压缩画布并吃掉学生的笔迹。`textCheckHint` 使用 `invisible` 而不是 `gone`，就是这条规则的实际应用。
- **列表一律 `RecyclerView` + `ListAdapter`/`DiffUtil`**，全项目只有两处：学习记录页整页一个、练习页结算区的错题回顾一个。**记录页的「状态 → 行」映射在纯 Kotlin 的 `HistoryRows.kt`**（`HistoryState.rows` 是**派生属性**——加进主构造参数会把行列表存第二份，并扰动 `StateFlow` 的 conflate）；行里只放原始数据，格式化留在适配器里，因为 `DiffUtil` 的内容比较跑在后台线程，而 `SimpleDateFormat` 不是线程安全的。`HistoryRows.kt` 已加入 `PureKotlinBoundaryTest` 白名单；`HistoryAdapter.kt` / `ReviewAdapter.kt` 含 `androidx.recyclerview`，**绝不能**加入。行间距靠 item 布局根节点的 `layout_marginTop`，不要给 RecyclerView 加 `DividerItemDecoration`。
- **列表适配器在 `init` 里设 `stateRestorationPolicy = PREVENT_WHEN_EMPTY`**，但**别以为它在承重**：真机实测（旋屏 / 进程被杀后重建，各开关一次）**两种路径下它都不改变结果**。旋屏本来就保得住位置——`HistoryViewModel` 跨配置变更存活，`onCreate` 里的 `render(viewModel.state.value)` 在状态恢复之前已提交过一次行列表；进程被杀重建则两种情况下都回到顶部。保留它是为了 render 时机一旦变化时滚动位置不会无声地丢。
- `MathPracticeActivity` 被刻意移除了 `configChanges`：现在状态由 ViewModel 持有，所以旋转、多窗口、折叠屏和字体缩放都会重建 Activity。题目、进度、年级、作答记录和当前阶段会保留；画布笔迹和 TFLite 解释器不会保留（前者因为 `onSizeChanged` 重建 bitmap，后者因为 `KerasTFLite` 会从 assets 重新复制模型）。注意，`configChanges` 本来也从未保护画布——它只保护状态机，而状态机现在由 ViewModel 拥有。
- 年级 `Spinner` 是所选年级的**第二个真值来源**（它会恢复自己的选择）。`render()` 只在选择与 `state.grade` 不一致时写入它，而 `MathPracticeViewModel` 对已经是当前值的 `SelectGrade` 不做处理——这道闸门阻止了 render→dispatch→render 循环，同时也吸收了 Spinner 初始 `position = 0` 的回调。不要移除它。
- 墨迹/背景阈值（32）**只**存在于 `SegmentConfig.inkThreshold`。`MnistPreprocessor` 完全不做阈值处理——它只裁剪切分器交给它的框。如果任何地方再出现一个阈值，裁剪框和归一化对“什么算墨迹”的认知就会不一致。
- 对于画布上的单个连通段，颈部切分守卫会退化为纯粹的宽高比测试（`W < splitAspectFactor * H`，默认 1.35），所以正常的 0–9 手写永远不会被切开。这是比例属性，不是结构保证——降低 `splitAspectFactor`，或者把数字写得足够扁以至于超过 1.35，仍然会触发切分。`InkSegmenterTest.singleDigit_*` 固定了这一点。
- `FingerPaintView.strokeWidth` 不是**外观选择**——实测准确率主要由 `strokeWidth / 手写数字高度` 的比例决定，而不是由切分算法或模型决定：安全区间是 ≤0.19，0.20–0.25 开始退化，≥0.30 会崩塌。当前的 32f 对大约 170–800px 高的数字能保持该比例处于安全区间。把它改回接近 64f 会让 4 位以上数字串不可用（实测 4 位数整串准确率会从 87% 降到 7%）。修改前必须重新测量。
- **搬动源码包时要同时改三处「不在 Kotlin 里」的引用**，它们都不会给出编译错误：
  1. `AndroidManifest.xml` 里的 activity 是**相对类名**（`.recognize.MainActivity`），写错的后果是点入口时
     `ActivityNotFoundException`，编译期一声不吭；
  2. 布局里的自定义 View 必须写**全限定名**（`com.github.lany192.mnist.recognize.FingerPaintView`），
     写错是运行时 inflate 崩溃 —— `FingerPaintView` 与其它 Kotlin 文件没有任何代码耦合，接线只在这两行 XML 里；
  3. Room 导出 schema 的目录名就是 `@Database` 类的全限定名（`app/schemas/<全限定名>/`，**已被 git 跟踪**）。
     改了包名却不 `git mv` 这个目录，Room **不报错**，只在新路径再写一份，旧目录烂在仓库里。
     已确认 `identityHash` 由 schema 内容决定、与包名无关，所以搬包**不会**触发虚假迁移。
- **子包里的文件引用 `R` 必须显式 `import com.github.lany192.mnist.R`。** Kotlin 不自动导入 `namespace` 包，
  而 `import com.github.lany192.mnist.databinding.XxxBinding` 本来就是全限定路径、不受搬包影响 —— 所以漏的通常只有 `R`。
- `settings.gradle.kts` 中的 Maven 仓库使用阿里云镜像——不要移除；这是中国大陆网络访问所必需的。
- Gradle daemon JVM toolchain 固定在 `gradle/gradle-daemon-jvm.properties` 中的 17 版本（本地已安装；缺失时会自动 provision）。
- venv 的 pip 可能在部分升级后损坏；使用 `https://bootstrap.pypa.io/pip/3.9/get-pip.py` 修复（该 venv 使用 Python 3.9）。

## MVI 约定

每个页面（`MainActivity`、`MathPracticeActivity`、`HistoryActivity`）都采用 MVI：每个页面都有一个 `XxxContract.kt`（State / Intent / Effect）和一个 `XxxViewModel.kt`。Activity 只做三件事——把输入转换成 Intent、订阅 State 并渲染、执行一次性 Effect。

- **`*Contract.kt` 和 `*ViewModel.kt` 不得 `import android.`**（`androidx.*` 可以——`androidx.lifecycle.ViewModel` 是普通 JVM 类）。该约束由 `PureKotlinBoundaryTest` 强制执行。一旦 ViewModel 接触 `Bitmap` 或 `Context`，它就会完全脱离 JVM 单元测试——本项目没有 Robolectric。
白名单按**相对路径**索引（源码分了子包），并且每个源文件都必须被明确归类为纯 Kotlin 或 Android 依赖 ——
新增文件忘了分类会直接让 `./gradlew test` 失败，而不是静默漏检。
- **状态机 ViewModel 有意保持同步。** `MutableStateFlow.value =` 和 `Channel.trySend` 从不会挂起，所以 `dispatch()` 返回时状态已经就位，测试可以直接断言 `state.value`——不需要 `kotlinx-coroutines-test`，也不需要 `TestDispatcher`。更硬的理由是：JVM 测试中不存在 `Dispatchers.Main`（`coroutines-android` 是 Android 产物），所以一旦出现 `viewModelScope.launch`，测试会以 `Module with the Main dispatcher had failed to initialize` 失败，而不是编译失败。

  异步操作的分界线是**“这个操作的结果能否被重新推导出来？”**：
  - **写入不能。** 把写入交给注入的端口，并在 `dispatch()` 返回路径上同步交付——`MathPracticeViewModel` 就是通过 `PracticeRecorder` 这么做的；其实现立即返回，并在应用级 `CoroutineScope` 上执行 I/O。**绝不要让 Activity 从 `lifecycleScope` 写入数据库**：旋转会取消协程，触发写入的 Effect 已经被消费且不会重放——结果就是结算页正常渲染、什么都没写入，也没有任何错误提示。
  - **读取可以**（Room 的 `Flow` 在每次订阅时都会重新发射）。View 负责收集并通过 `dispatch(Loaded(...))` 推入；因此 `HistoryViewModel` 保持无参、纯 reducer 的 ViewModel。
- **所有状态转移都发生在唯一的 `private fun reduce(intent)` 中。** 辅助方法只能返回 `Transition`，不得直接触碰 `_state.value`。
- **Effect 只承载一次性事件。任何新建 Activity 必须能够重现的内容都属于 State。** 画布冻结就是典型例子：它是 `phase is Answering` 的纯函数，所以由 `render()` 推导出来。如果把它做成 Effect，旋转后新 Activity 的画布会变成可写，而状态仍然显示“确认中”——学生会以为画布已锁定，随意涂写，然后点击“确认”，结果却被判定为之前的数字。
- **`render()` 必须幂等**（每次 `onStart` 都会重放），并且**绝不能调用 `dispatch()`**。
- 绝不要把 `Bitmap` / `IntArray` / 任何 identity-equals 对象放进 State——`StateFlow` 会按 `equals` 做 conflate，从而导致漏渲染或虚假渲染。
- 识别刻意留在 Activity 中：`MnistRecognizer` 接收 `Bitmap`。Activity 只把画布转换成纯数据 `DigitInput`（`CanvasEmpty` / `NotRecognized` / `Digits`；`Digits` 现在还携带 `decimalIndexes`）；如何处理空画布、无法识别的笔迹或数字过多，属于 ViewModel 策略——这些是流程中唯一的分支，必须能够在 JVM 上测试。
- ViewModel 通过给每个主构造参数提供默认值来获得无参构造。**删除一个默认值后，`by viewModels()` 会在运行时抛出 `Cannot create an instance of class`，而编译期毫无警告。**

## 持久化（Room）

练习历史保存在本地 SQLite 数据库（`practice.db`，两张表）中。不会上传；卸载应用会清除数据。

- **`PracticeRepository.kt` / `PracticeRecorder` 是唯一的持久化边界**——纯 Kotlin，位于边界测试白名单中。如果要移除 Room（KSP 失效，或改用其他存储），只需修改 `RoomPracticeRepository.kt`。
- **Room 3.x 换了 Maven 坐标，也换了包名，两处都得改**：坐标为 `androidx.room3:room3-runtime` / `room3-compiler`，源码里的 `androidx.room.*` 全部变成 `androidx.room3.*`（只改 `libs.versions.toml` 会编译不过，这至少是响的）。KSP 参数**没有**跟着改，仍是 `room.schemaLocation`，`app/schemas/` 的目录布局也没变。Room 3 删掉了 SupportSQLite，`databaseBuilder(...).build()` 在未显式 `setDriver` 时会默认 new 一个 `AndroidSQLiteDriver()`（`androidx.sqlite:sqlite-framework` 由 `room3-runtime` 传递带入），底层仍是同一个 `practice.db`，DB 文件格式没有变化。
- **从 Room 2 升到 Room 3 不会让已装机上的数据库报 integrity 失败。** 同一个 schema 下 Room 3 生成的 `identityHash` 与 Room 2 逐字节相同（已实测：`PracticeDatabase_Impl` 与 `1.json` 都是 `01d2113913db3da5098c7c59b5bea53c`），`room_master_table` 里存的值没有变。
- **删掉 `app/schemas/` 下的文件后，增量构建不会把它写回来。** KSP 任务判定 UP-TO-DATE，而导出目录在 `build/` 之外、不是它的 tracked output——`./gradlew assembleDebug` 会成功退出，仓库里却少了一份 schema，下一版写迁移的人拿到的是残缺基线。要恢复必须先 `clean`；改动实体之后也请确认 schema 文件确实被重写了（它现在以 `\n` 写出，见 `.gitattributes`）。
- **`answer_record.correct` 由 ViewModel 写入，绝不能通过 SQL 的 `CAST(written AS INTEGER)` 重新计算。** 判定必须按数值比较（“068”就是 68），否则第二套判定实现迟早会与主实现漂移。
- **错题本的规则是“每个 `(expression, correct_answer)` 的最新记录如果答错，就是错题”。** `MAX(id)` 子查询**不得**带 `WHERE correct = 0` 过滤：如果带上它，一道题曾经答错、后来答对之后仍会返回旧的错误记录，从而**永远无法离开错题本**。`HistorySummaryTest.mistakes_problemCorrectedLater_disappears` 固定了这一点。
- **按 `(expression, correct_answer)` 分组，不能只按 `expression` 分组。** 题干文本会跨年级复用，把文本当作身份标识会让出题器的输出悄然变成主键。
- **按自增 `id` 排序，绝不能按 `created_at` 排序**——同一组练习的所有行共享同一个时间戳，基于时间的排序不稳定。`created_at` 只用于展示。
- **枚举列存储 `name`，绝不能存储 `ordinal`**——重排 `Grade` 会静默破坏已有数据。
- **重做记录与其他练习一样写入数据库**（这正是错题本能够自愈的原因）**但通过 `practice_session.source = 'REVIEW'` 排除在正确率统计之外**。刚看过答案的题不能计入分数。
- **一组练习完成时在一个事务中归档；中途退出不会写入任何内容。** 已知取舍：完成 10 题中的 9 题后退出会丢失这些记录。如果将来需要修复，应在退出时增加 `Abandon` intent——而不是改成逐题写入。（逐题写入需要预先创建 session 行，还会让半途而废的练习污染正确率分母。）
- **`practice_session` 刻意不存 `total` / `correct`**——它们是 `answer_record` 的聚合结果，存第二份必然会漂移。历史列表通过 `GROUP BY session_id` 现算。
- minSdk 24 限制了 SQLite：**不能使用窗口函数**（3.25 / API 30+），也**不能使用 `RETURNING`**（3.35 / API 31+）。时间戳用 `SimpleDateFormat` 格式化，不要用 `java.time`（API 26+）。
- **在重做模式下，年级 `Spinner` 必须是 `INVISIBLE` + `isEnabled = false`，绝不能是 `GONE`**——那一行是 `wrap_content`，缩小它会让 `weight=1` 的画布变大，从而重建 `drawingBitmap` 并擦掉学生的手写内容。（与检查提示相同的失败模式。）
- **重做时忽略 `SelectGrade`。** 此时 Spinner 已禁用且不可见，唯一可能到达的回调是系统初始的 `position = 0`；如果放行它，重做模式会被自身初始化拆掉——真机观察到表现为“重做页显示了一组全新的年级题”，而且那组题还会被当作普通练习归档，拉低正确率。
- **`Problem` 不能变成 `Parcelable`**：`android.os.Parcel` 属于 `android.*`，这会让 `MathProblem.kt` 掉出白名单，并连带让 `MathProblemGeneratorTest` 失去 JVM 可测性。跨页面传题使用两个平行数组，以及纯函数 `problemsOf`（长度不匹配时返回 null——Activity 会直接结束，而不是让 `problems[index]` 越界）。
- **读取通过推入，写入通过注入**——参见 MVI 一节。`HistoryActivity` 合并三个 Flow 并 dispatch `Loaded`；`MathPracticeActivity` 通过 `MathPracticeViewModel.factory(...)` 接收 `PracticeRecorder`。
- **数据库测试没有 JVM 路径**（`room3-runtime` 是 KMP publication；Android module 的单元测试总会解析到 `-android` variant）。把策略留在纯 Kotlin 中（`HistorySummary`），DAO 中只保留显然正确的查询；这样 `./gradlew test` 能覆盖错题本规则，而仪器测试只验证数据库能打开并正确映射。

## 约定

- 提交信息：使用中文编写 Conventional Commits（例如 `build(gradle): 升级 Gradle 和依赖配置以支持新版本`）。
