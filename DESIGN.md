# 界面设计规范

本文档记录本项目界面**实际使用**的颜色、字号、间距、布局骨架与文案约定，作为新增页面、控件和调整样式时的依据。

> **设计方向：Calm UI（柔和极简）。**
> 克制的暖中性底色、大圆角 soft surface、单一低饱和强调色、去阴影、去边框；
> 视觉层次由「字阶 + 字距 + 表面色差 + 留白」建立，不由卡片和投影建立。
>
> 本文中的**取值全部收敛在 `res/values/colors.xml`、`dimens.xml`、`styles.xml`**，
> 布局里不再出现字面色值与字面字号。文中标注「现状」的是既有事实，标注「约定」的是新增代码必须遵守的规则。

---

## 0. 为什么是 Calm UI

2026 年 Android 端最主流的是 Google 的 Material 3 / M3 Expressive（Google I/O 2025 发布，已是新应用默认目标），
儿童教育类应用的最佳实践则是「圆角 + ≥48dp 触控目标 + 即时反馈 + 语义色」。
本项目的既有哲学（工具型、无品牌色、靠留白分层）与 Calm UI 同源，问题从来不是方向而是**执行**——
死灰底色、无令牌、系统默认按钮、无圆角层次。

因此本次改版的选择是：**保留「克制」哲学，把它从「没做设计」推进到「刻意做减法」**，
并借用 Material 3 的**组件与状态系统**（`MaterialButton` 的涟漪与禁用态）作为实现底座，
**但不采用它的视觉语言**（不用 elevation、不用 container 梯度分层、不用动态取色）。

---

## 1. 技术前提

| 项 | 现状 |
|---|---|
| 界面技术 | XML 布局 + ViewBinding（`buildFeatures.viewBinding = true`），无 Compose |
| 主题 | `Theme.Material3.Light`（`res/values/styles.xml` 的 `AppTheme`），**保留 ActionBar**但改为扁平浅色 |
| 令牌 | `colors.xml`（颜色）、`dimens.xml`（间距与尺寸）、`styles.xml`（5 个 `TextAppearance` + 4 个按钮样式 + ActionBar 样式） |
| 控件 | AppCompat `TextView` / `Spinner` / `ScrollView` + **显式 MaterialButton** |
| 形状 | 3 个 `<shape>` drawable（`bg_canvas` / `bg_block` / `ic_chevron_right`）+ 3 个 ColorStateList（`res/color/`） |
| 自定义字体 | 无。字体族只用 `sans-serif`（默认）与 `sans-serif-medium` |
| 深色模式 | **未适配**：主题是 `Light` 而非 `DayNight`，没有 `values-night` 目录 |
| 列表 | 不用 `RecyclerView`：`activity_history` 用「`LinearLayout` + 运行时 inflate 子项」，数据量是个位到几十条 |
| 页面骨架 | 三个 Activity 都是「根 `LinearLayout`（vertical）+ `background=@color/surface` + `paddingStart/End=page_padding_h`」 |
| 页面标题 | 由 ActionBar 提供（`AndroidManifest.xml` 的 `android:label`），页面内部不重复标题 |
| 二级页返回入口 | ActionBar 左上角的返回箭头（`setDisplayHomeAsUpEnabled(true)`），**页面内部没有返回按钮**（见 1.3） |

### 1.1 为什么保留 ActionBar

改版曾考虑改成 `NoActionBar` + 页面内大标题（Calm UI 的典型做法），**最终放弃**：
本项目**没有任何 `WindowInsets` / `fitsSystemWindows` / 边到边处理代码**，而 `targetSdk = 36`
会让 Android 15+ 强制边到边。ActionBar 同时承担着标题与系统栏 inset 两项职责，去掉它就必须
补一套 inset 处理，否则标题会直接钻到状态栏底下。这里只把它从「`#1E1E1E` 近黑底」改成与页面同色的扁平浅色条。

**约定**：要改 `NoActionBar` 之前，先把三个 Activity 的 inset 处理补齐（`ViewCompat.setOnApplyWindowInsetsListener`
+ `WindowCompat.setDecorFitsSystemWindows`），并同步更新本文档。

### 1.2 为什么父主题是 Material3 但不是 DayNight

- **Material3**：`material` 依赖本来就在 `build.gradle.kts` 里但一直没被使用；`MaterialButton`、
  ColorStateList 状态系统、`ThemeEnforcement` 都要求 Material 主题。用非 Material 主题配 Material 组件会直接抛异常。
- **Light 而非 DayNight**：没有 `values-night` 时，`DayNight` 会让 Material3 自带的深色资源在系统深色模式下生效，
  渲染出一个**只有一半被适配过**的界面。当前 Light-only 是明确现状，不是疏漏。
  要支持深色模式：先建 `values-night/`，逐个核对本文档所有色对的对比度，再改父主题。

**约定**：`Theme.Material3.Light` 只作为**组件底座**使用。不要因为换了主题就去用 `MaterialCardView`、
`elevation`、`colorSurfaceContainer*` 梯度——本文档的颜色角色已把 container 梯度全部压平。

### 1.3 二级页的返回入口放在 ActionBar 左上角

两个二级页（练习页 `MathPracticeActivity`、学习记录页 `HistoryActivity`）的返回入口统一是
**ActionBar 左上角的返回箭头**，由 `supportActionBar?.setDisplayHomeAsUpEnabled(true)` 开启，
点击落在各 Activity 的 `onOptionsItemSelected`（`android.R.id.home`）里，行为与系统返回键一致。

**约定**：

- **不要在页面内容区新增返回按钮。** 学习记录页原先在底部按钮行有一个「返回」文字按钮，已移除；
  同一页出现两个返回入口会让「返回」看起来像两种不同的操作。
- 尤其不要把返回按钮加到**练习页的首行**：那一行已被「年级标签 + Spinner + 进度」占满，
  再塞一个控件会撑高这一行（Spinner 的行高未必等于 `touch_target`），
  挤矮 `weight=1` 的画布 → `onSizeChanged` 重建 `drawingBitmap` → **静默擦掉学生刚写的字迹**（见第 7 节）。
  ActionBar 与页面内容区相互独立，改它不参与画布的高度计算。
- 练习页用返回箭头中途退出**不归档**已完成但未收尾的那组练习，这与系统返回键的既有取舍一致（见 `CLAUDE.md`）。

---

## 2. 颜色

### 2.1 色板

全部定义在 `res/values/colors.xml`，**布局里不得再出现 `#` 开头的字面色值**。

| 令牌 | 值 | 用途 | 对比度 |
|---|---|---|---|
| `surface` | `#F7F6F3` | 页面底色 | — |
| `surface_raised` | `#FFFFFF` | 画布、soft 色块 | — |
| `on_surface` | `#1C1B1A` | 主文字 | 16.6:1 on surface |
| `on_surface_variant` | `#6B6862` | 次要文字 | **5.14:1** on surface |
| `accent` | `#3E5C76` | **唯一强调色**：主按钮、文字按钮 | 6.48:1 on surface |
| `on_accent` | `#FFFFFF` | 强调色上的文字 | — |
| `accent_container` | `#E7ECF2` | 强调色的极浅填充 | — |
| `on_accent_container` | `#2C4359` | 浅填充上的文字 | — |
| `outline` | `#E3E1DC` | 极细分隔。**当前未被使用**，保留给将来的列表分隔 | — |
| `correct` | `#3F6B4F` | 答对，仅用于**反馈文字** | 5.67:1 on surface |
| `wrong` | `#8C4A3F` | 答错，仅用于**反馈文字** | 6.14:1 on surface |
| `on_surface_disabled` | `#A5A29B` | 禁用态文字 | — |
| `accent_disabled_container` | `#E4E2DE` | 禁用态主按钮底色 | — |
| `colorPrimary` / `colorPrimaryDark` / `colorAccent` | → `accent` / `surface` | 主题槽位别名，保留旧名以免主题引用处大面积改名 | — |

### 2.2 约定

- **只有一个彩色：`accent`。** 不要引入第二个强调色，也不要调高它的饱和度。
- 语义色 `correct` / `wrong` **只用于反馈文字**，绝不用作背景色块、图标或边框。刻意去饱和，
  是为了保持 Calm 的低刺激——**不要**换成 `#4CAF50` / `#F44336` 这类 Material 默认语义色。
- 层次只用两种手段：**表面色差**（`surface` ↔ `surface_raised`）与**字色深浅**（`on_surface` ↔ `on_surface_variant`）。
  不要写第三个灰来分第三层。
- 需要改色时**先改 `colors.xml`**，不要在布局里加新的字面值。

### 2.3 按钮颜色必须是 ColorStateList，不能是纯色

`res/color/` 下的 `btn_primary_bg.xml` / `btn_primary_text.xml` / `btn_text_label.xml` 是必需的。

> **这是已经踩过的坑**：把 `backgroundTint` 与 `android:textColor` 直接写成 `@color/accent` / `@color/on_accent`，
> 会**覆盖掉 MaterialButton 自带的禁用态外观**——`HistoryActivity` 在错题本为空时会
> `buttonReviewMistakes.isEnabled = false`，但按钮看起来仍然是饱和可点的墨蓝，用户点了没反应却不知道为什么。
>
> **约定**：任何可能被 `isEnabled = false` 的按钮，其 `backgroundTint` 与 `textColor` 都必须指向 `res/color/` 下的 selector。

---

## 3. 排版

### 3.1 字号档位

定义在 `styles.xml`，**布局里用 `style="@style/TextAppearance.App.*"`，不得再写 `android:textSize` 字面值**。

| 样式 | 字号 | 字重 | 用途 |
|---|---|---|---|
| `TextAppearance.App.Display` | 40sp | `sans-serif-medium` | 练习题面（`textExpression`） |
| `TextAppearance.App.Headline` | 30sp | `sans-serif-medium` | **页面主视觉**：识别结果 / 判定 / 结算成绩 / 累计正确率 |
| `TextAppearance.App.Title` | 17sp | `sans-serif-medium` | 分区小标题 |
| `TextAppearance.App.Body` | 15sp | normal | 列表主行、明细行、入口行 |
| `TextAppearance.App.Caption` | 13sp | normal + `letterSpacing 0.02` | 提示行、进度、年级标签 |
| `App.ActionBar.Title` | 19sp | `sans-serif-medium` | ActionBar 标题 |

### 3.2 约定

- **只用这 6 档**。不要为某个控件临时取 18sp、20sp 这类值。
- **字重只有「默认 / `sans-serif-medium`」两种。** 不要用 `android:textStyle="bold"`——
  bold 在中文下过重，Calm UI 要的是「清晰但不喊」。也不要引入自定义字体文件。
- 每个页面同时只可见**一个** 30sp 主视觉。练习页的 `textFeedback` 与 `textSummary` 分属
  `groupQuestion` / `groupResult` 两个互斥容器，永远不会同时显示——新增同类元素时保持这个约束。
- 不使用 `italic`。

---

## 4. 间距

### 4.1 令牌

定义在 `dimens.xml`，**布局里不得再写 `layout_margin*` 字面值**。

| 令牌 | 值 |
|---|---|
| `space_xs` / `space_sm` / `space_md` / `space_lg` / `space_xl` / `space_2xl` | 4 / 8 / 12 / 16 / 24 / 32 dp |
| `page_padding_h` / `page_padding_v` | 20 / 16 dp |
| `practice_padding_v` | **12dp**（见 4.3） |
| `radius_canvas` / `radius_block` | 24 / 16 dp |
| `touch_target` / `entry_row_height` | 48 / 56 dp |
| `canvas_home_height` | 300dp |
| `canvas_practice_min_height` | 160dp |

### 4.2 通用约定

- **间距只用 4 / 8 / 12 / 16 / 24 / 32 六档。**
- 用 `layout_marginTop` 表达「与上一行的距离」，**不要用 `layout_margin` 整圈加边距**——
  那会与根容器的 padding 叠加，产生一层看不出来源的额外留白。
- 水平方向用 `layout_marginStart` / `paddingStart`（不要用 `Left` / `Right`，项目 `supportsRtl="true"`）。
- **同一用途必须取同一个值**：所有分区标题都是 `space_xl`，所有列表项都是 `space_sm`。

### 4.3 练习页的垂直间距是预算，不是审美

练习页的垂直间距**刻意比另外两页紧**：根 padding 用 `practice_padding_v`(12dp) 而不是 `page_padding_v`(16dp)，
块间用 `space_sm`(8dp) 而不是 `space_lg`(16dp)。

**原因**：该页画布占 `weight=1` 吃剩余空间，而实测准确率主要由「**笔宽 / 手写数字高度**」的比例决定
（安全区 ≤0.19，见第 6 节）。**画布每高一个 dp，比例就多一分留在安全区的机会。**
这里省下的每一个 dp 都是直接从画布身上让出来的。

**约定**：不要为了「跟首页对齐」把这些值调大。要调整练习页的垂直间距，先确认画布实际高度仍远大于 168dp
（`32f / 0.19`，即笔宽决定的字高下限）。

---

## 5. 形状与视觉层次

**只有两档圆角：24dp（画布）与 16dp（soft 色块）。按钮是全圆角胶囊。**

- `bg_canvas.xml` — 纯白 + 24dp 圆角，只用在包住画布的 `FrameLayout` 上。
- `bg_block.xml` — 纯白 + 16dp 圆角，用在列表项、统计项、错题项。

**现状：没有任何阴影、没有任何边框。** 三个布局中不存在 `elevation`、不存在 `CardView`、
不存在 `stroke`、也不存在 1dp 分割线。

**约定**：

- 视觉层次靠**字号 + 字距 + 表面色差 + 留白**表达。
- **允许**用 `bg_block` 把一组内容「浮」成一块——这是本项目唯一允许的分组手段。
- **禁止**引入 `MaterialCardView`、`elevation`、`app:strokeColor`、渐变或分割线。
  确实需要时，先更新本文档再动手。
- 功能入口行（首页的「数学练习」「学习记录」）是整块可点的 `Widget.App.Button.Block`，
  右端 `ic_chevron_right` 指示符，靠色差成块，不画框不投影。

### 5.1 图标

项目只用一个业务图标：`ic_chevron_right.xml`（功能入口行的 `›`）。
颜色由 `app:iconTint` 决定，**不要**在 vector 里写死颜色。启动图标资源不在此规范范围内。

---

## 6. 画布组件（`FingerPaintView`）

画布是本项目唯一的自定义 View，也是视觉与识别精度耦合最紧的地方。

| 属性 | 值 | 说明 |
|---|---|---|
| 背景 | `@color/surface_raised`（纯白） | 与 `exportDrawingBitmap()` 内部 `drawColor(WHITE)` 一致 |
| 笔迹 | `Color.BLACK`，`STROKE`，`Cap.ROUND` + `Join.ROUND` | 对齐 MNIST 训练分布 |
| 笔宽 | `32f` | **不是外观选择**，见 6.1 |
| 圆角 | 无（**由外层容器裁剪**） | 见 6.2 |
| 高度（识别页） | 固定 `canvas_home_height` = 300dp | 单数字识别，字写得足够大 |
| 高度（练习页） | 外层容器 `0dp` + `weight=1` + `minHeight=160dp` | 吃掉剩余空间，画布越高越容易把比例留在安全区 |
| 冻结输入 | `inputEnabled = false` | 见 6.4 |

`FingerPaintView` **没有任何自定义 XML 属性**（`attrs` 参数被忽略，无 `declare-styleable`）。
笔宽、笔色、`touchTolerance` 全部硬编码在 `init` 里。

### 6.1 笔宽是精度参数，不是样式参数

实测准确率主要由「**笔宽 / 手写数字高度**」的比例决定（安全区 ≤0.19，0.20–0.25 开始退化，≥0.30 崩塌），
而不是由切分算法或模型决定。`32f` 让安全区覆盖约 170–800px 的字高；改回接近 `64f` 会让 4 位数整串准确率
从 87% 掉到 7%。**修改笔宽前必须重新测量。**

推论（属于界面规范的一部分）：**识别页把一次识别的位数上限定为 4 位**（`MAX_RECOGNIZED_DIGITS`），
因为位数越多每个数字越小、比例越容易越过悬崖。界面文案要主动引导学生「数字写大、写少」。

### 6.2 圆角靠容器裁剪，不靠画布自己

白色画布本身是方的（`drawingBitmap` 是一个矩形）。做法是把它包进一个 `FrameLayout`：

```xml
<FrameLayout
    android:background="@drawable/bg_canvas"
    android:clipToOutline="true" ...>
    <com.github.lany192.mnist.FingerPaintView ... android:background="@color/surface_raised" />
</FrameLayout>
```

**为什么安全**：`clipToOutline` 只影响**绘制**。`exportDrawingBitmap()` 读的是 `FingerPaintView`
内部的 `drawingBitmap`，仍是原始分辨率的纯白底位图，识别输入完全不受圆角影响。

**约定**：
- 绝不要把圆角做成「画布自己带 alpha 的圆角背景」——那才会污染导出。
- **包住画布的 `FrameLayout` 不得加 `padding`**，它会直接吃掉画布尺寸。
- `activity_main` 的 `FingerPaintView` 背景必须保持 `@color/surface_raised`；`exportDrawingBitmap()`
  会用纯白覆盖，所以显示用的背景色本身不影响识别，但**不要**改成非白——那会让「画布看起来是什么颜色」
  与「导出的位图是什么」分叉，下一个改代码的人会据此做出错误假设。

### 6.3 画布尺寸是硬约束

- **识别页**：固定高度，不参与 `weight`，因此可以自由地改周围布局，无笔迹丢失风险。
- **练习页**：占据 `weight=1` 的剩余空间，`onSizeChanged` 会**重建 `drawingBitmap`**。
  因此**任何改变画布高度的改动都会静默擦掉学生刚写的字迹**，学生的直接感受是「提交把答案吃掉了」。
  相关规则见第 7 节。

另外：`FingerPaintView.clear()` 必须保持空安全（`drawingBitmap` 在 `onSizeChanged` 中才创建），
`MathPracticeActivity.onCreate` 会在首次布局前清空画布以展示新题。

### 6.4 冻结画布

用 `FingerPaintView.inputEnabled = false`，**不能用 `setEnabled(false)`**——
该 View 无条件覆写 `onTouchEvent` 且恒返回 `true`，标准的 enabled 拦截在这里不生效。

---

## 7. 可见性切换规则（本项目最硬的界面约束）

这条规则纯看视觉效果是**反直觉**的，但它防的是数据丢失，所以写进设计规范。

练习页画布占据 `weight=1` 的剩余空间，因此**画布周围任何在 `GONE` 和 `VISIBLE` 之间切换的兄弟控件，
只要它那行是 `wrap_content`，都会挤动画布高度 → 触发 `onSizeChanged` 重建位图 → 擦掉学生刚写的字迹**。

| 控件 | 处理方式 | 原因 |
|---|---|---|
| `textCheckHint` | 默认 `invisible`，切换 `VISIBLE`/`INVISIBLE` | 用 `gone` 占位为 0，一出现就把画布挤矮 |
| `textGradeLabel` / `spinnerGrade` | 重做态 `INVISIBLE` + `isEnabled=false` | 该行是 `wrap_content`；缩小它会让画布变高 |
| `buttonReviewMistakes` | 用 `gone`，但**放在 `groupResult` 内部** | 作答态 `groupResult` 整体 `GONE`，所以它不参与作答态的高度计算 |
| `textFeedback` / `textDetail` | 作答态写入空串 `""`，**不用 `INVISIBLE` 也不用 `GONE`** | 空 TextView 仍保留行高，等价于预留位；这样结果出现时画布不跳 |

**约定**：

1. 给练习页画布新增兄弟控件时，条件显示一律用 `INVISIBLE`，或者给它预留固定高度（`minHeight`）。
   用 `GONE` 前必须先确认它不会改变画布高度。
2. **XML 里的默认可见性必须与 `render()` 对初始状态（`Answering`）生成的结果逐项一致。**
   `render()` 在 `onCreate` 和每次 `onStart` 都会运行；两者不一致时，第一次 render 就会翻转可见性、挤动画布。
3. **也不要改 `textCheckHint` 的 `TextAppearance` 档位**——字号变化等价于行高变化，
   会静态地改变画布高度。改字号前先确认画布实际高度仍远大于 168dp。
4. 按钮行所有按钮统一 `minHeight = touch_target`(48dp)，行高恒定，因此按钮的显隐不影响画布。
5. `activity_main` / `activity_history` 没有 `weight` 画布，可以自由使用 `GONE`
   （`textEmpty`、`textMistakesBeyond`、`titleSessions`、`titleMistakes` 就是这么做的）。

---

## 8. 语义色与反馈

**这是本次改版唯一的功能性新增。** 在此之前答对答错只靠文案区分。

| 控件 | 阶段 | 文案 | 颜色 |
|---|---|---|---|
| `textFeedback` | `Confirming` | 「识别结果：X」 | `on_surface`（中性） |
| `textFeedback` | `Judged` | 「回答错误，正确答案是 X」 | `wrong` |
| `textFeedback` | `Answering` | `""` | 保持 `on_surface` |
| `textSummary` | `Finished` 且全部答对 | 「全部答对 N 题，太棒了！」 | `correct` |
| `textSummary` | `Finished` 且有错题 | 「答对 X / Y 题」 | `on_surface` |
| 识别页 `textResult` | — | 「识别结果：X」 | `on_surface`（识别无对错） |

**约定：颜色必须与文案同源。** 上面的每一行颜色都由**选择那句文案的同一个条件**决定
（见 `MathPracticeActivity.render()`）。这保证不可能出现「文案说答错、颜色说答对」的漂移。
**不要**改成去解析 `textView.text` 的内容来判断对错。

> **注意**：`Judged` 态由 `MathPracticeViewModel` 保证只可能是答错——答对会直接 `advance` 到下一题，
> 不停顿。因此**界面上不存在「绿色答对反馈」的时机**，`correct` 只用在结算页的全对场景。
> 如果将来要加「答对也停留一下」的正向反馈，那是 ViewModel 状态机的改动，会连带影响
> `MathPracticeViewModelTest`，不属于界面规范范围。

---

## 9. 文案规范

### 9.1 存放位置

- 所有面向用户的文案写在 `res/values/strings.xml`，**Kotlin 侧不出现用户可见的字面量**（`Activity` 里一律 `getString(...)`）。
- 前缀分组：识别页无前缀（`hint_write` / `result_format` / `toast_*`）；练习页 `math_*`；历史页 `history_*`。
- 带参数的文案用**位置化占位符** `%1$d` / `%1$s`，不用裸 `%d`，便于将来调整语序。
- `String.format` 之外的拼接只用于不可本地化的场合（例如历史页把年级名与「错题重做」标签拼成一句）。

### 9.2 年级名单

`grade_names` 数组的**顺序必须与 `Grade` 枚举一致**：界面把 Spinner 的 `position` 直接当作枚举下标使用
（`Grade.values()[position]`），历史页也用 `grade.ordinal` 去查这个数组。改动任一侧都必须同时改另一侧。

### 9.3 语气

面向小学生，因此：

- 人称用「你」（"你写的是 96"），不用「您」，不用被动语态。
- 错误提示要**说清下一步怎么办**，不能只说"错了"：`toast_no_ink`「没检测到有效笔画，请写大一点」、
  `toast_too_many`「位数太多……请写大一些并减少位数」。
- 提示行要给出具体写法：`hint_write`「数字尽量写大、留空隙；小数点写在右下角」。
  这不是装饰性文案——它直接影响识别准确率（见 6.1）。
- 结算文案在全部答对时给正向反馈（`math_summary_all_correct`），并且会用 `correct` 色强化。

### 9.4 反馈渠道

| 渠道 | 用途 | 例 |
|---|---|---|
| 行内 `TextView` | 需要留存、需要逐位核对的信息（识别结果、明细、成绩、错题回顾） | `textFeedback`、`textDetail` |
| `Toast` | 一次性、不需要留存的操作拒绝 | 空画布、未识别、位数超限、小数点位置非法 |

**约定**：`Toast` 只用于「这次操作没生效，请重来」；任何需要学生看了之后再决定下一步的信息
（尤其是识别明细）必须走行内 `TextView`，不能用 `Toast`——练习页的「确认 / 重写」两次点击流程
依赖学生能看清逐字形拆分。

---

## 10. 禁止事项

- 禁止引入 Jetpack Compose 或自建设计令牌模块。本项目是单模块 XML 项目，混入第二套体系会立刻分叉。
- 禁止在布局中写**字面色值**、**字面字号**或**字面间距**——一律用 `@color/`、`@style/TextAppearance.App.*`、`@dimen/`。
- 禁止新增第 7 档字号或第 7 档间距。
- 禁止引入第二个强调色。语义色 `correct` / `wrong` 只用于反馈文字，不得用作背景、图标或边框。
- 禁止**新增阴影、边框、分割线或 `MaterialCardView`**；分组只用 `bg_block` 的软色块。
  允许的例外只有 `outline` 色，且它当前未被使用。
- 禁止在练习页给画布的兄弟控件用 `GONE` 做条件显示（见第 7 节）。
- 禁止为了"调外观"修改 `FingerPaintView` 的 `strokeWidth`、`touchTolerance`、笔迹颜色
  或 `exportDrawingBitmap()` 的导出逻辑（见 6.1 / 6.2）。
- 禁止给包住画布的 `FrameLayout` 加 `padding`。
- 禁止把可能被禁用的按钮的 `backgroundTint` / `textColor` 写成纯色（见 2.3）。
- 禁止在 Kotlin 代码里硬编码用户可见文案。
- 禁止新增 `values-night` 或深色资源而不同步更新本文档——当前的 Light-only 是明确现状，不是疏漏。
- 禁止改用 `NoActionBar` 而不补齐 inset 处理（见 1.1）。

---

## 11. 待收敛项

以下都是**现状**，不是缺陷。清单按收益排序，尚未执行：

| 现状 | 影响 | 建议 |
|---|---|---|
| 无深色模式（`Light` 而非 `DayNight`） | 系统深色下界面仍为浅色 | 建 `values-night/`，逐项核对本文档所有色对的对比度，再改父主题 |
| `textCheckHint` / `textHint` / `textFeedback` 在作答态写入空串以预留行高 | 练习页约有 70dp 的高度被预留位占用，无法让给画布 | 这是**刻意的**：不预留就会在结果出现时挤动画布、清空笔迹。除非改动 `onSizeChanged` 的重建策略，否则不要动 |
| `item_grade_stat` / `item_history_session` 是单行整串文本（`"%1$s　答对 %2$d/%3$d"`） | 无法把年级名左对齐、成绩右对齐做成两栏 | 改成两 TextView 需要同时改 `HistoryActivity.renderStats` / `renderSessions` |
| `outline` 色已定义但未被使用 | 读代码时会疑惑它是否该用 | 需要列表分隔线时再用，否则可删 |
| ActionBar 标题来自 `app_name`（"MnistTFLite"） | 首页标题是包名风格，不是中文产品名 | 改 `strings.xml` 的 `app_name`，或给 `MainActivity` 单独设 `android:label` |
| `constraintlayout` 依赖已引入但代码未使用 | 读代码时无法判断项目用哪套布局体系 | 明确它是预留还是传递依赖，或移除 |

**约定**：执行上述任何一项收敛时，**必须保持渲染结果逐像素不变**——这些改动是纯重构，不携带视觉变更；
视觉如有变化需在提交信息中写明并同步更新本文档。

---

## 12. 快速参考

```
底色   surface #F7F6F3 │ soft 色块 / 画布  surface_raised #FFFFFF
文字   on_surface #1C1B1A（16.6:1）│ on_surface_variant #6B6862（5.14:1）
强调   accent #3E5C76（唯一彩色，6.48:1）│ 禁用 on_surface_disabled #A5A29B
语义   correct #3F6B4F（仅结算全对）│ wrong #8C4A3F（仅判定答错）—— 只用于文字

字号  40 Display 题面 │ 30 Headline 主视觉 │ 17 Title 分区 │ 15 Body │ 13 Caption
字重  只有 默认 / sans-serif-medium，不用 bold

间距  4/8/12/16/24/32 │ 页边距 20×16 │ 练习页垂直 12（省给画布）│ 触控目标 48
圆角  24 画布 │ 16 色块 │ 全圆角 按钮        阴影/边框/分割线：一律没有

画布  32f 笔宽（精度参数，勿改）│ 识别页 300dp 固定 / 练习页 weight=1 + minHeight 160dp
      圆角靠外层 FrameLayout 的 clipToOutline 裁，容器不得加 padding

返回  二级页统一用 ActionBar 左上角的返回箭头，页面内容区不设返回按钮
      ActionBar 起排位置 = page_padding_h，与页面内容左边缘对齐

铁律  练习页画布周围的条件显示用 INVISIBLE，不用 GONE —— 否则 onSizeChanged 会清空笔迹
      按钮的 backgroundTint/textColor 必须用 res/color/ 的 selector，否则禁用态不可见
      颜色与文案同源：不解析 text 内容判断对错

文案  全部走 strings.xml，位置化占位符 %1$s，人称用「你」
```

---

*最后更新: 2026-09-15*
*基于当前仓库实际代码（`app/src/main/res/`、三个 Activity 的 `render()`）逐项核对生成，
并在真机（1080×2340）上逐页截图验证过：识别页 / 练习页 / 学习记录页（空态与有数据态），
含「写字 → 提交 → 确认」全程画布尺寸不变的回归验证。*
*识别链路、MVI 与持久化的工程约束见 `AGENTS.md` / `CLAUDE.md`，本文只覆盖界面。*
