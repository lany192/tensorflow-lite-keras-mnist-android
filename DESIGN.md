# 界面设计规范

本文档记录本项目界面**实际使用**的颜色、字号、间距、布局骨架与文案约定，作为新增页面、控件和调整样式时的依据。

> **前置事实**：本项目**没有设计令牌模块**，也**没有使用 Jetpack Compose**。
> 界面全部是纯 XML 布局 + ViewBinding（AppCompat），取值以字面值直接写在
> `app/src/main/res/layout/` 下；主题只有 `values/styles.xml` 里一个 `AppTheme`。
> 因此本文的定位不是"实现一套设计系统"，而是把现有界面的取值固定下来、并约束后续改动，
> 避免同一套视觉继续分叉。文中标注「现状」的是既有事实，标注「约定」的是新增代码必须遵守的规则。

---

## 1. 技术前提

| 项 | 现状 |
|---|---|
| 界面技术 | XML 布局 + ViewBinding（`buildFeatures.viewBinding = true`），无 Compose |
| 主题 | `Theme.AppCompat.Light.DarkActionBar`（`res/values/styles.xml`），`colorPrimary`/`colorPrimaryDark`/`colorAccent` 均为 `#1E1E1E` |
| 控件 | AppCompat `TextView` / `Button` / `Spinner` / `ScrollView`，全部用系统默认样式，无自定义 style |
| 自定义资源 | **无** `dimens.xml`、无 `values-night`、无自定义字体（`fontFamily` 全程未出现）、无 `<shape>` drawable、无 Card / elevation |
| 深色模式 | **未适配**：主题是 Light 系，没有 `values-night` 目录 |
| `material` 依赖 | 已在 `app/build.gradle.kts` 引入，但**当前代码未引用任何 Material 控件或 `Theme.MaterialComponents`** |
| 页面骨架 | 三个 Activity 都是「根 `LinearLayout`（`vertical`）+ `padding=10dp` + `background="#dddddd"`」 |
| 页面标题 | 由 ActionBar 提供（`AndroidManifest.xml` 的 `android:label`），页面内部不再重复标题 |
| 列表 | 不用 `RecyclerView`：`activity_history` 用「`LinearLayout` + 运行时 inflate 子项」，数据量是个位到几十条 |

**约定**：新页面沿用同一套骨架（根 `LinearLayout` vertical + `10dp` padding + `#dddddd` 背景 + ActionBar 标题），不要新建主题、不要引入 ConstraintLayout 之外的布局体系（`constraintlayout` 依赖已引入但当前未使用）。

---

## 2. 颜色

### 2.1 事实色表

| 用途 | 值 | 书写位置 | 出现次数 |
|---|---|---|---|
| 页面背景 | `#dddddd` | 三个 Activity 根布局的 `android:background` | 3 |
| 画布背景 | `#ffffff` | `FingerPaintView` 的 `android:background` | 2 |
| 笔迹 | `Color.BLACK` | `FingerPaintView` 的 `paint`（Kotlin 常量，非 XML） | 1 |
| 次要文字 | `#666666` | 书写提示行、识别明细、"你写的是…" | 8 |
| 主文字 | 不指定 `textColor`，走主题默认 | 标题、结果、列表主行 | — |
| 主题色 | `#1E1E1E` | `res/values/colors.xml` 的 `colorPrimary`/`colorPrimaryDark`/`colorAccent` | 3 |

**约定**：

- 颜色只用上表的值。**不要引入新的灰阶或任何彩色**——本项目是无品牌色的工具型界面，视觉层次完全靠「字重 + 字号 + 两种灰」建立。
- 主文字与次要文字的区分方式是**「不写 `textColor`」 vs 「写 `#666666`」**，不要写第三个颜色来区分出第三层次。
- 语义色（红/绿/黄）当前**不存在**：答对答错只靠文案（"答对 7 / 10 题"、"回答错误，正确答案是 68"）表达，不靠颜色。新增颜色前先更新本文档。
- 需要改色时，**先落到 `colors.xml` 再引用**，不要继续在布局里加新的字面值。

### 2.2 已知问题：次要文字对比度

`#666666` 放在 `#dddddd` 页面底上，对比度约 **4.23:1**，低于 WCAG AA 对正常字号文本要求的 4.5:1；而使用它的全是 12sp / 14sp 文本。若将来要做无障碍适配，`#595959` 可到约 5.16:1。**当前未修改**，仅在本文档记录。

---

## 3. 排版

### 3.1 字号档位（现状）

字号全部写成 `android:textSize` 字面值，共 21 处、5 档：

| 字号 | 字重 | 用途 | 位置 |
|---|---|---|---|
| `40sp` | bold | 练习题面（`23 + 45 = ?`） | `activity_math_practice.textExpression` |
| `28sp` | bold | **页面主视觉**：识别结果 / 判错信息 / 结算成绩 / 累计正确率 | `textResult`、`textFeedback`、`textSummary`、`textOverall` |
| `16sp` | bold | 分区小标题 | `titleSessions`、`titleMistakes` |
| `16sp` | normal | 列表主行、明细行、进度、错题回顾 | `item_grade_stat`、`item_history_session`、`item_mistake.textMistakeProblem`、`textEmpty`、`textReview`、`textProgress` |
| `14sp` | normal | 识别明细、"你写的是…" | `textDetail`×2、`textMistakeWritten` |
| `12sp` | normal | 书写提示、超出提示 | `textHint`×2、`textCheckHint`、`textMistakesBeyond` |
| 不指定（主题默认 ≈14sp） | normal | 年级标签 | `textGradeLabel` |

### 3.2 约定

- **只用这 5 档字号：40 / 28 / 16 / 14 / 12。** 不要为某个控件临时取 18sp、20sp 这类值。
- `textStyle="bold"` 只用于三类：题面、页面主视觉、分区小标题。**提示行和明细行一律不加粗。**
- 每个页面同时只可见**一个** 28sp bold 主视觉。练习页的 `textFeedback` 与 `textSummary` 都是 28sp bold，但它们分属 `groupQuestion` / `groupResult` 两个互斥容器，永远不会同时显示——新增同类元素时保持这个约束。
- 字重只有「默认 / bold」两种：不使用 `italic`、不使用 `textStyle` 以外的字重手段。
- 字体恒为系统默认，**不引入自定义字体文件**。
- `textGradeLabel` 目前依赖主题默认值；新增控件时若依赖默认值，在本文档登记，不要留下"看不出字号是多少"的隐式依赖。

---

## 4. 间距

### 4.1 垂直外边距（`layout_marginTop`）

| 值 | 用途 | 出现位置 |
|---|---|---|
| `24dp` | 空态提示与顶部主视觉之间 | `activity_history.textEmpty` |
| `16dp` | 分区标题上方 | `titleSessions`、`titleMistakes` |
| `12dp` | 主视觉下方（结果 → 明细 / 成绩 → 错题标题） | `activity_main.textResult`、`activity_math_practice` 错题标题 |
| `8dp` | 块与块、按钮行、画布上方 | 各页按钮行、`containerStats`、练习题面与画布 |
| `6dp` | 提示行紧跟画布 | `activity_main.textHint`（**仅此一处**） |
| `4dp` | 明细/提示紧跟上一行 | `textDetail`、`textCheckHint`、`textHint`、`textFeedback` |
| `2dp` | 列表项之间 | `item_mistake` 根布局 |

### 4.2 水平与内边距

| 值 | 用途 |
|---|---|
| 根容器 `padding=10dp` | 三个 Activity 统一 |
| `layout_marginLeft=8dp` | 按钮之间 |
| `layout_marginLeft=4dp` | 标签与 Spinner 之间 |
| `paddingTop/Bottom=2dp` | 列表项行（`item_grade_stat`、`item_history_session`） |

### 4.3 约定

- 根容器固定 `10dp`；**其余间距只用 2 / 4 / 8 / 12 / 16 / 24 六档**。`6dp` 是既有遗留值，新界面不要再引入（就近取 `4dp` 或 `8dp`）。
- 用 `layout_marginTop` 表达「与上一行的距离」，**不要用 `layout_margin` 整圈加边距**——那会与根容器的 `10dp` padding 叠加，产生一层看不出来源的额外留白。
- 不写 `SpaceVertical*` 之类的辅助函数或 `<dimen>` 引用（本项目没有）。但**同一用途必须取同一个值**：所有分区标题都是 `16dp`，所有按钮行都是 `8dp`。

---

## 5. 形状与视觉层次

**现状：没有任何圆角、边框、阴影、卡片。**

- 三个布局中不存在 `<shape>` drawable，`android:background` 只接纯色字面值。
- 没有 CardView、没有 `elevation`、没有分割线（`View` 高度 1dp 那种也没有）。
- 按钮的圆角与底色来自 AppCompat 默认样式，由 `AppTheme` 的 `colorAccent` 决定；**没有为按钮写过任何自定义 style**。

**约定**：

- 视觉层次靠**字号 + 字重 + 两种灰 + 留白**表达，不靠卡片、圆角或阴影。新增界面不要引入 Card 或自定义 `<shape>`。
- 需要视觉分组时，用 `16dp` 上边距 + `16sp bold` 小标题（`activity_history` 的「历史练习」「错题本」就是范例），不要画框。
- 确实需要卡片效果时，先更新本文档再动手。

---

## 6. 画布组件（`FingerPaintView`）

画布是本项目唯一的自定义 View，也是视觉与识别精度耦合最紧的地方。

| 属性 | 值 | 说明 |
|---|---|---|
| 背景 | `#ffffff`（XML 字面值） | 与 `exportDrawingBitmap()` 内部 `drawColor(WHITE)` 一致 |
| 笔迹 | `Color.BLACK`，`STROKE`，`Cap.ROUND` + `Join.ROUND` | 对齐 MNIST 训练分布 |
| 笔宽 | `32f` | **不是外观选择**，见下 |
| 高度（识别页） | 固定 `300dp` | 单数字识别，字写得足够大 |
| 高度（练习页） | `0dp` + `weight=1` + `minHeight=160dp` | 吃掉剩余空间，画布越高越容易把比例留在安全区 |
| 冻结输入 | `inputEnabled = false` | 见 6.2 |

### 6.1 笔宽是精度参数，不是样式参数

实测准确率主要由「**笔宽 / 手写数字高度**」的比例决定（安全区 ≤0.19，0.20–0.25 开始退化，≥0.30 崩塌），而不是由切分算法或模型决定。`32f` 让安全区覆盖约 170–800px 的字高；改回接近 `64f` 会让 4 位数整串准确率从 87% 掉到 7%。**修改笔宽前必须重新测量。**

推论（属于界面规范的一部分）：**识别页把一次识别的位数上限定为 4 位**（`MAX_RECOGNIZED_DIGITS`），因为位数越多每个数字越小、比例越容易越过悬崖。界面文案要主动引导学生「数字写大、写少」。

### 6.2 画布尺寸是硬约束

练习页的画布吃 `weight=1` 的剩余空间，`onSizeChanged` 会**重建 `drawingBitmap`**。因此**任何改变画布高度的布局改动都会静默擦掉学生刚写的字迹**，学生的直接感受是「提交把答案吃掉了」。相关规则见第 7 节。

另外：`FingerPaintView.clear()` 必须保持空安全（`drawingBitmap` 在 `onSizeChanged` 中才创建），`MathPracticeActivity.onCreate` 会在首次布局前清空画布以展示新题。

---

## 7. 可见性切换规则（本项目最硬的界面约束）

这条规则纯看视觉效果是**反直觉**的，但它防的是数据丢失，所以写进设计规范。

练习页画布占据 `weight=1` 的剩余空间，因此**画布周围任何在 `GONE` 和 `VISIBLE` 之间切换的兄弟控件，只要它那行是 `wrap_content`，都会挤动画布高度 → 触发 `onSizeChanged` 重建位图 → 擦掉学生刚写的字迹**。

| 控件 | 处理方式 | 原因 |
|---|---|---|
| `textCheckHint` | 默认 `invisible`，切换 `VISIBLE`/`INVISIBLE` | 用 `gone` 占位为 0，一出现就把画布挤矮 |
| `textGradeLabel` / `spinnerGrade` | 重做态 `INVISIBLE` + `isEnabled=false` | 该行是 `wrap_content`；缩小它会让画布变高 |
| `buttonReviewMistakes` | 用 `gone`，但**放在 `groupResult` 内部** | 作答态 `groupResult` 整体 `GONE`，所以它不参与作答态的高度计算 |

**约定**：

1. 给练习页画布新增兄弟控件时，条件显示一律用 `INVISIBLE`，或者给它预留固定高度（`minHeight`）。用 `GONE` 前必须先确认它不会改变画布高度。
2. **XML 里的默认可见性必须与 `render()` 对初始状态生成的可见性一致。** `render()` 在 `onCreate` 和每次 `onStart` 都会运行；两者不一致时，第一次 render 就会翻转可见性、挤动画布。
3. 冻结画布用 `FingerPaintView.inputEnabled = false`，**不能用 `setEnabled(false)`**——该 View 无条件覆写 `onTouchEvent` 且恒返回 `true`，标准的 enabled 拦截在这里不生效。
4. `activity_history` 没有画布，因此它可以自由使用 `GONE`（`textEmpty`、`textMistakesBeyond` 就是这么做的）。

---

## 8. 文案规范

### 8.1 存放位置

- 所有面向用户的文案写在 `res/values/strings.xml`，**Kotlin 侧不出现用户可见的字面量**（`Activity` 里一律 `getString(...)`）。
- 前缀分组：识别页无前缀（`hint_write` / `result_format` / `toast_*`）；练习页 `math_*`；历史页 `history_*`。
- 带参数的文案用**位置化占位符** `%1$d` / `%1$s`，不用裸 `%d`，便于将来调整语序。
- `String.format` 之外的拼接只用于不可本地化的场合（例如历史页把年级名与「错题重做」标签拼成一句）。

### 8.2 年级名单

`grade_names` 数组的**顺序必须与 `Grade` 枚举一致**：界面把 Spinner 的 `position` 直接当作枚举下标使用（`Grade.values()[position]`），历史页也用 `grade.ordinal` 去查这个数组。改动任一侧都必须同时改另一侧。

### 8.3 语气

面向小学生，因此：

- 人称用「你」（"你写的是 96"），不用「您」，不用被动语态。
- 错误提示要**说清下一步怎么办**，不能只说"错了"：`toast_no_ink`「没检测到有效笔画，请写大一点」、`toast_too_many`「位数太多……请写大一些并减少位数」。
- 提示行要给出具体写法：`hint_write`「数字尽量写大、留空隙；小数点写在右下角」。这不是装饰性文案——它直接影响识别准确率（见 6.1）。
- 结算文案在全部答对时给正向反馈（`math_summary_all_correct`）。

### 8.4 反馈渠道

| 渠道 | 用途 | 例 |
|---|---|---|
| 行内 `TextView` | 需要留存、需要逐位核对的信息（识别结果、明细、成绩、错题回顾） | `textFeedback`、`textDetail` |
| `Toast` | 一次性、不需要留存的操作拒绝 | 空画布、未识别、位数超限、小数点位置非法 |

**约定**：`Toast` 只用于「这次操作没生效，请重来」；任何需要学生看了之后再决定下一步的信息（尤其是识别明细）必须走行内 `TextView`，不能用 `Toast`——练习页的「确认 / 重写」两次点击流程依赖学生能看清逐字形拆分。

---

## 9. 禁止事项

- 禁止引入 Jetpack Compose、Material3 主题或自建设计令牌模块。本项目是单模块 XML 项目，混入第二套体系会立刻分叉。
- 禁止在布局中新增第 6 档字号（40/28/16/14/12 之外）或第 7 档间距（2/4/8/12/16/24 之外）。
- 禁止新增语义色（红/绿/黄）或品牌色。
- 禁止在练习页给画布的兄弟控件用 `GONE` 做条件显示（见第 7 节）。
- 禁止为了"调外观"修改 `FingerPaintView` 的 `strokeWidth`、画布背景色或 `exportDrawingBitmap()` 的导出逻辑（见 6.1）。
- 禁止在 Kotlin 代码里硬编码用户可见文案。
- 禁止新增 `values-night` 或深色资源而不同步更新本文档——当前的 Light-only 是明确现状，不是疏漏。

---

## 10. 待收敛项

以下都是**现状**，不是缺陷，但继续增长会让改样式越来越贵。清单按收益排序，尚未执行：

| 现状 | 影响 | 建议 |
|---|---|---|
| `#dddddd` / `#ffffff` / `#666666` 共 13 处字面值 | 改一次颜色要全局搜索替换，且容易漏 | 收敛到 `colors.xml`（如 `bg_page` / `bg_canvas` / `text_secondary`）后引用 |
| 21 处 `textSize` 字面值 | 无法保证"同用途同字号" | 收敛到 `styles.xml` 的 `TextAppearance` 并复用 |
| 20+ 处 `layout_margin*` 字面值 | 同上 | 收敛到 `dimens.xml` |
| `textGradeLabel` 未指定 `textSize` | 字号隐式依赖主题默认值 | 显式补上，或抽成样式 |
| `#666666` 在 `#dddddd` 上对比度约 4.23:1 | 低于 WCAG AA 的 4.5:1 | 无障碍适配时改为 `#595959`（约 5.16:1） |
| `material` 依赖已引入但代码未使用 | 读代码时无法判断项目用的是 AppCompat 还是 Material 体系 | 明确它是预留还是传递依赖，或移除 |
| `constraintlayout` 依赖已引入但代码未使用 | 同上 | 同上 |

**约定**：执行上述任何一项收敛时，**必须保持渲染结果逐像素不变**——这些改动是纯重构，不携带视觉变更；视觉如有变化需在提交信息中写明并同步更新本文档。

---

## 11. 快速参考

```
页面背景      #dddddd      画布背景   #ffffff      笔迹  Color.BLACK
次要文字      #666666      主文字     不写 textColor（走主题默认）
主题色        #1E1E1E（colors.xml 的 colorPrimary / colorPrimaryDark / colorAccent）

字号  40sp bold 题面 │ 28sp bold 页面主视觉 │ 16sp 列表/小标题(bold) │ 14sp 明细 │ 12sp 提示
间距  根 padding 10dp │ 块间 8dp │ 标题上 16dp │ 空态上 24dp │ 明细上 4dp │ 列表项 2dp
形状  无圆角、无边框、无阴影、无卡片 —— 层次靠字号 + 字重 + 两种灰 + 留白
画布  32f 笔宽（精度参数，勿改）│ 识别页固定 300dp / 练习页 weight=1 + minHeight 160dp
铁律  练习页画布周围的条件显示用 INVISIBLE，不用 GONE —— 否则 onSizeChanged 会清空笔迹
文案  全部走 strings.xml，位置化占位符 %1$s，人称用「你」
```

---

*最后更新: 2026-09-13*
*基于当前仓库实际代码（`app/src/main/res/layout/`、`values/`、`FingerPaintView.kt`、三个 Activity 的 `render()`）逐项核对生成。*
*识别链路、MVI 与持久化的工程约束见 `AGENTS.md` / `CLAUDE.md`，本文只覆盖界面。*
