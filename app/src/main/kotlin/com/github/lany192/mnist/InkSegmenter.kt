package com.github.lany192.mnist

import kotlin.math.abs
import kotlin.math.roundToInt

/**
 * 闭区间包围盒，坐标语义与 Bitmap 像素一致（right/bottom 为包含端点）。
 */
data class Box(val left: Int, val top: Int, val right: Int, val bottom: Int) {
    val width: Int get() = right - left + 1
    val height: Int get() = bottom - top + 1
    val centerX: Float get() = (left + right) / 2f
    val area: Int get() = width * height

    fun union(other: Box): Box = Box(
        minOf(left, other.left),
        minOf(top, other.top),
        maxOf(right, other.right),
        maxOf(bottom, other.bottom)
    )
}

/**
 * 切分阈值。默认值针对本 App 的绘制条件（白底黑字、笔宽 64px、画布约 1080x900px）调校。
 */
data class SegmentConfig(
    /** 墨水阈值，必须与 MnistPreprocessor 内部一致，否则切分框和归一化框会错位。 */
    val inkThreshold: Int = 32,
    /** 小于该面积的连通域视为噪点。一次误触的圆点约 3200px，孤立抗锯齿像素 ≤ 4px。 */
    val minComponentArea: Int = 20,
    /** 两个连通域水平交叠比例达到该值即认为属于同一个数字（修复多笔画数字、"5"的横杠）。 */
    val mergeOverlapRatio: Float = 0.3f,
    /** 仅当存在多个组时生效：高度不足最高组该比例的组视为误触圆点。 */
    val minHeightRatio: Float = 0.25f,
    /** 由字高推算单个数字的参考宽度（手写数字宽高比典型 0.6~0.9）。 */
    val avgDigitAspect: Float = 0.75f,
    /** 触发二次切分的最小宽度：相对参考宽度。 */
    val splitWidthFactor: Float = 1.7f,
    /** 触发二次切分的最小宽高比（实际生效的主要守卫）。 */
    val splitAspectFactor: Float = 1.35f,
    /** 谷值判定：候选列的墨量不得超过最大列墨量的该比例。 */
    val neckMaxFraction: Float = 0.6f,
    /** 谷值搜索时排除左右各该比例的宽度，避免切在数字边缘。 */
    val neckSearchMargin: Float = 0.30f,
    /** 判定"谷壁"的倍数：相邻列墨量超过候选列该倍数即视为平台被谷壁封住。 */
    val neckWallFactor: Float = 2f,
    /** 最大递归切分深度。 */
    val maxSplitDepth: Int = 3,
    /** 实心块判定：包围盒填充率超过该值。圆点填充率 π/4≈0.785。 */
    val solidFillRatio: Float = 0.72f,
    /** 实心块判定：宽高比须落在该区间内，避免误杀填充率同样为 1.0 的实心竖条"1"。 */
    val blobAspectMin: Float = 0.75f,
    val blobAspectMax: Float = 1.33f,
    val enableNeckSplit: Boolean = true,
    val enableBlobFilter: Boolean = true,
    /** 连通域数量的防御上限，超过则判定为病态输入并放弃。 */
    val maxComponents: Int = 500
)

/**
 * 从整张手写画布中切出各个数字。
 *
 * 流程：8 连通域标记 → 按水平交叠聚组（同一数字的多个笔画）→ 过滤噪点/实心圆点
 * → 守卫式颈部二次切分（处理粘连的两个数字）→ 按书写顺序排序。
 *
 * 本类不依赖任何 Android API，因此切分逻辑可在 JVM 单元测试中直接验证。
 */
object InkSegmenter {

    /** ARGB 像素转墨水灰度：白底（255）= 0，黑笔（0）= 255。与旧版 preprocess 定义一致。 */
    fun toInkGray(pixels: IntArray): IntArray {
        val ink = IntArray(pixels.size)
        for (i in pixels.indices) {
            ink[i] = 0xff - (pixels[i] and 0xff)
        }
        return ink
    }

    /**
     * @param ink [toInkGray] 的输出，长度为 width*height。
     * @return 按书写顺序（centerX 升序）排列的紧包围盒；无有效笔画时返回空列表。
     */
    fun segment(
        ink: IntArray,
        width: Int,
        height: Int,
        config: SegmentConfig = SegmentConfig()
    ): List<Box> {
        if (width <= 0 || height <= 0 || ink.size < width * height) return emptyList()
        val components = labelComponents(ink, width, height, config)
        if (components.isEmpty()) return emptyList()
        var groups = groupComponents(components, config)
        if (groups.size > 1) groups = dropShortGroups(groups, config)
        if (config.enableBlobFilter) {
            groups = groups.filterNot { isSolidBlob(it, ink, width, config) }
        }
        if (groups.isEmpty()) return emptyList()
        // 参考宽度只由字高推导、不含宽度信息，所以只有一个组时二次切分的守卫退化成三条并列条件：
        // 宽度 >= 1.275H 与宽高比 >= 1.35（后者更严，是绑定约束），外加"切缝列只有一段墨"。
        // 正常手写 0-9 的宽高比都在 1.35 以下；宽而空心的数字（宽高比可能超过 1.35）由
        // "切缝列只有一段墨"挡住——空心数字的中段必然有上下两段墨。
        // 这是几条判据的组合而非结构保证：把笔画写成够宽的单段实心条仍会进入切分分支。
        val refWidth = medianHeight(groups) * config.avgDigitAspect
        val out = ArrayList<Box>(groups.size)
        for (group in groups) {
            splitWideGroup(ink, width, group, refWidth, 0, config, out)
        }
        out.sortBy { it.centerX }
        return out
    }

    private class Component(val box: Box, val area: Int)

    /** 迭代式 BFS 的 8 连通域标记。用 4 连通会把抗锯齿后的斜笔画碎成大量孤立像素点。 */
    private fun labelComponents(
        ink: IntArray,
        width: Int,
        height: Int,
        config: SegmentConfig
    ): List<Component> {
        val visited = BooleanArray(width * height)
        val queue = IntArray(width * height)
        val components = ArrayList<Component>()
        for (start in 0 until width * height) {
            if (visited[start] || ink[start] <= config.inkThreshold) continue
            var head = 0
            var tail = 0
            queue[tail++] = start
            visited[start] = true
            var minX = start % width
            var maxX = minX
            var minY = start / width
            var maxY = minY
            var area = 0
            while (head < tail) {
                val p = queue[head++]
                area++
                val x = p % width
                val y = p / width
                if (x < minX) minX = x
                if (x > maxX) maxX = x
                if (y < minY) minY = y
                if (y > maxY) maxY = y
                for (dy in -1..1) {
                    val ny = y + dy
                    if (ny < 0 || ny >= height) continue
                    val row = ny * width
                    for (dx in -1..1) {
                        if (dx == 0 && dy == 0) continue
                        val nx = x + dx
                        if (nx < 0 || nx >= width) continue
                        val q = row + nx
                        if (visited[q] || ink[q] <= config.inkThreshold) continue
                        visited[q] = true
                        queue[tail++] = q
                    }
                }
            }
            if (area >= config.minComponentArea) {
                components.add(Component(Box(minX, minY, maxX, maxY), area))
            }
            if (components.size > config.maxComponents) return emptyList()
        }
        return components
    }

    /**
     * 按水平交叠把连通域聚成一个数字。判据只看 X 交叠、忽略 Y：
     * 这既能让"5"的横杠与下半弧（上下分离）合并，也不受数字上下错位影响。
     */
    private fun groupComponents(components: List<Component>, config: SegmentConfig): List<Box> {
        val sorted = components.sortedBy { it.box.left }
        val groups = ArrayList<Box>()
        var current: Box? = null
        for (component in sorted) {
            val box = component.box
            if (current == null) {
                current = box
                continue
            }
            val intersection = minOf(current.right, box.right) - maxOf(current.left, box.left) + 1
            val denominator = minOf(current.width, box.width)
            if (intersection > 0 && denominator > 0 &&
                intersection.toFloat() / denominator >= config.mergeOverlapRatio
            ) {
                current = current.union(box)
            } else {
                groups.add(current)
                current = box
            }
        }
        if (current != null) groups.add(current)
        return groups
    }

    private fun dropShortGroups(groups: List<Box>, config: SegmentConfig): List<Box> {
        val refHeight = groups.maxOf { it.height }
        if (refHeight <= 0) return groups
        val kept = groups.filter { it.height >= config.minHeightRatio * refHeight }
        return kept.ifEmpty { groups }
    }

    /**
     * 误触留下的实心圆点：填充率高、接近正方形，且**内部没有足够大的孔洞**。
     *
     * 孔洞判据不可省略：把 "0"/"6"/"8"/"9" 写得小一点，粗笔画会把环孔挤没，
     * 填充率同样越过阈值，仅靠填充率会把这类数字整组删掉（静默漏一位）。
     * 环状数字再小也总是留有空腔，误触圆点则一定是实心的。
     */
    private fun isSolidBlob(box: Box, ink: IntArray, width: Int, config: SegmentConfig): Boolean {
        val aspect = box.width.toFloat() / box.height
        if (aspect < config.blobAspectMin || aspect > config.blobAspectMax) return false
        var inkCount = 0
        for (y in box.top..box.bottom) {
            val row = y * width
            for (x in box.left..box.right) {
                if (ink[row + x] > config.inkThreshold) inkCount++
            }
        }
        if (inkCount.toFloat() / box.area < config.solidFillRatio) return false
        // 孔洞还必须有面积下限：一个 1 像素的采样空洞就足以让误触圆点"复活"成一位数字。
        return largestHoleArea(ink, width, box, config) < config.minComponentArea
    }

    /**
     * 包围盒内最大的"孔洞"面积：不接触包围盒边界的背景连通域就是孔洞。
     * 用面积而非布尔值，是为了让 [isSolidBlob] 能用 [SegmentConfig.minComponentArea] 做下限。
     */
    private fun largestHoleArea(ink: IntArray, width: Int, box: Box, config: SegmentConfig): Int {
        val w = box.width
        val h = box.height
        val seen = BooleanArray(w * h)
        val queue = IntArray(w * h)
        var largest = 0
        for (startY in 0 until h) {
            for (startX in 0 until w) {
                val start = startY * w + startX
                if (seen[start] || isInk(ink, width, box, startX, startY, config)) continue
                var head = 0
                var tail = 0
                queue[tail++] = start
                seen[start] = true
                var touchesBorder = false
                var area = 0
                while (head < tail) {
                    val p = queue[head++]
                    area++
                    val x = p % w
                    val y = p / w
                    if (x == 0 || y == 0 || x == w - 1 || y == h - 1) touchesBorder = true
                    // 背景走 4 连通：前景是 8 连通，背景必须配对成 4 连通，否则当前景被夹到
                    // 单像素对角时孔洞会从对角"漏"到框外，环状数字被漏判成实心块。
                    if (x > 0 && !seen[p - 1] && !isInk(ink, width, box, x - 1, y, config)) {
                        seen[p - 1] = true
                        queue[tail++] = p - 1
                    }
                    if (x < w - 1 && !seen[p + 1] && !isInk(ink, width, box, x + 1, y, config)) {
                        seen[p + 1] = true
                        queue[tail++] = p + 1
                    }
                    if (y > 0 && !seen[p - w] && !isInk(ink, width, box, x, y - 1, config)) {
                        seen[p - w] = true
                        queue[tail++] = p - w
                    }
                    if (y < h - 1 && !seen[p + w] && !isInk(ink, width, box, x, y + 1, config)) {
                        seen[p + w] = true
                        queue[tail++] = p + w
                    }
                }
                if (!touchesBorder) {
                    largest = maxOf(largest, area)
                }
            }
        }
        return largest
    }

    private fun isInk(ink: IntArray, width: Int, box: Box, x: Int, y: Int, config: SegmentConfig): Boolean =
        ink[(box.top + y) * width + box.left + x] > config.inkThreshold

    private fun medianHeight(groups: List<Box>): Float {
        val heights = groups.map { it.height }.sorted()
        val n = heights.size
        return if (n % 2 == 1) {
            heights[n / 2].toFloat()
        } else {
            (heights[n / 2 - 1] + heights[n / 2]) / 2f
        }
    }

    /**
     * 守卫式颈部切分：只有同时满足"够宽""够扁"且中段存在明显墨量谷值时才会切开。
     * 守卫不满足时宁可保留为一个框（错一个数字），也不盲目切分（可能错成两个）。
     */
    private fun splitWideGroup(
        ink: IntArray,
        width: Int,
        box: Box,
        refWidth: Float,
        depth: Int,
        config: SegmentConfig,
        out: MutableList<Box>
    ) {
        val tooWide = box.width >= config.splitWidthFactor * refWidth
        val tooFlat = box.width >= config.splitAspectFactor * box.height
        if (!config.enableNeckSplit || depth >= config.maxSplitDepth || !tooWide || !tooFlat) {
            out.add(box)
            return
        }
        val margin = (config.neckSearchMargin * box.width).roundToInt()
        val from = box.left + margin
        val to = box.right - margin
        if (from >= to) {
            out.add(box)
            return
        }
        val stats = columnStats(ink, width, box, config)
        val columnInk = stats.ink
        // 候选列必须"只有一段墨"。缺少这条判据时，单个"宽而空心"的数字会被误切：
        // 空心数字中段的列墨量只有上下两条笔画带（≈2×笔宽），远低于左右壁的整字高（两段墨），
        // 谷值判据会把一个环看成"两瓣 + 细颈"。
        //
        // 筛选必须发生在求最小墨量**之前**：若先在整个窗口上取最小值、再检查选中列的段数，
        // 那么当"细颈（1 段）"与"空心数字中段（2 段）"的墨量恰好相等时（固定笔宽下很常见），
        // 平台会落在 2 段的那些列上，整次切分被放弃，而窗口里其实存在合法的 1 段切口。
        //
        // 分母取整个框的最大列墨量（笔画内部），分子取候选列的最小值（接缝）：若两者都只在
        // 窗口内统计，一段墨量均匀的粘连区会得到 min == max，永远切不开。
        val maxInk = columnInk.maxOrNull() ?: 0
        var minInk = Int.MAX_VALUE
        for (x in from..to) {
            if (stats.runs[x - box.left] > 1) continue
            val value = columnInk[x - box.left]
            if (value < minInk) minInk = value
        }
        if (maxInk <= 0 || minInk == Int.MAX_VALUE || minInk > config.neckMaxFraction * maxInk) {
            out.add(box)
            return
        }
        // 取最小墨量**连续段**的中点，而不是首个最小值：接缝往往是一条墨量均匀的窄带，
        // 从最左端下刀会把右侧数字的一部分划给左边，两个框同时变成垃圾。
        // 也不能简单地对"首个最小值…末个最小值"取中点：三个数字连环粘连时会有两条细颈，
        // 其间隔着一段实心笔画，取首末中点会把切点落到那段实心笔画正中间。
        var splitX = -1
        var bestStart = -1
        var bestEnd = -1
        var bestLength = -1
        var bestDistance = Int.MAX_VALUE
        val windowCenter = (from + to) / 2
        var runStart = -1
        for (x in from..(to + 1)) {
            val isCandidate = x <= to &&
                stats.runs[x - box.left] <= 1 &&
                columnInk[x - box.left] == minInk
            if (isCandidate) {
                if (runStart < 0) runStart = x
            } else if (runStart >= 0) {
                val length = x - runStart
                val runCenter = (runStart + x - 1) / 2
                val distance = abs(runCenter - windowCenter)
                if (length > bestLength || (length == bestLength && distance < bestDistance)) {
                    bestLength = length
                    bestDistance = distance
                    bestStart = runStart
                    bestEnd = x - 1
                    splitX = runCenter
                }
                runStart = -1
            }
        }
        if (splitX < 0) {
            out.add(box)
            return
        }
        // 平台只有一侧挨着墨量跃升（真正的"谷壁"）、另一侧被搜索窗口截断时，它并不是一条
        // 均匀细颈，而是低墨量区一直延伸进了相邻数字内部——例如接触区与"7"的长横墨量相同、
        // 连成一个平台。此时真正的接缝在谷壁那一端，取平台中心会把数字的长横一分为二。
        val wall = config.neckWallFactor * minInk
        val leftWall = bestStart > from && columnInk[bestStart - 1 - box.left] > wall
        val rightWall = bestEnd < to && columnInk[bestEnd + 1 - box.left] > wall
        if (leftWall != rightWall) {
            splitX = if (leftWall) bestStart else bestEnd
        }
        val left = tightBox(ink, width, box.left, box.top, splitX, box.bottom, config)
        val right = tightBox(ink, width, splitX + 1, box.top, box.right, box.bottom, config)
        if (left == null || right == null) {
            out.add(box)
            return
        }
        splitWideGroup(ink, width, left, refWidth, depth + 1, config, out)
        splitWideGroup(ink, width, right, refWidth, depth + 1, config, out)
    }

    /** 逐列的墨量与"该列被墨迹切成的段数"，下标为 x - box.left。细颈的段数为 1。 */
    private class ColumnStats(val ink: IntArray, val runs: IntArray)

    private fun columnStats(ink: IntArray, width: Int, box: Box, config: SegmentConfig): ColumnStats {
        val stats = ColumnStats(IntArray(box.width), IntArray(box.width))
        for (x in box.left..box.right) {
            var count = 0
            var runs = 0
            var inRun = false
            for (y in box.top..box.bottom) {
                val isInk = ink[y * width + x] > config.inkThreshold
                if (isInk) {
                    count++
                    if (!inRun) runs++
                }
                inRun = isInk
            }
            stats.ink[x - box.left] = count
            stats.runs[x - box.left] = runs
        }
        return stats
    }

    /** 在给定子区域内重新求紧包围盒，保证切分后的框仍然紧贴墨迹。 */
    private fun tightBox(
        ink: IntArray,
        width: Int,
        left: Int,
        top: Int,
        right: Int,
        bottom: Int,
        config: SegmentConfig
    ): Box? {
        var minX = right
        var maxX = left - 1
        var minY = bottom
        var maxY = top - 1
        for (y in top..bottom) {
            val row = y * width
            for (x in left..right) {
                if (ink[row + x] > config.inkThreshold) {
                    if (x < minX) minX = x
                    if (x > maxX) maxX = x
                    if (y < minY) minY = y
                    if (y > maxY) maxY = y
                }
            }
        }
        if (maxX < minX || maxY < minY) return null
        return Box(minX, minY, maxX, maxY)
    }
}
