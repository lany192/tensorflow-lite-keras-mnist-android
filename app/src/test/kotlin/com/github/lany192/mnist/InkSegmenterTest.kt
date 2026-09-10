package com.github.lany192.mnist

import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

/**
 * 切分算法的纯 JVM 测试：输入是 [InkSegmenter.toInkGray] 语义的数组（墨=255，白底=0）。
 * 所有画布尺寸都取小值，期望包围盒可以手算出来。
 */
class InkSegmenterTest {

    @Test
    fun emptyCanvas_returnsNoBoxes() {
        val ink = canvas(200, 100)
        assertTrue(InkSegmenter.segment(ink, 200, 100).isEmpty())
    }

    /** 单数字不变量的核心断言：输出的框就是全图墨迹包围盒。 */
    @Test
    fun singleDigit_boxEqualsTightInkBox() {
        val ink = canvas(200, 260)
        rect(ink, 200, 20, 30, 99, 199)
        assertEquals(listOf(Box(20, 30, 99, 199)), InkSegmenter.segment(ink, 200, 260))
    }

    /** 宽高比 1.08 的单个数字（"0"形）不得被二次切分。用空心框而非实心块，否则会命中实心块过滤。 */
    @Test
    fun singleDigit_wideButNotSplit() {
        val ink = canvas(200, 160)
        frame(ink, 200, 10, 10, 149, 139, 10)
        assertEquals(listOf(Box(10, 10, 149, 139)), InkSegmenter.segment(ink, 200, 160))
    }

    /**
     * 宽高比 2.0 的单个空心数字（"0"被写扁）也不得被切分。
     *
     * 这条是回归用例：空心数字中段的列墨量只有上下两条笔画带，远低于左右壁的整字高，
     * 只看谷值比时会通过（40 vs 0.6*140），然后被当成"两个环 + 细颈"切成 3 个框。
     * 真正区分二者的判据是"切缝列只有一段墨"——空心数字的中段必有上下两段。
     */
    @Test
    fun singleDigit_wideHollow_notSplit() {
        val ink = canvas(300, 160)
        frame(ink, 300, 10, 10, 289, 149, 20)
        assertEquals(listOf(Box(10, 10, 289, 149)), InkSegmenter.segment(ink, 300, 160))
    }

    /** 同时验证：上下错位不影响切分（合并判据只看 X 交叠）。 */
    @Test
    fun twoSeparatedDigits_returnsTwoBoxesInOrder() {
        val ink = canvas(200, 160)
        rect(ink, 200, 10, 10, 39, 99)
        rect(ink, 200, 80, 20, 119, 139)
        assertEquals(
            listOf(Box(10, 10, 39, 99), Box(80, 20, 119, 139)),
            InkSegmenter.segment(ink, 200, 160)
        )
    }

    /** "5" 的横杠与下半弧上下分离但水平重叠，必须合并成一个数字。 */
    @Test
    fun twoStrokesOfOneDigit_merged() {
        val ink = canvas(200, 120)
        rect(ink, 200, 10, 0, 59, 9)
        rect(ink, 200, 10, 30, 59, 99)
        assertEquals(listOf(Box(10, 0, 59, 99)), InkSegmenter.segment(ink, 200, 120))
    }

    @Test
    fun partiallyOverlappingStrokes_merged() {
        val ink = canvas(200, 120)
        rect(ink, 200, 10, 0, 59, 9)
        rect(ink, 200, 20, 40, 69, 99)
        assertEquals(listOf(Box(10, 0, 69, 99)), InkSegmenter.segment(ink, 200, 120))
    }

    /** 锁死"不得过度合并"：紧挨着写的"11"必须是两个数字。 */
    @Test
    fun adjacentNarrowBars_notMerged() {
        val ink = canvas(200, 120)
        rect(ink, 200, 10, 10, 14, 109)
        rect(ink, 200, 16, 10, 20, 109)
        assertEquals(
            listOf(Box(10, 10, 14, 109), Box(16, 10, 20, 109)),
            InkSegmenter.segment(ink, 200, 120)
        )
    }

    /** 钉死 8 连通选型：4 连通会把这串对角像素碎成 10 个互不相连的域。 */
    @Test
    fun eightConnectivity_isUsed() {
        val ink = canvas(40, 40)
        for (i in 0..9) dot(ink, 40, i, i)
        val config = SegmentConfig(minComponentArea = 1)
        assertEquals(listOf(Box(0, 0, 9, 9)), InkSegmenter.segment(ink, 40, 40, config))
    }

    @Test
    fun tinySpeck_filtered_out() {
        val ink = canvas(200, 200)
        rect(ink, 200, 10, 10, 59, 99)
        rect(ink, 200, 150, 150, 152, 152)
        assertEquals(listOf(Box(10, 10, 59, 99)), InkSegmenter.segment(ink, 200, 200))
    }

    @Test
    fun onlySpeck_returnsNoBoxes() {
        val ink = canvas(200, 200)
        rect(ink, 200, 100, 100, 102, 102)
        assertTrue(InkSegmenter.segment(ink, 200, 200).isEmpty())
    }

    /** minHeightRatio 仅在存在多个组时生效，且不能误伤宽高比正常的矮笔画。 */
    @Test
    fun shortBlob_filtered_whenMultipleGroups() {
        val ink = canvas(200, 220)
        rect(ink, 200, 10, 10, 59, 199)
        rect(ink, 200, 150, 10, 162, 29)
        assertEquals(listOf(Box(10, 10, 59, 199)), InkSegmenter.segment(ink, 200, 220))
    }

    /** 只剩一个误触圆点：填充率高且接近正方形，判定为非数字。 */
    @Test
    fun solidDot_filtered_asBlob() {
        val ink = canvas(300, 300)
        disk(ink, 300, 150, 150, 32)
        assertTrue(InkSegmenter.segment(ink, 300, 300).isEmpty())
    }

    /**
     * 小号闭合环数字（"0"/"8"/"6"/"9"）被 64px 粗笔画挤小后，环孔变窄、填充率同样越过阈值，
     * 但它的包围盒内部仍有被墨迹包围的空腔，不能当作实心块删掉。
     * 旧实现只看填充率，这个用例会失败（表现为整组消失、静默漏一位）。
     */
    @Test
    fun smallClosedLoopDigit_notFilteredAsBlob() {
        val ink = canvas(200, 200)
        frame(ink, 200, 20, 20, 119, 119, 25)
        assertEquals(listOf(Box(20, 20, 119, 119)), InkSegmenter.segment(ink, 200, 200))
    }

    /**
     * 误触圆点内部一个 1 像素的空洞，不得让它"复活"成一位数字。
     *
     * 回归用例：若孔洞判据只看"有没有孔"（存在性）而不设面积下限，中心一个像素的采样空洞
     * 就会翻转结论，实心圆点被当成数字送进模型。
     */
    @Test
    fun solidDotWithPinhole_stillFiltered() {
        val ink = canvas(300, 300)
        disk(ink, 300, 150, 150, 32)
        ink[150 * 300 + 150] = 0
        assertTrue(InkSegmenter.segment(ink, 300, 300).isEmpty())
    }

    /**
     * 接触区与相邻数字的长横笔画墨量相同时，切点必须落在接触区，不能漂进那个数字内部。
     *
     * 回归用例：接触区（x 80..89）与"7"的长横（x 90..189）列墨量都是 20 且相邻，
     * 连成一个直到窗口右缘的长平台。若一律取平台中心（x≈107），会把长横一分为二，
     * 两个框都是垃圾。判据是"平台只有一侧有谷壁、另一侧被窗口截断 → 切在谷壁那一端"。
     */
    @Test
    fun neckPlateauContinuingIntoDigit_cutAtWallNotCenter() {
        val ink = canvas(260, 130)
        rect(ink, 260, 10, 10, 79, 109)
        rect(ink, 260, 80, 45, 89, 64)
        rect(ink, 260, 90, 45, 189, 64)
        rect(ink, 260, 170, 45, 189, 109)
        val boxes = InkSegmenter.segment(ink, 260, 130)
        assertEquals(2, boxes.size)
        // 长横笔画（x 90..189）不能被切开：除了它与竖腿相接的部分，切点必须在 90 之前
        assertTrue("切点应落在接触区 80..89，实际切在 ${boxes[0].right}", boxes[0].right in 80..89)
    }

    /** blob 过滤器不能误杀填充率同样是 1.0 的实心竖条"1"。 */
    @Test
    fun thickVerticalBar_notFilteredAsBlob() {
        val ink = canvas(400, 320)
        rect(ink, 400, 100, 10, 163, 309)
        assertEquals(listOf(Box(100, 10, 163, 309)), InkSegmenter.segment(ink, 400, 320))
    }

    /** 不规则形状（随机游走）再次验证"单数字输出即紧包围盒"。 */
    @Test
    fun singleDigit_irregularShape_pointsToSameBBox() {
        val width = 400
        val height = 400
        val ink = canvas(width, height)
        val random = java.util.Random(20260910L)
        var x = 100
        var y = 100
        dot(ink, width, x, y)
        for (i in 0 until 300) {
            if (random.nextBoolean()) x++ else y++
            dot(ink, width, x, y)
        }
        val expected = Box(100, 100, x, y)
        assertEquals(listOf(expected), InkSegmenter.segment(ink, width, height))
    }

    /**
     * 两个数字被一条细颈连在一起：守卫式颈部切分必须拆成两个。
     *
     * 细颈 x∈[25,129] 的墨量完全均匀，切缝只能是"最小墨量平台的中点"。
     * 几何中心 (10+144)/2 = 77 与平台中点重合，所以这个断言同时钉死了
     * "平台中点"这条规则——若改回"平台起点"会切在 51，从左环里割走一块。
     */
    @Test
    fun twoTouchingDigits_withNeck_areSplit() {
        val ink = canvas(300, 130)
        rect(ink, 300, 10, 10, 24, 109)
        rect(ink, 300, 130, 10, 144, 109)
        rect(ink, 300, 25, 55, 129, 59)
        val boxes = InkSegmenter.segment(ink, 300, 130)
        assertEquals(2, boxes.size)
        assertEquals(Box(10, 10, 77, 109), boxes[0])
        assertEquals(Box(78, 10, 144, 109), boxes[1])
    }

    /**
     * 三个数字连环粘连（两条细颈）时，切点必须落在细颈上，绝不能落到中间那个数字的实心笔画里。
     *
     * 回归用例：若对"首个最小值…末个最小值"直接取中点，窗口内两条细颈的墨量相同且各长 71px，
     * 中点是 x=207 —— 正好落在中间那根实心竖条（200..214）的中心，把一个数字劈成两半。
     */
    @Test
    fun threeDigitsJoinedByTwoNecks_neverSplitThroughSolidInk() {
        val ink = canvas(500, 130)
        rect(ink, 500, 10, 10, 24, 109)
        rect(ink, 500, 25, 55, 199, 59)
        rect(ink, 500, 200, 10, 214, 109)
        rect(ink, 500, 215, 55, 389, 59)
        rect(ink, 500, 390, 10, 404, 109)
        val boxes = InkSegmenter.segment(ink, 500, 130)
        for (box in boxes) {
            assertTrue(
                "包围盒边界 ${box.left}..${box.right} 落进了实心竖条 200..214 内部",
                box.left !in 201..213 && box.right !in 201..213
            )
        }
    }

    /**
     * 细颈（1 段墨）与空心数字中段（2 段墨）的列墨量恰好相等时，仍必须在细颈处切开。
     *
     * 回归用例：候选列的筛选必须先于"求最小墨量"。若先在整个窗口上取最小值再检查选中列的
     * 段数，平台会落在空心框的中段（2 段）上，整次切分被放弃；而窗口里其实存在合法的
     * 1 段切口（x 55..59）。固定笔宽下"细颈墨量 == 上下两条笔画带墨量"是常态而非巧合。
     */
    @Test
    fun neckInkTiesWithHollowDigitMiddle_stillSplit() {
        val ink = canvas(200, 130)
        rect(ink, 200, 10, 10, 49, 109)
        rect(ink, 200, 50, 45, 59, 54)
        frame(ink, 200, 60, 10, 159, 109, 5)
        val boxes = InkSegmenter.segment(ink, 200, 130)
        assertEquals(2, boxes.size)
        assertTrue("切点应落在细颈 50..59，实际切在 ${boxes[0].right}", boxes[0].right in 50..59)
    }

    /** 实心粘连且无谷值：宁可保留为一个框，也不盲目切开。 */
    @Test
    fun twoTouchingSolidDigits_noNeck_notSplit() {
        val ink = canvas(300, 120)
        rect(ink, 300, 10, 10, 84, 109)
        rect(ink, 300, 84, 10, 158, 109)
        assertEquals(listOf(Box(10, 10, 158, 109)), InkSegmenter.segment(ink, 300, 120))
    }

    @Test
    fun longThinBar_notSplit() {
        val ink = canvas(400, 40)
        rect(ink, 400, 0, 15, 399, 24)
        assertEquals(listOf(Box(0, 15, 399, 24)), InkSegmenter.segment(ink, 400, 40))
    }

    @Test
    fun neckSplit_disabledByConfig_stillOneBox() {
        val ink = canvas(300, 130)
        rect(ink, 300, 10, 10, 24, 109)
        rect(ink, 300, 130, 10, 144, 109)
        rect(ink, 300, 25, 55, 129, 59)
        val config = SegmentConfig(enableNeckSplit = false)
        assertEquals(listOf(Box(10, 10, 144, 109)), InkSegmenter.segment(ink, 300, 130, config))
    }

    @Test
    fun sorting_isLeftToRight() {
        val ink = canvas(200, 160)
        rect(ink, 200, 80, 20, 119, 139)
        rect(ink, 200, 10, 10, 39, 99)
        val boxes = InkSegmenter.segment(ink, 200, 160)
        assertEquals(listOf(Box(10, 10, 39, 99), Box(80, 20, 119, 139)), boxes)
    }

    @Test
    fun toInkGray_mapsWhiteToZeroBlackToMax() {
        val pixels = intArrayOf(
            0xFFFFFFFF.toInt(),
            0xFF000000.toInt(),
            0xFF808080.toInt()
        )
        assertEquals(listOf(0, 255, 127), InkSegmenter.toInkGray(pixels).toList())
    }

    private fun canvas(width: Int, height: Int) = IntArray(width * height)

    private fun rect(ink: IntArray, width: Int, left: Int, top: Int, right: Int, bottom: Int) {
        for (y in top..bottom) {
            for (x in left..right) {
                ink[y * width + x] = 255
            }
        }
    }

    /** 空心矩形框，用于模拟笔画围成的数字轮廓（填充率低，不会命中实心块过滤）。 */
    private fun frame(
        ink: IntArray,
        width: Int,
        left: Int,
        top: Int,
        right: Int,
        bottom: Int,
        thickness: Int
    ) {
        rect(ink, width, left, top, right, top + thickness - 1)
        rect(ink, width, left, bottom - thickness + 1, right, bottom)
        rect(ink, width, left, top, left + thickness - 1, bottom)
        rect(ink, width, right - thickness + 1, top, right, bottom)
    }

    private fun dot(ink: IntArray, width: Int, x: Int, y: Int) {
        ink[y * width + x] = 255
    }

    private fun disk(ink: IntArray, width: Int, centerX: Int, centerY: Int, radius: Int) {
        for (y in centerY - radius..centerY + radius) {
            for (x in centerX - radius..centerX + radius) {
                val dx = x - centerX
                val dy = y - centerY
                if (dx * dx + dy * dy <= radius * radius) {
                    ink[y * width + x] = 255
                }
            }
        }
    }
}
