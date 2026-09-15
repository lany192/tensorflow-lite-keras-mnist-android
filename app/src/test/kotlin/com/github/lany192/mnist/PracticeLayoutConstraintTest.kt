package com.github.lany192.mnist

import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertTrue
import org.junit.Test
import org.w3c.dom.Document
import org.w3c.dom.Element
import org.w3c.dom.Node
import java.io.File
import javax.xml.parsers.DocumentBuilderFactory

/**
 * 把练习页布局里那条**画布铁律**变成一条会失败的测试。
 *
 * 规则本身是：画布占 `weight=1`，而 `FingerPaintView.onSizeChanged` 会重建 `drawingBitmap`，
 * 于是**任何改变画布高度的改动都会静默擦掉学生刚写的字迹** —— 学生的感受是"提交把答案吃掉了"。
 *
 * 结算区的错题回顾列表之所以安全，不是因为它是 RecyclerView，而是因为它满足两个结构条件：
 * 整体待在作答态 `GONE` 的 `groupResult` 里，且高度由 `0dp` + `weight=1` 定死。
 * 这两个条件一旦被破坏，它就变成又一个能挤动画布的兄弟控件，而**界面上不会有任何异常**。
 *
 * 直接解析 XML，不需要 Robolectric，也不需要 `android.*` —— 与 `PureKotlinBoundaryTest`
 * 用文件系统代替运行时的思路一致。
 */
class PracticeLayoutConstraintTest {

    @Test
    fun reviewList_staysInsideTheAlwaysHiddenResultGroup() {
        val root = document("activity_math_practice.xml").documentElement

        val group = findById(root, "groupResult")
        assertNotNull("groupResult 不见了", group)
        assertEquals(
            "groupResult 的默认可见性必须与 render() 对作答态的结果一致（作答态 GONE）",
            "gone",
            androidAttr(group!!, "visibility"),
        )

        val list = findById(root, "recyclerReview")
        assertNotNull("错题回顾列表不见了", list)
        assertTrue(
            "错题回顾必须留在 groupResult 内 —— 挪出去它就会参与画布的高度计算",
            isDescendantOf(list!!, group),
        )
    }

    @Test
    fun reviewList_hasAFixedWeightedHeight() {
        val list = findById(document("activity_math_practice.xml").documentElement, "recyclerReview")
        assertNotNull("错题回顾列表不见了", list)

        assertEquals(
            "必须写死 0dp：wrap_content 会让列表高度跟着行数变，从而撑开 groupResult",
            "0dp",
            androidAttr(list!!, "layout_height"),
        )
        assertEquals(
            "必须占 weight：weight 给的 EXACTLY 规格让 RecyclerView 不测量子项，高度与行数无关",
            "1",
            androidAttr(list, "layout_weight"),
        )
    }

    /** 单测的工作目录是 `app/`，与 `PureKotlinBoundaryTest` 用的是同一个前提。 */
    private fun document(name: String): Document {
        val file = File("src/main/res/layout/$name")
        assertTrue("找不到 ${file.path} —— 单元测试的工作目录应当是 app/", file.exists())
        return DocumentBuilderFactory.newInstance().newDocumentBuilder().parse(file)
    }

    /** 不带命名空间解析，所以属性名照写 `android:layout_height` 即可。 */
    private fun androidAttr(element: Element, name: String): String =
        element.getAttribute("android:$name")

    private fun findById(node: Node, id: String): Element? {
        if (node is Element && node.getAttribute("android:id") == "@+id/$id") return node
        val children = node.childNodes
        for (i in 0 until children.length) {
            findById(children.item(i), id)?.let { return it }
        }
        return null
    }

    private fun isDescendantOf(node: Node, ancestor: Node): Boolean {
        var parent = node.parentNode
        while (parent != null) {
            if (parent === ancestor) return true
            parent = parent.parentNode
        }
        return false
    }
}
