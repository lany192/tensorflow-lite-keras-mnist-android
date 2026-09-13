package com.github.lany192.mnist

import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test
import java.io.File

/**
 * 把 AGENTS.md 里"本文件不得 import `android.*`"的约定变成一条会失败的测试。
 *
 * 这条约定在文档里反复出现，正说明它容易被踩 —— 靠人记不如靠测试。边界一旦被破坏，
 * 表现是"单元测试莫名其妙跑不起来"（`Stub!` / `RuntimeException: Method ... not mocked`），
 * 而不是编译错误，排查起来很绕。
 *
 * 只匹配 `import android.`（**带点**），不匹配 `androidx.`：`androidx.lifecycle.ViewModel`
 * 这类是纯 JVM 类，单测里可以正常使用。
 */
class PureKotlinBoundaryTest {

    @Test
    fun pureKotlinFiles_haveNoAndroidImports() {
        for (name in PURE_KOTLIN_FILES) {
            val file = File("src/main/kotlin/com/github/lany192/mnist/$name")
            assertTrue("找不到 $name —— 单元测试的工作目录应当是 app/", file.exists())
            file.readLines().forEachIndexed { index, line ->
                assertFalse(
                    "$name:${index + 1} 引入了 Android 依赖，JVM 单测会跑不起来：${line.trim()}",
                    line.trimStart().startsWith("import android.")
                )
            }
        }
    }

    private companion object {
        /**
         * 这个白名单**就是**那条边界线的定义：列在这里的文件必须能在 JVM 上直接跑。
         * 新增纯 Kotlin 类时记得加进来。
         */
        val PURE_KOTLIN_FILES = listOf(
            "InkSegmenter.kt",
            "MathProblem.kt",
            "MathProblemGenerator.kt",
            "DigitInput.kt",
            "MainContract.kt",
            "MainViewModel.kt",
            "MathPracticeContract.kt",
            "MathPracticeViewModel.kt",
            "PracticeRepository.kt",
            "HistorySummary.kt",
            "HistoryContract.kt",
            "HistoryViewModel.kt",
        )
    }
}
