package com.github.lany192.mnist

import java.io.File
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

/**
 * 把 AGENTS.md 里"本文件不得 import `android.*`"的约定变成一条会失败的测试。
 *
 * 这条约定在文档里反复出现，正说明它容易被踩 —— 靠人记不如靠测试。边界一旦被破坏，
 * 表现是"单元测试莫名其妙跑不起来"（`Stub!` / `RuntimeException: Method ... not mocked`），
 * 而不是编译错误，排查起来很绕。
 *
 * 只匹配 `import android.`（**带点**），不匹配 `androidx.`：`androidx.lifecycle.ViewModel`
 * 这类是纯 JVM 类，单测里可以正常使用。
 *
 * 源码按功能模块分了子包（`recognize` / `practice` / `history` / `data`）之后，「哪些文件是纯 Kotlin」
 * 不再能从包名上看出来，这两个清单成了唯一的真相来源 —— 所以 [everySourceFile_isClassified]
 * 会强制每个新文件都必须被明确归类，而不是默默漏掉。
 */
class PureKotlinBoundaryTest {

    @Test
    fun pureKotlinFiles_haveNoAndroidImports() {
        for (relativePath in PURE_KOTLIN_FILES) {
            val file = sourceFile(relativePath)
            file.readLines().forEachIndexed { index, line ->
                assertFalse(
                    "$relativePath:${index + 1} 引入了 Android 依赖，JVM 单测会跑不起来：${line.trim()}",
                    line.trimStart().startsWith("import android.")
                )
            }
        }
    }

    /**
     * 分类完整性：`src/main/kotlin` 下每个 `.kt` 都必须出现在 [PURE_KOTLIN_FILES] 或
     * [ANDROID_DEPENDENT_FILES] 里。
     *
     * 这一条防的是**静默失效**：新增一个纯 Kotlin 类却忘了加进白名单，边界测试不会报错、
     * 也不会覆盖它，而这类文件恰恰是最该被 JVM 单测覆盖的。现在忘了分类会让 `./gradlew test` 直接失败。
     */
    @Test
    fun everySourceFile_isClassified() {
        val onDisk = File(SOURCE_ROOT).walkTopDown()
            .filter { it.isFile && it.extension == "kt" }
            .map { it.relativeTo(File(SOURCE_ROOT)).invariantSeparatorsPath }
            .toSortedSet()

        val classified = (PURE_KOTLIN_FILES + ANDROID_DEPENDENT_FILES).toSortedSet()

        assertEquals(
            "有源文件没有被归类 —— 纯 Kotlin 的加进 PURE_KOTLIN_FILES，依赖 Android 的加进 ANDROID_DEPENDENT_FILES",
            emptySet<String>(),
            onDisk - classified,
        )
        assertEquals("清单里有已经不存在的文件", emptySet<String>(), classified - onDisk)
    }

    private fun sourceFile(relativePath: String): File {
        val file = File("$SOURCE_ROOT/$relativePath")
        assertTrue("找不到 $relativePath —— 单元测试的工作目录应当是 app/", file.exists())
        return file
    }

    private companion object {
        const val SOURCE_ROOT = "src/main/kotlin/com/github/lany192/mnist"

        /**
         * 这个清单**就是**那条边界线的定义：列在这里的文件必须能在 JVM 上直接跑。
         *
         * `PracticeEntities.kt` 只依赖 `androidx.room3` 与协程，按同一把尺子（只禁 `import android.`）
         * 它也是纯 Kotlin，所以归在这一侧。
         */
        val PURE_KOTLIN_FILES = listOf(
            "recognize/DigitInput.kt",
            "recognize/InkSegmenter.kt",
            "recognize/MainContract.kt",
            "recognize/MainViewModel.kt",
            "practice/MathProblem.kt",
            "practice/MathProblemGenerator.kt",
            "practice/MathPracticeContract.kt",
            "practice/MathPracticeViewModel.kt",
            "history/HistorySummary.kt",
            "history/HistoryContract.kt",
            "history/HistoryRows.kt",
            "history/HistoryViewModel.kt",
            "data/PracticeRepository.kt",
            "data/PracticeEntities.kt",
        )

        /** 依赖 Android 运行时、只能在仪器测试或真机上验的文件。 */
        val ANDROID_DEPENDENT_FILES = listOf(
            "recognize/MainActivity.kt",
            "recognize/FingerPaintView.kt",
            "recognize/KerasTFLite.kt",
            "recognize/MnistPreprocessor.kt",
            "recognize/MnistRecognizer.kt",
            "practice/MathPracticeActivity.kt",
            "practice/ReviewAdapter.kt",
            "history/HistoryActivity.kt",
            "history/HistoryAdapter.kt",
            "data/PracticeDatabase.kt",
            "data/RoomPracticeRepository.kt",
        )
    }
}
