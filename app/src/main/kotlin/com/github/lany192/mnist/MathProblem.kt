package com.github.lany192.mnist

/**
 * 练习难度等级。
 *
 * **枚举顺序必须与 `R.array.grade_names` 完全一致** —— 界面用 Spinner 的 position 直接映射到
 * 这里（`Grade.values()[position]`），两处一旦错位，选"一年级"会出六年级的题，而且不会有任何
 * 编译期报错。标签本身刻意不放在这个枚举里，否则会和 strings.xml 形成两份真值来源。
 */
enum class Grade { FIRST, SECOND, THIRD, FOURTH, FIFTH, SIXTH }

/**
 * 一道口算题。
 *
 * @param expression 等号**左边**的题面，如 `"23 + 45"`、`"3.5 + 1.5"`、`"1/4 + 3/4"`、`"80 的 25%"`。
 *   刻意不含 `"= ?"`，由界面拼接 —— 这样单元测试可以直接把 expression 丢给独立求值器验算。
 * @param answer 标准答案。**只可能是 1..9999 的整数**：识别链路只能读 0-9 十个数字，
 *   答案若带小数点或分数线就无法判定；答案也不会是 0，因为画布上单独写一个 "0" 是闭合环，
 *   在部分切分路径上与"什么都没写"难以区分。
 */
data class Problem(val expression: String, val answer: Int)
