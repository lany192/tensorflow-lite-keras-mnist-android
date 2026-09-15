package com.github.lany192.mnist.data

import android.content.Context
import androidx.room.Database
import androidx.room.Room
import androidx.room.RoomDatabase
import com.github.lany192.mnist.practice.MathPracticeViewModel

/**
 * 练习记录的本地数据库。
 *
 * `exportSchema = true`（配合 `app/build.gradle.kts` 里的 `room.schemaLocation`）会在编译期
 * 导出 schema 快照。数据一旦写进用户设备就无法撤回，将来改版本时迁移写错的代价远高于维护
 * 这个目录，所以**不要**为了方便关掉它。
 */
@Database(
    entities = [PracticeSessionEntity::class, AnswerRecordEntity::class],
    version = 1,
    exportSchema = true,
)
abstract class PracticeDatabase : RoomDatabase() {

    abstract fun practiceDao(): PracticeDao

    companion object {
        private const val NAME = "practice.db"

        @Volatile
        private var instance: PracticeDatabase? = null

        /**
         * 双检锁单例。用 `applicationContext` 而不是调用方的 Context —— 这类单例的存活期
         * 与进程一致，持有 Activity 会直接泄漏。
         */
        fun get(context: Context): PracticeDatabase =
            instance ?: synchronized(this) {
                instance ?: Room.databaseBuilder(
                    context.applicationContext,
                    PracticeDatabase::class.java,
                    NAME,
                ).build().also { instance = it }
            }

        /** 读取入口。三个 Flow 由 View 层收集后灌进 ViewModel。 */
        fun repository(context: Context): PracticeRepository =
            RoomPracticeRepository(get(context).practiceDao())

        /** 写入入口。注入给 [MathPracticeViewModel]，在它的 `dispatch` 返回路径上被同步交付。 */
        fun recorder(context: Context): PracticeRecorder = RoomPracticeRepository(get(context).practiceDao())
    }
}
