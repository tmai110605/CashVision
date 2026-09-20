package com.cashvision.app

import android.content.Context
import com.google.gson.Gson
import java.io.File
import java.io.FileWriter
import java.io.PrintWriter
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale

data class FrameLogRecord(
    val timestamp_ms: Long,
    val frame_index: Int,
    val mode: String,
    val state: String,
    val has_note: Boolean,
    val quality_score: Float,
    val called_full: Boolean,
    val latency_light_ms: Double,
    val latency_mq_ms: Double,
    val latency_yolo_ms: Double,
    val latency_total_ms: Double,
    val battery_current_ma: Double,
    val battery_power_mw: Double,
    val cumulative_joules: Double,
    val denomination: String?,
    val is_torn: Boolean,
    val confidence: Float
)

class SessionLogger(private val context: Context) {

    private val gson = Gson()
    private var writer: PrintWriter? = null
    var activeLogFile: File? = null
        private set

    fun startSession(mode: String): File {
        val timeStamp = SimpleDateFormat("yyyyMMdd_HHmmss", Locale.US).format(Date())
        val logDir = context.getExternalFilesDir(null) ?: context.filesDir
        val file = File(logDir, "session_${mode}_$timeStamp.jsonl")
        activeLogFile = file
        writer = PrintWriter(FileWriter(file, true))
        return file
    }

    fun logFrame(record: FrameLogRecord) {
        val json = gson.toJson(record)
        writer?.println(json)
        writer?.flush()
    }

    fun endSession() {
        writer?.flush()
        writer?.close()
        writer = null
    }
}
