package com.cashvision.app

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import android.content.Context
import android.graphics.Bitmap
import android.os.SystemClock
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.FloatBuffer
import kotlin.math.exp
import kotlin.math.sqrt

data class QGResult(
    val qualityScore: Float,
    val probGood: Float,
    val probUnder: Float,
    val probOver: Float,
    val predClass: String,
    val isPassed: Boolean,
    val hasNote: Boolean,
    val latencyMs: Double
)

data class YoloResult(
    val denomination: String?,
    val confidence: Float,
    val isTorn: Boolean,
    val tearCount: Int,
    val speechText: String,
    val latencyMs: Double
)

class OnnxInferenceEngine(private val context: Context) {

    private lateinit var ortEnv: OrtEnvironment
    private lateinit var sessionQG: OrtSession
    private lateinit var sessionMQTone: OrtSession
    private lateinit var sessionYOLO: OrtSession

    val classNames = listOf("10k", "100k", "20k", "200k", "50k", "500k", "torn")
    private val denomMap = mapOf(
        0 to "10.000",
        1 to "100.000",
        2 to "20.000",
        3 to "200.000",
        4 to "50.000",
        5 to "500.000"
    )

    var isReady: Boolean = false
        private set

    fun init() {
        ortEnv = OrtEnvironment.getEnvironment()
        val opts = OrtSession.SessionOptions().apply {
            setIntraOpNumThreads(2) // 2 threads optimal for mobile CPU
        }

        val qgBytes = context.assets.open("quality_gate.onnx").readBytes()
        val mqBytes = context.assets.open("mqtone.onnx").readBytes()
        val yoloBytes = context.assets.open("yolov8n_cashvision.onnx").readBytes()

        sessionQG = ortEnv.createSession(qgBytes, opts)
        sessionMQTone = ortEnv.createSession(mqBytes, opts)
        sessionYOLO = ortEnv.createSession(yoloBytes, opts)
        isReady = true
    }

    /**
     * 1. Run lightweight QualityGate on 64x64 thumbnail
     */
    fun inferQualityGate(thumb64: Bitmap): QGResult {
        val t0 = SystemClock.elapsedRealtimeNanos()

        val pixels = IntArray(64 * 64)
        thumb64.getPixels(pixels, 0, 64, 0, 0, 64, 64)

        // Compute pixel standard deviation to detect banknote presence in frame (has_note)
        var sum = 0.0
        var sumSq = 0.0
        val n = 64 * 64

        // Buffer RGB Planar [1, 3, 64, 64]
        val buffer = ByteBuffer.allocateDirect(1 * 3 * 64 * 64 * 4)
            .order(ByteOrder.nativeOrder())
            .asFloatBuffer()

        val rPlane = FloatArray(n)
        val gPlane = FloatArray(n)
        val bPlane = FloatArray(n)

        for (i in 0 until n) {
            val p = pixels[i]
            val r = (p shr 16 and 0xFF)
            val g = (p shr 8 and 0xFF)
            val b = (p and 0xFF)
            val gray = 0.299 * r + 0.587 * g + 0.114 * b

            sum += gray
            sumSq += gray * gray

            rPlane[i] = r / 255.0f
            gPlane[i] = g / 255.0f
            bPlane[i] = b / 255.0f
        }

        val mean = sum / n
        val variance = (sumSq / n) - (mean * mean)
        val stdDev = sqrt(variance.coerceAtLeast(0.0))
        val hasNote = stdDev > 15.0 // Banknote texture yields stdDev > 15

        buffer.put(rPlane)
        buffer.put(gPlane)
        buffer.put(bPlane)
        buffer.rewind()

        val inputTensor = OnnxTensor.createTensor(ortEnv, buffer, longArrayOf(1, 3, 64, 64))
        val output = sessionQG.run(mapOf("input" to inputTensor))
        val logits = (output[0].value as Array<FloatArray>)[0]

        // Softmax
        val exp0 = exp(logits[0].toDouble())
        val exp1 = exp(logits[1].toDouble())
        val exp2 = exp(logits[2].toDouble())
        val sumExp = exp0 + exp1 + exp2

        val pGood = (exp0 / sumExp).toFloat()
        val pUnder = (exp1 / sumExp).toFloat()
        val pOver = (exp2 / sumExp).toFloat()

        var predClass = "good"
        var maxProb = pGood
        if (pUnder > maxProb) { predClass = "underexposed"; maxProb = pUnder }
        if (pOver > maxProb) { predClass = "overexposed"; maxProb = pOver }

        val tau = 0.6f
        val isBlocked = (predClass != "good") && (maxProb > tau)
        val isPassed = !isBlocked
        val qualityScore = if (isPassed) pGood else (1.0f - maxProb)

        val latencyMs = (SystemClock.elapsedRealtimeNanos() - t0) / 1_000_000.0
        inputTensor.close()
        output.close()

        return QGResult(
            qualityScore = qualityScore,
            probGood = pGood,
            probUnder = pUnder,
            probOver = pOver,
            predClass = predClass,
            isPassed = isPassed,
            hasNote = hasNote,
            latencyMs = latencyMs
        )
    }

    /**
     * Create OnnxTensor [1, 3, 640, 640] from 640x640 Bitmap
     */
    fun createFrame640Tensor(bitmap640: Bitmap): OnnxTensor {
        val n = 640 * 640
        val pixels = IntArray(n)
        bitmap640.getPixels(pixels, 0, 640, 0, 0, 640, 640)

        val buffer = ByteBuffer.allocateDirect(1 * 3 * n * 4)
            .order(ByteOrder.nativeOrder())
            .asFloatBuffer()

        val rPlane = FloatArray(n)
        val gPlane = FloatArray(n)
        val bPlane = FloatArray(n)

        for (i in 0 until n) {
            val p = pixels[i]
            rPlane[i] = (p shr 16 and 0xFF) / 255.0f
            gPlane[i] = (p shr 8 and 0xFF) / 255.0f
            bPlane[i] = (p and 0xFF) / 255.0f
        }

        buffer.put(rPlane)
        buffer.put(gPlane)
        buffer.put(bPlane)
        buffer.rewind()

        return OnnxTensor.createTensor(ortEnv, buffer, longArrayOf(1, 3, 640, 640))
    }

    /**
     * 2. Run MQTone illumination enhancement (Input: 640x640 -> Output: Corrected Tensor 640x640)
     */
    fun inferMQTone(inputTensor: OnnxTensor): Pair<OnnxTensor, Double> {
        val t0 = SystemClock.elapsedRealtimeNanos()
        val result = sessionMQTone.run(mapOf("input" to inputTensor))
        val correctedTensor = result[0] as OnnxTensor
        val latencyMs = (SystemClock.elapsedRealtimeNanos() - t0) / 1_000_000.0
        return Pair(correctedTensor, latencyMs)
    }

    /**
     * 3. Run YOLOv8n (Input: Tensor 640x640 -> Output: Denomination + Torn status)
     */
    fun inferYOLO(inputTensor: OnnxTensor, confThresh: Float = 0.25f): YoloResult {
        val t0 = SystemClock.elapsedRealtimeNanos()
        val result = sessionYOLO.run(mapOf("images" to inputTensor))
        
        // Output shape [1, 84, 8400]
        val output = (result[0].value as Array<Array<FloatArray>>)[0]
        val numChannels = output.size      // 84 (4 bbox + 80 classes)
        val numAnchors = output[0].size    // 8400

        var bestDenomId: Int? = null
        var bestDenomConf = 0.0f
        var tearCount = 0

        for (i in 0 until numAnchors) {
            // Check 6 banknote denominations (classes 0..5)
            for (c in 0..5) {
                val score = output[4 + c][i]
                if (score > confThresh && score > bestDenomConf) {
                    bestDenomConf = score
                    bestDenomId = c
                }
            }
            // Check tear label (class 6: torn)
            val tornScore = output[4 + 6][i]
            if (tornScore > confThresh) {
                tearCount++
            }
        }

        val latencyMs = (SystemClock.elapsedRealtimeNanos() - t0) / 1_000_000.0
        result.close()

        val isTorn = tearCount > 0
        val denomStr = bestDenomId?.let { denomMap[it] }

        val speechText = if (denomStr != null) {
            if (isTorn) "$denomStr VND, torn detected"
            else "$denomStr VND, intact"
        } else {
            "Banknote denomination unrecognized"
        }

        return YoloResult(
            denomination = denomStr,
            confidence = bestDenomConf,
            isTorn = isTorn,
            tearCount = tearCount,
            speechText = speechText,
            latencyMs = latencyMs
        )
    }
}
