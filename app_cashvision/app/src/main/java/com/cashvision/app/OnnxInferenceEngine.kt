package com.cashvision.app

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import android.content.Context
import android.graphics.Bitmap
import android.graphics.RectF
import android.os.SystemClock
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.FloatBuffer
import kotlin.math.exp
import kotlin.math.max
import kotlin.math.min
import kotlin.math.sqrt

data class QGResult(
    val qualityScore: Float,
    val probGood: Float,
    val probUnder: Float,
    val probOver: Float,
    val predClass: String, // "good", "underexposed", "overexposed"
    val isPassed: Boolean,
    val hasNote: Boolean,
    val meanLuminance: Double,
    val stdDev: Double,
    val latencyMs: Double
)

data class YoloResult(
    val denomination: String?,
    val denomIndex: Int?,
    val confidence: Float,
    val scores: FloatArray, // 6 denomination scores
    val noteBox: RectF?,
    val isTorn: Boolean,
    val tornConfidence: Float,
    val tearCount: Int,
    val tearBoxes: List<RectF>,
    val speechText: String,
    val latencyMs: Double
)

class OnnxInferenceEngine(private val context: Context) {

    private lateinit var ortEnv: OrtEnvironment
    private lateinit var sessionQG: OrtSession
    private lateinit var sessionMQTone: OrtSession
    private lateinit var sessionYOLO: OrtSession

    val classNames = listOf("10k", "100k", "20k", "200k", "50k", "500k", "torn")
    val denomMap = mapOf(
        0 to "10,000",
        1 to "100,000",
        2 to "20,000",
        3 to "200,000",
        4 to "50,000",
        5 to "500,000"
    )

    private val speechMap = mapOf(
        0 to "10 thousand Dong",
        1 to "100 thousand Dong",
        2 to "20 thousand Dong",
        3 to "200 thousand Dong",
        4 to "50 thousand Dong",
        5 to "500 thousand Dong"
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
     * 1. Run lightweight QualityGate on 64x64 thumbnail (< 1.7 ms on mobile CPU)
     * Performs spatial texture variance gating (R1) & photometric readiness (R2)
     */
    fun inferQualityGate(thumb64: Bitmap): QGResult {
        val t0 = SystemClock.elapsedRealtimeNanos()

        val pixels = IntArray(64 * 64)
        thumb64.getPixels(pixels, 0, 64, 0, 0, 64, 64)

        // Compute pixel standard deviation to detect banknote presence in frame (R1)
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
        val hasNote = stdDev > 15.0 // Rule R1: Banknote texture yields stdDev > 15.0

        buffer.put(rPlane)
        buffer.put(gPlane)
        buffer.put(bPlane)
        buffer.rewind()

        val inputTensor = OnnxTensor.createTensor(ortEnv, buffer, longArrayOf(1, 3, 64, 64))
        val output = sessionQG.run(mapOf("input" to inputTensor))
        val logits = (output[0].value as Array<FloatArray>)[0]

        // Softmax probabilities
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

        // Rule R2: Photometric readiness threshold tau = 0.60
        val tau = 0.60f
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
            meanLuminance = mean,
            stdDev = stdDev,
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
     * 3. Run dual-task YOLOv8n (Input: Tensor 640x640 -> Output: Denomination scores, bounding boxes, tear defects)
     * Rule R5: filters detections < confThresh (default theta_conf = 0.25)
     */
    fun inferYOLO(inputTensor: OnnxTensor, confThresh: Float = 0.25f): YoloResult {
        val t0 = SystemClock.elapsedRealtimeNanos()
        val result = sessionYOLO.run(mapOf("images" to inputTensor))

        // Output shape [1, 84, 8400] or [1, 11, 8400]
        val output = (result[0].value as Array<Array<FloatArray>>)[0]
        val numAnchors = output[0].size // 8400

        val maxScores = FloatArray(6) { 0.0f }
        var bestAnchor = -1
        var bestDenomId: Int? = null
        var bestDenomConf = 0.0f

        var maxTornConf = 0.0f
        val rawTearBoxes = mutableListOf<RectF>()
        var tearCount = 0

        for (i in 0 until numAnchors) {
            // Check 6 banknote denominations (classes 0..5)
            for (c in 0..5) {
                val score = output[4 + c][i]
                if (score > maxScores[c]) {
                    maxScores[c] = score
                }
                if (score > confThresh && score > bestDenomConf) {
                    bestDenomConf = score
                    bestDenomId = c
                    bestAnchor = i
                }
            }

            // Check tear label (class 6: torn)
            val tornScore = output[4 + 6][i]
            if (tornScore > maxTornConf) {
                maxTornConf = tornScore
            }
            if (tornScore > confThresh) {
                tearCount++
                if (rawTearBoxes.size < 5) {
                    val cx = output[0][i]
                    val cy = output[1][i]
                    val w = output[2][i]
                    val h = output[3][i]
                    val left = max(0f, (cx - w / 2f) / 640f)
                    val top = max(0f, (cy - h / 2f) / 640f)
                    val right = min(1f, (cx + w / 2f) / 640f)
                    val bottom = min(1f, (cy + h / 2f) / 640f)
                    rawTearBoxes.add(RectF(left, top, right, bottom))
                }
            }
        }

        // Compute normalized note bounding box
        var noteBox: RectF? = null
        if (bestAnchor >= 0) {
            val cx = output[0][bestAnchor]
            val cy = output[1][bestAnchor]
            val w = output[2][bestAnchor]
            val h = output[3][bestAnchor]
            val left = max(0f, (cx - w / 2f) / 640f)
            val top = max(0f, (cy - h / 2f) / 640f)
            val right = min(1f, (cx + w / 2f) / 640f)
            val bottom = min(1f, (cy + h / 2f) / 640f)
            noteBox = RectF(left, top, right, bottom)
        }

        val latencyMs = (SystemClock.elapsedRealtimeNanos() - t0) / 1_000_000.0
        result.close()

        val isTorn = tearCount > 0 && maxTornConf >= confThresh
        val denomStr = bestDenomId?.let { denomMap[it] }

        val speechText = if (bestDenomId != null) {
            val denomText = speechMap[bestDenomId] ?: "$denomStr Dong"
            if (isTorn) "$denomText, warning: torn defect detected!"
            else "$denomText, banknote intact."
        } else {
            "Banknote unrecognized."
        }

        return YoloResult(
            denomination = denomStr,
            denomIndex = bestDenomId,
            confidence = bestDenomConf,
            scores = maxScores,
            noteBox = noteBox,
            isTorn = isTorn,
            tornConfidence = maxTornConf,
            tearCount = tearCount,
            tearBoxes = rawTearBoxes,
            speechText = speechText,
            latencyMs = latencyMs
        )
    }
}
