package com.cashvision.app

import android.Manifest
import android.content.Context
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.os.Build
import android.os.Bundle
import android.os.CountDownTimer
import android.os.SystemClock
import android.os.VibrationEffect
import android.os.Vibrator
import android.os.VibratorManager
import android.speech.tts.TextToSpeech
import android.widget.Button
import android.widget.RadioButton
import android.widget.RadioGroup
import android.widget.TextView
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageProxy
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import java.io.File
import java.util.Locale
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors

class MainActivity : AppCompatActivity() {

    private lateinit var inferenceEngine: OnnxInferenceEngine
    private lateinit var batteryMeter: BatteryMeter
    private lateinit var sessionLogger: SessionLogger
    private lateinit var tts: TextToSpeech

    private val controller = CascadeController()
    private lateinit var cameraExecutor: ExecutorService

    // UI Views
    private lateinit var previewView: PreviewView
    private lateinit var tvStateBadge: TextView
    private lateinit var tvFps: TextView
    private lateinit var tvCalledFull: TextView
    private lateinit var tvLatencyBreakdown: TextView
    private lateinit var tvTotalLatency: TextView
    private lateinit var tvPower: TextView
    private lateinit var tvEnergy: TextView
    private lateinit var tvResult: TextView
    private lateinit var btnRecord: Button
    private lateinit var rgMode: RadioGroup

    private var currentMode = "cascade" // "cascade", "b0", "b1", "b2"
    private var isRecording = false
    private var sessionFrames = 0
    private var calledFullCount = 0
    private var recordingTimer: CountDownTimer? = null
    private var b1Triggered = false

    private var lastFpsTimestamp = SystemClock.elapsedRealtime()
    private var frameCounter = 0
    private var currentFps = 0.0

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        initViews()
        initServices()

        if (allPermissionsGranted()) {
            startCamera()
        } else {
            ActivityCompat.requestPermissions(
                this,
                arrayOf(Manifest.permission.CAMERA, Manifest.permission.VIBRATE),
                REQUEST_CODE_PERMISSIONS
            )
        }
    }

    private fun initViews() {
        previewView = findViewById(R.id.previewView)
        tvStateBadge = findViewById(R.id.tvStateBadge)
        tvFps = findViewById(R.id.tvFps)
        tvCalledFull = findViewById(R.id.tvCalledFull)
        tvLatencyBreakdown = findViewById(R.id.tvLatencyBreakdown)
        tvTotalLatency = findViewById(R.id.tvTotalLatency)
        tvPower = findViewById(R.id.tvPower)
        tvEnergy = findViewById(R.id.tvEnergy)
        tvResult = findViewById(R.id.tvResult)
        btnRecord = findViewById(R.id.btnRecord)
        rgMode = findViewById(R.id.rgMode)

        rgMode.setOnCheckedChangeListener { _, checkedId ->
            currentMode = when (checkedId) {
                R.id.rbB0 -> "b0"
                R.id.rbB1 -> "b1"
                else -> "cascade"
            }
            controller.reset()
            b1Triggered = false
            calledFullCount = 0
            sessionFrames = 0
            tvStateBadge.text = "MODE: ${currentMode.uppercase()}"
        }

        btnRecord.setOnClickListener {
            if (!isRecording) {
                startBenchmarkSession()
            } else {
                stopBenchmarkSession()
            }
        }
    }

    private fun initServices() {
        cameraExecutor = Executors.newSingleThreadExecutor()
        batteryMeter = BatteryMeter(this)
        sessionLogger = SessionLogger(this)

        inferenceEngine = OnnxInferenceEngine(this)
        Thread {
            try {
                inferenceEngine.init()
                runOnUiThread {
                    Toast.makeText(this, "✅ Successfully loaded 3 ONNX models!", Toast.LENGTH_SHORT).show()
                }
            } catch (e: Exception) {
                e.printStackTrace()
                runOnUiThread {
                    Toast.makeText(this, "❌ Model loading error: ${e.message}", Toast.LENGTH_LONG).show()
                }
            }
        }.start()

        tts = TextToSpeech(this) { status ->
            if (status == TextToSpeech.SUCCESS) {
                tts.language = Locale("vi", "VN")
            }
        }
    }

    private fun startCamera() {
        val cameraProviderFuture = ProcessCameraProvider.getInstance(this)
        cameraProviderFuture.addListener({
            val cameraProvider = cameraProviderFuture.get()
            val preview = androidx.camera.core.Preview.Builder().build().also {
                it.setSurfaceProvider(previewView.surfaceProvider)
            }

            val imageAnalyzer = ImageAnalysis.Builder()
                .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                .build()
                .also {
                    it.setAnalyzer(cameraExecutor) { imageProxy ->
                        processFrame(imageProxy)
                    }
                }

            cameraProvider.unbindAll()
            cameraProvider.bindToLifecycle(
                this,
                CameraSelector.DEFAULT_BACK_CAMERA,
                preview,
                imageAnalyzer
            )
        }, ContextCompat.getMainExecutor(this))
    }

    private fun processFrame(imageProxy: ImageProxy) {
        if (!inferenceEngine.isReady) {
            imageProxy.close()
            return
        }

        try {
            val t0 = SystemClock.elapsedRealtimeNanos()
            val bitmap = imageProxy.toBitmap()
            imageProxy.close()

            // 1. Lightweight QualityGate evaluated on 64x64 thumbnail
            val thumb64 = Bitmap.createScaledBitmap(bitmap, 64, 64, true)
            val qgResult = inferenceEngine.inferQualityGate(thumb64)

        // 2. Controller decision per Mode
        val shouldRunFull = when (currentMode) {
            "b0" -> true // B0: Run full pipeline every frame
            "b1" -> {
                // B1: Single-shot (infer once when banknote detected)
                if (!qgResult.hasNote) {
                    b1Triggered = false
                    false
                } else if (!b1Triggered) {
                    b1Triggered = true
                    true
                } else {
                    false
                }
            }
            else -> controller.decide(qgResult.hasNote, qgResult.qualityScore) // Cascade
        }

        var tMqMs = 0.0
        var tYoloMs = 0.0
        var denomName: String? = null
        var isTorn = false
        var yoloConf = 0.0f
        var speech = ""

        if (shouldRunFull) {
            calledFullCount++
            val frame640 = Bitmap.createScaledBitmap(bitmap, 640, 640, true)
            val raw640Tensor = inferenceEngine.createFrame640Tensor(frame640)

            // Run MQTone illumination enhancement
            val (correctedTensor, mqMs) = inferenceEngine.inferMQTone(raw640Tensor)
            tMqMs = mqMs

            // Run YOLO on enhanced image
            val yResult = inferenceEngine.inferYOLO(correctedTensor)
            tYoloMs = yResult.latencyMs
            denomName = yResult.denomination
            isTorn = yResult.isTorn
            yoloConf = yResult.confidence
            speech = yResult.speechText

            correctedTensor.close()
            raw640Tensor.close()

            // Broadcast result
            if (denomName != null) {
                if (currentMode == "cascade") {
                    controller.onDetectionResult(true)
                    if (controller.state == CascadeState.CONFIRMED) {
                        triggerHapticFeedback(isTorn)
                        tts.speak(speech, TextToSpeech.QUEUE_FLUSH, null, null)
                    }
                } else if (currentMode == "b1") {
                    triggerHapticFeedback(isTorn)
                    tts.speak(speech, TextToSpeech.QUEUE_FLUSH, null, null)
                }
            } else if (currentMode == "cascade") {
                controller.onDetectionResult(false)
            }
        }

        val totalMs = (SystemClock.elapsedRealtimeNanos() - t0) / 1_000_000.0

        // 3. Energy & battery profiling
        val (currentMa, powerMw, cumJoules) = batteryMeter.sample()

        // Compute FPS
        frameCounter++
        val now = SystemClock.elapsedRealtime()
        if (now - lastFpsTimestamp >= 1000) {
            currentFps = frameCounter * 1000.0 / (now - lastFpsTimestamp)
            frameCounter = 0
            lastFpsTimestamp = now
        }

        // Record log if benchmark session is active
        if (isRecording) {
            sessionFrames++
            val log = FrameLogRecord(
                timestamp_ms = System.currentTimeMillis(),
                frame_index = sessionFrames,
                mode = currentMode,
                state = controller.state.name,
                has_note = qgResult.hasNote,
                quality_score = qgResult.qualityScore,
                called_full = shouldRunFull,
                latency_light_ms = qgResult.latencyMs,
                latency_mq_ms = tMqMs,
                latency_yolo_ms = tYoloMs,
                latency_total_ms = totalMs,
                battery_current_ma = currentMa,
                battery_power_mw = powerMw,
                cumulative_joules = cumJoules,
                denomination = denomName,
                is_torn = isTorn,
                confidence = yoloConf
            )
            sessionLogger.logFrame(log)
        }

        // Update Telemetry HUD interface
        runOnUiThread {
            updateHUD(
                qgResult, totalMs, tMqMs, tYoloMs, currentMa, powerMw, cumJoules, denomName, isTorn
            )
        }
        } catch (e: Exception) {
            e.printStackTrace()
        }
    }

    private fun updateHUD(
        qg: QGResult, totalMs: Double, mqMs: Double, yoloMs: Double,
        currentMa: Double, powerMw: Double, joules: Double,
        denom: String?, isTorn: Boolean
    ) {
        tvStateBadge.text = if (currentMode == "cascade") "STATE: ${controller.state.name}" else "MODE: ${currentMode.uppercase()}"
        tvFps.text = String.format("Speed: %.1f FPS", currentFps)

        val totalF = if (isRecording) sessionFrames else (sessionFrames + 1)
        val pct = if (totalF > 0) (calledFullCount.toDouble() / totalF * 100.0) else 0.0
        tvCalledFull.text = String.format("Full Calls: %d / %d (%.1f%%)", calledFullCount, totalF, pct)

        tvLatencyBreakdown.text = String.format("Light: %.1fms | MQ: %.1fms | YOLO: %.1fms", qg.latencyMs, mqMs, yoloMs)
        tvTotalLatency.text = String.format("Total: %.1f ms", totalMs)

        tvPower.text = String.format("Pin: %.0f mA | %.0f mW", currentMa, powerMw)
        tvEnergy.text = String.format("Energy: %.2f J", joules)

        if (denom != null) {
            val tornTag = if (isTorn) "⚠️ TORN" else "INTACT"
            tvResult.text = "Result: $denom VND ($tornTag)"
        } else {
            tvResult.text = "Result: Scanning... (Q: ${(qg.qualityScore * 100).toInt()}%)"
        }
    }

    private fun triggerHapticFeedback(isTorn: Boolean) {
        val vibrator = if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.S) {
            val vibratorManager = getSystemService(Context.VIBRATOR_MANAGER_SERVICE) as VibratorManager
            vibratorManager.defaultVibrator
        } else {
            @Suppress("DEPRECATION")
            getSystemService(Context.VIBRATOR_SERVICE) as Vibrator
        }

        if (isTorn) {
            // Torn: Double vibration (100ms buzz - 80ms pause - 150ms buzz)
            vibrator.vibrate(VibrationEffect.createWaveform(longArrayOf(0, 100, 80, 150), -1))
        } else {
            // Intact: Single vibration (150ms)
            vibrator.vibrate(VibrationEffect.createOneShot(150, VibrationEffect.DEFAULT_AMPLITUDE))
        }
    }

    private fun startBenchmarkSession() {
        isRecording = true
        sessionFrames = 0
        calledFullCount = 0
        b1Triggered = false
        controller.reset()
        batteryMeter.reset()

        val logFile = sessionLogger.startSession(currentMode)
        btnRecord.text = "⏳ STOP BENCHMARK (30S REMAINING)"
        btnRecord.setBackgroundColor(ContextCompat.getColor(this, R.color.accent_red))

        recordingTimer = object : CountDownTimer(30000, 1000) {
            override fun onTick(millisUntilFinished: Long) {
                btnRecord.text = "⏳ STOP BENCHMARK (${millisUntilFinished / 1000}S REMAINING)"
            }

            override fun onFinish() {
                stopBenchmarkSession()
                Toast.makeText(
                    this@MainActivity,
                    "🎉 30s benchmark finished!\nSaved: ${logFile.name}",
                    Toast.LENGTH_LONG
                ).show()
                tts.speak("30-second benchmark session completed", TextToSpeech.QUEUE_FLUSH, null, null)
            }
        }.start()
    }

    private fun stopBenchmarkSession() {
        recordingTimer?.cancel()
        recordingTimer = null
        isRecording = false
        sessionLogger.endSession()

        btnRecord.text = getString(R.string.start_bench)
        btnRecord.setBackgroundColor(ContextCompat.getColor(this, R.color.accent_blue))
    }

    private fun allPermissionsGranted() = arrayOf(Manifest.permission.CAMERA, Manifest.permission.VIBRATE).all {
        ContextCompat.checkSelfPermission(baseContext, it) == PackageManager.PERMISSION_GRANTED
    }

    override fun onRequestPermissionsResult(
        requestCode: Int, permissions: Array<String>, grantResults: IntArray
    ) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults)
        if (requestCode == REQUEST_CODE_PERMISSIONS) {
            if (allPermissionsGranted()) {
                startCamera()
            } else {
                Toast.makeText(this, "Camera permission is required for this application!", Toast.LENGTH_SHORT).show()
                finish()
            }
        }
    }

    override fun onDestroy() {
        super.onDestroy()
        cameraExecutor.shutdown()
        tts.stop()
        tts.shutdown()
        sessionLogger.endSession()
    }

    companion object {
        private const val REQUEST_CODE_PERMISSIONS = 10
    }
}
