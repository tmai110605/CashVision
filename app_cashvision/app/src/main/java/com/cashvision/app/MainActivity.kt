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
import android.view.View
import android.widget.Button
import android.widget.ImageButton
import android.widget.LinearLayout
import android.widget.TextView
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import androidx.camera.core.Camera
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
    private var camera: Camera? = null

    // UI Views
    private lateinit var previewView: PreviewView
    private lateinit var overlayView: OverlayView
    private lateinit var tvSubtitle: TextView
    private lateinit var btnTorch: ImageButton
    private lateinit var btnMute: ImageButton
    private lateinit var btnTelemetryToggle: ImageButton

    private lateinit var tvStateBadge: TextView
    private lateinit var tvOpticalGuidance: TextView

    private lateinit var tvDefectStatus: TextView
    private lateinit var tvHeroDenomination: TextView
    private lateinit var tvVerificationNote: TextView
    private lateinit var btnRepeatTts: Button
    private lateinit var btnScanNew: Button

    private lateinit var telemetryCard: LinearLayout
    private lateinit var tvFps: TextView
    private lateinit var chipR1: TextView
    private lateinit var chipR2: TextView
    private lateinit var chipR3: TextView
    private lateinit var chipR4: TextView
    private lateinit var chipR6: TextView
    private lateinit var chipR7: TextView
    private lateinit var chipR8: TextView
    private lateinit var tvLatencyBreakdown: TextView
    private lateinit var tvTotalLatency: TextView
    private lateinit var tvPower: TextView
    private lateinit var tvEnergy: TextView
    private lateinit var btnRecord: Button

    // App Control States
    private var isTorchOn = false
    private var isMuted = false
    private var isTelemetryVisible = false

    // Session Benchmark Recorder
    private var isRecording = false
    private var sessionFrames = 0
    private var calledFullCount = 0
    private var recordingTimer: CountDownTimer? = null

    // FPS Telemetry
    private var lastFpsTimestamp = SystemClock.elapsedRealtime()
    private var frameCounter = 0
    private var currentFps = 20.5

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
        overlayView = findViewById(R.id.overlayView)
        tvSubtitle = findViewById(R.id.tvSubtitle)
        btnTorch = findViewById(R.id.btnTorch)
        btnMute = findViewById(R.id.btnMute)
        btnTelemetryToggle = findViewById(R.id.btnTelemetryToggle)

        tvStateBadge = findViewById(R.id.tvStateBadge)
        tvOpticalGuidance = findViewById(R.id.tvOpticalGuidance)

        tvDefectStatus = findViewById(R.id.tvDefectStatus)
        tvHeroDenomination = findViewById(R.id.tvHeroDenomination)
        tvVerificationNote = findViewById(R.id.tvVerificationNote)
        btnRepeatTts = findViewById(R.id.btnRepeatTts)
        btnScanNew = findViewById(R.id.btnScanNew)

        telemetryCard = findViewById(R.id.telemetryCard)
        tvFps = findViewById(R.id.tvFps)
        chipR1 = findViewById(R.id.chipR1)
        chipR2 = findViewById(R.id.chipR2)
        chipR3 = findViewById(R.id.chipR3)
        chipR4 = findViewById(R.id.chipR4)
        chipR6 = findViewById(R.id.chipR6)
        chipR7 = findViewById(R.id.chipR7)
        chipR8 = findViewById(R.id.chipR8)
        tvLatencyBreakdown = findViewById(R.id.tvLatencyBreakdown)
        tvTotalLatency = findViewById(R.id.tvTotalLatency)
        tvPower = findViewById(R.id.tvPower)
        tvEnergy = findViewById(R.id.tvEnergy)
        btnRecord = findViewById(R.id.btnRecord)

        // Torch Toggle
        btnTorch.setOnClickListener {
            camera?.let { cam ->
                if (cam.cameraInfo.hasFlashUnit()) {
                    isTorchOn = !isTorchOn
                    cam.cameraControl.enableTorch(isTorchOn)
                    btnTorch.setColorFilter(
                        if (isTorchOn) ContextCompat.getColor(this, R.color.accent_gold)
                        else ContextCompat.getColor(this, R.color.white)
                    )
                } else {
                    Toast.makeText(this, "Torch is not available on this device", Toast.LENGTH_SHORT).show()
                }
            }
        }

        // Speech Audio Mute Toggle
        btnMute.setOnClickListener {
            isMuted = !isMuted
            btnMute.setImageResource(
                if (isMuted) android.R.drawable.ic_lock_silent_mode
                else android.R.drawable.ic_lock_silent_mode_off
            )
            btnMute.setColorFilter(
                if (isMuted) ContextCompat.getColor(this, R.color.accent_red)
                else ContextCompat.getColor(this, R.color.accent_cyan)
            )
            val msg = if (isMuted) "Voice speech muted" else "Voice speech unmuted"
            Toast.makeText(this, msg, Toast.LENGTH_SHORT).show()
        }

        // Diagnostics Drawer Toggle
        btnTelemetryToggle.setOnClickListener {
            isTelemetryVisible = !isTelemetryVisible
            telemetryCard.visibility = if (isTelemetryVisible) View.VISIBLE else View.GONE
            btnTelemetryToggle.setColorFilter(
                if (isTelemetryVisible) ContextCompat.getColor(this, R.color.accent_gold)
                else ContextCompat.getColor(this, R.color.accent_green)
            )
        }

        // Re-read result via TTS
        btnRepeatTts.setOnClickListener {
            if (controller.state == CascadeState.CONFIRMED && controller.latchedDenom != null) {
                val statusStr = if (controller.latchedIsTorn) "warning: torn defect detected!" else "banknote intact."
                val speech = "${controller.latchedDenom} Dong, $statusStr"
                speakDirect(speech)
                triggerHapticFeedback(controller.latchedIsTorn)
            } else {
                speakDirect("Scanning for banknote...")
            }
        }

        // Unlock and Scan New Banknote
        btnScanNew.setOnClickListener {
            controller.reset()
            overlayView.state = CascadeState.SEARCHING
            overlayView.noteBox = null
            overlayView.tearBoxes = emptyList()
            overlayView.denomination = null
            updateSearchingUI()
            speakDirect("Ready to scan new banknote")
        }

        // 30s Benchmark Session Trigger
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
                    Toast.makeText(this, "✅ CashVision AI Engine Ready!", Toast.LENGTH_SHORT).show()
                }
            } catch (e: Exception) {
                e.printStackTrace()
                runOnUiThread {
                    Toast.makeText(this, "❌ Model initialization error: ${e.message}", Toast.LENGTH_LONG).show()
                }
            }
        }.start()

        tts = TextToSpeech(this) { status ->
            if (status == TextToSpeech.SUCCESS) {
                tts.language = Locale.US
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
            camera = cameraProvider.bindToLifecycle(
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

            // 1. Lightweight Tier-1 Texture Gating (R1) & Tier-2 Photometric CNN (R2) on 64x64 thumbnail
            val thumb64 = Bitmap.createScaledBitmap(bitmap, 64, 64, true)
            val qgResult = inferenceEngine.inferQualityGate(thumb64)

            // 2. Meta-Level Control: Energy-aware conditional execution arbitration
            val shouldRunFull = controller.decide(
                hasNote = qgResult.hasNote,
                quality = qgResult.qualityScore,
                predClass = qgResult.predClass,
                meanLum = qgResult.meanLuminance,
                stdDev = qgResult.stdDev
            )

            var tMqMs = 0.0
            var tYoloMs = 0.0
            var currentYoloResult: YoloResult? = null

            // 3. Execution of Heavyweight Pipeline (MQTone + YOLOv8n) strictly during verification bursts
            if (shouldRunFull) {
                calledFullCount++
                val frame640 = Bitmap.createScaledBitmap(bitmap, 640, 640, true)
                val raw640Tensor = inferenceEngine.createFrame640Tensor(frame640)

                // Learnable illumination enhancement (MQTone)
                val (correctedTensor, mqMs) = inferenceEngine.inferMQTone(raw640Tensor)
                tMqMs = mqMs

                // Dual-task denomination & tear localization detector
                val yResult = inferenceEngine.inferYOLO(correctedTensor)
                tYoloMs = yResult.latencyMs
                currentYoloResult = yResult

                correctedTensor.close()
                raw640Tensor.close()

                // Submit to Temporal Consensus FSM (Rules R5, R6, R7)
                val newlyConfirmed = controller.onVerificationResult(yResult, qgResult.meanLuminance)
                if (newlyConfirmed) {
                    val isTorn = controller.latchedIsTorn
                    triggerHapticFeedback(isTorn)
                    speakDirect(yResult.speechText)
                }
            }

            val totalMs = (SystemClock.elapsedRealtimeNanos() - t0) / 1_000_000.0

            // 4. Hardware Power & Battery Telemetry
            val (currentMa, powerMw, cumJoules) = batteryMeter.sample()

            // Compute Cadence (FPS)
            frameCounter++
            val now = SystemClock.elapsedRealtime()
            if (now - lastFpsTimestamp >= 1000) {
                currentFps = frameCounter * 1000.0 / (now - lastFpsTimestamp)
                frameCounter = 0
                lastFpsTimestamp = now
            }

            // Record Log for Empirical Evaluation
            if (isRecording) {
                sessionFrames++
                val log = FrameLogRecord(
                    timestamp_ms = System.currentTimeMillis(),
                    frame_index = sessionFrames,
                    mode = "cascade",
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
                    denomination = controller.latchedDenom ?: currentYoloResult?.denomination,
                    is_torn = controller.latchedIsTorn || (currentYoloResult?.isTorn ?: false),
                    confidence = controller.latchedConfidence
                )
                sessionLogger.logFrame(log)
            }

            // 5. Update Visual Multimodal Interface
            runOnUiThread {
                updateUI(
                    qg = qgResult,
                    yolo = currentYoloResult,
                    totalMs = totalMs,
                    mqMs = tMqMs,
                    yoloMs = tYoloMs,
                    currentMa = currentMa,
                    powerMw = powerMw,
                    cumJoules = cumJoules
                )
            }
        } catch (e: Exception) {
            e.printStackTrace()
        }
    }

    private fun updateUI(
        qg: QGResult,
        yolo: YoloResult?,
        totalMs: Double,
        mqMs: Double,
        yoloMs: Double,
        currentMa: Double,
        powerMw: Double,
        cumJoules: Double
    ) {
        val state = controller.state

        // Update Overlay View
        overlayView.state = state
        overlayView.lightingClass = qg.predClass
        overlayView.qualityScore = qg.qualityScore

        if (state == CascadeState.CONFIRMED) {
            overlayView.denomination = controller.latchedDenom
            overlayView.confidence = controller.latchedConfidence
            overlayView.isTorn = controller.latchedIsTorn
            overlayView.noteBox = controller.latchedNoteBox
            overlayView.tearBoxes = controller.latchedTearBoxes
        } else if (yolo != null) {
            overlayView.denomination = yolo.denomination
            overlayView.confidence = yolo.confidence
            overlayView.isTorn = yolo.isTorn
            overlayView.noteBox = yolo.noteBox
            overlayView.tearBoxes = yolo.tearBoxes
        }

        // Update State Badge
        when (state) {
            CascadeState.SEARCHING -> {
                tvStateBadge.text = "🔍 SEARCHING FOR BANKNOTE"
                tvStateBadge.setTextColor(ContextCompat.getColor(this, R.color.accent_cyan))
                tvStateBadge.setBackgroundResource(R.drawable.bg_badge_state)
            }
            CascadeState.READY_TO_VERIFY -> {
                tvStateBadge.text = "⚡ HOLD STEADY • VERIFYING"
                tvStateBadge.setTextColor(ContextCompat.getColor(this, R.color.accent_gold))
                tvStateBadge.setBackgroundResource(R.drawable.bg_badge_state)
            }
            CascadeState.CONFIRMED -> {
                tvStateBadge.text = "✔ VERIFIED & CONFIRMED"
                tvStateBadge.setTextColor(ContextCompat.getColor(this, R.color.accent_green))
                tvStateBadge.setBackgroundResource(R.drawable.bg_badge_state)
            }
        }

        // Update Optical Guidance
        when (qg.predClass) {
            "overexposed" -> {
                tvOpticalGuidance.text = getString(R.string.lighting_over)
                tvOpticalGuidance.setTextColor(ContextCompat.getColor(this, R.color.accent_gold))
            }
            "underexposed" -> {
                tvOpticalGuidance.text = getString(R.string.lighting_under)
                tvOpticalGuidance.setTextColor(ContextCompat.getColor(this, R.color.accent_red))
            }
            else -> {
                val qPct = (qg.qualityScore * 100).toInt()
                tvOpticalGuidance.text = "Optimal Lighting (Quality: $qPct%)"
                tvOpticalGuidance.setTextColor(ContextCompat.getColor(this, R.color.text_secondary))
            }
        }

        // Update Hero Result Card
        if (state == CascadeState.CONFIRMED) {
            val denom = controller.latchedDenom ?: "---"
            tvHeroDenomination.text = "$denom ₫"
            setDenominationColor(controller.latchedDenomIndex)

            val isTorn = controller.latchedIsTorn
            if (isTorn) {
                tvDefectStatus.text = getString(R.string.defect_torn)
                tvDefectStatus.setBackgroundResource(R.drawable.bg_pill_torn)
                tvDefectStatus.setTextColor(ContextCompat.getColor(this, R.color.accent_red))
                tvVerificationNote.text = "Warning: Physical substrate tear or edge fracture detected"
            } else {
                tvDefectStatus.text = getString(R.string.defect_intact)
                tvDefectStatus.setBackgroundResource(R.drawable.bg_pill_intact)
                tvDefectStatus.setTextColor(ContextCompat.getColor(this, R.color.accent_green))
                tvVerificationNote.text = "Banknote intact • Latched active (Conserving energy)"
            }
        } else if (state == CascadeState.READY_TO_VERIFY) {
            tvHeroDenomination.text = "..."
            tvHeroDenomination.setTextColor(ContextCompat.getColor(this, R.color.accent_gold))
            tvDefectStatus.text = "⚡ VERIFYING"
            tvDefectStatus.setBackgroundResource(R.drawable.bg_rule_chip)
            tvDefectStatus.setTextColor(ContextCompat.getColor(this, R.color.accent_gold))
            tvVerificationNote.text = "Aggregating multi-frame temporal consensus (Rules R6/R7)..."
        } else {
            updateSearchingUI()
        }

        // Update Telemetry Drawer (if visible)
        if (isTelemetryVisible) {
            val rule = controller.currentRuleStatus
            tvFps.text = String.format("Cadence: %.1f FPS", currentFps)

            chipR1.text = String.format("R1 Texture: %.1f %s", rule.r1StdDev, if (rule.r1TexturePass) "✔" else "✖")
            chipR2.text = String.format("R2 Light: %d%% %s", (rule.r2Score * 100).toInt(), if (rule.r2PhotoPass) "✔" else "✖")
            chipR3.text = String.format("R3 Dwell: %d/3 %s", rule.r3Count, if (rule.r3StablePass) "✔" else "⏳")
            chipR4.text = String.format("R4 Burst: %d/2", rule.r4BurstCount)
            chipR6.text = String.format("R6 Consensus: %s", rule.r6ConsensusWinner ?: "None")
            chipR7.text = String.format("R7 Torn: %s (%d)", if (rule.r7DefectConfirmed) "TORN" else "INTACT", rule.r7TornFrameCount)
            chipR8.text = String.format("R8 Latch: %s (ΔI: %.1f)", if (rule.r8LatchActive) "LATCH" else "IDLE", rule.r8LuminanceDelta)

            tvLatencyBreakdown.text = String.format("QG: %.1fms | MQ: %.1fms | YOLO: %.1fms", qg.latencyMs, mqMs, yoloMs)
            tvTotalLatency.text = String.format("Amort: %.1f ms", totalMs)
            tvPower.text = String.format("Pin: %.0f mA | %.2f W", currentMa, powerMw / 1000.0)
            tvEnergy.text = String.format("Energy: %.2f J", cumJoules)
        }
    }

    private fun setDenominationColor(denomIndex: Int?) {
        val colorRes = when (denomIndex) {
            0 -> R.color.denom_10k
            1 -> R.color.denom_100k
            2 -> R.color.denom_20k
            3 -> R.color.denom_200k
            4 -> R.color.denom_50k
            5 -> R.color.denom_500k
            else -> R.color.white
        }
        tvHeroDenomination.setTextColor(ContextCompat.getColor(this, colorRes))
    }

    private fun updateSearchingUI() {
        tvHeroDenomination.text = "--- ₫"
        tvHeroDenomination.setTextColor(ContextCompat.getColor(this, R.color.white))
        tvDefectStatus.text = "🔍 AWAITING BANKNOTE"
        tvDefectStatus.setBackgroundResource(R.drawable.bg_rule_chip)
        tvDefectStatus.setTextColor(ContextCompat.getColor(this, R.color.text_secondary))
        tvVerificationNote.text = "Present banknote before camera • Automated expert verification"
    }

    private fun speakDirect(text: String) {
        if (!isMuted && ::tts.isInitialized) {
            tts.speak(text, TextToSpeech.QUEUE_FLUSH, null, "CASHVISION_ANNOUNCEMENT")
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
            // Dual pulse for structural fracture / tear alert (Rule R7)
            vibrator.vibrate(VibrationEffect.createWaveform(longArrayOf(0, 100, 80, 150), -1))
        } else {
            // Single pulse for intact confirmation
            vibrator.vibrate(VibrationEffect.createOneShot(150, VibrationEffect.DEFAULT_AMPLITUDE))
        }
    }

    private fun startBenchmarkSession() {
        isRecording = true
        sessionFrames = 0
        calledFullCount = 0
        controller.reset()
        batteryMeter.reset()

        val logFile = sessionLogger.startSession("cascade")
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
                speakDirect("30-second benchmark telemetry session completed")
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
                Toast.makeText(this, "Camera permission is required to operate CashVision!", Toast.LENGTH_SHORT).show()
                finish()
            }
        }
    }

    override fun onDestroy() {
        super.onDestroy()
        cameraExecutor.shutdown()
        if (::tts.isInitialized) {
            tts.stop()
            tts.shutdown()
        }
        sessionLogger.endSession()
    }

    companion object {
        private const val REQUEST_CODE_PERMISSIONS = 10
    }
}
