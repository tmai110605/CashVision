package com.cashvision.app

import android.graphics.RectF
import android.os.SystemClock
import kotlin.math.abs

enum class CascadeState {
    SEARCHING,
    READY_TO_VERIFY,
    CONFIRMED
}

/**
 * Diagnostic record representing the real-time evaluation status of Knowledge Base Rules R1-R8.
 */
data class RuleStatus(
    val r1TexturePass: Boolean,
    val r1StdDev: Double,
    val r2PhotoPass: Boolean,
    val r2Score: Float,
    val r2PredClass: String,
    val r3StablePass: Boolean,
    val r3Count: Int,
    val r4BurstCount: Int,
    val r5ConfPass: Boolean,
    val r6ConsensusWinner: String?,
    val r6ScoreSum: Float,
    val r7DefectConfirmed: Boolean,
    val r7TornFrameCount: Int,
    val r8LatchActive: Boolean,
    val r8LuminanceDelta: Double
)

/**
 * CashVision Stateful Forward-Chaining Inference Engine & Meta-Level Controller.
 * Implements 100% of the Expert System Architecture codified by Rules R1-R8 (Table 1 of manuscript):
 * - R1: Spatial Texture Gating (theta_texture = 15.0)
 * - R2: Photometric Readiness (tau = 0.60)
 * - R3: Pre-inference Temporal Stability (K_opt = 3 consecutive frames)
 * - R4: Bounded Verification Burst Budget (M_verify = 2 attempts)
 * - R5: Candidate Detection Filtering (theta_conf = 0.25)
 * - R6: Multi-Frame Confidence-Weighted Denomination Consensus
 * - R7: Two-Frame Defect Persistence Filter
 * - R8: Dual-Condition Latch Reset (texture drop or mean luminance shift > 25.0)
 */
class CascadeController(
    val textureThreshold: Double = 15.0,    // R1, R8a
    val qualityThreshold: Float = 0.60f,     // R2
    val requiredStableFrames: Int = 3,      // R3 (K_opt = 3)
    val maxVerifyFrames: Int = 2,           // R4 (M_verify = 2)
    val confThreshold: Float = 0.25f,       // R5 (theta_conf = 0.25)
    val transitionThreshold: Double = 25.0, // R8b (theta_trans = 25.0)
    val periodicCheckIntervalMs: Long = 1500L // R8c (1.5s periodic verification)
) {
    var state: CascadeState = CascadeState.SEARCHING
        private set

    var stableCount: Int = 0
        private set

    var verifyBurstCount: Int = 0
        private set

    // Accumulated denomination scores across verification burst frames (classes 0..5)
    val accumulatedScores = FloatArray(6) { 0.0f }

    // Count of triggered frames detecting torn defects
    var tornFrameCount: Int = 0
        private set

    // Latched Hypothesis (State 3: CONFIRMED)
    var latchedDenom: String? = null
        private set
    var latchedDenomIndex: Int? = null
        private set
    var latchedConfidence: Float = 0f
        private set
    var latchedIsTorn: Boolean = false
        private set
    var latchedNoteBox: RectF? = null
        private set
    var latchedTearBoxes: List<RectF> = emptyList()
        private set
    var latchedMeanLuminance: Double = 0.0
        private set
    var latchedTimestampMs: Long = 0L
        private set

    // Current Rule Status for Telemetry HUD
    var currentRuleStatus: RuleStatus = RuleStatus(
        r1TexturePass = false,
        r1StdDev = 0.0,
        r2PhotoPass = false,
        r2Score = 0.0f,
        r2PredClass = "good",
        r3StablePass = false,
        r3Count = 0,
        r4BurstCount = 0,
        r5ConfPass = false,
        r6ConsensusWinner = null,
        r6ScoreSum = 0.0f,
        r7DefectConfirmed = false,
        r7TornFrameCount = 0,
        r8LatchActive = false,
        r8LuminanceDelta = 0.0
    )
        private set

    private val denomMap = mapOf(
        0 to "10,000",
        1 to "100,000",
        2 to "20,000",
        3 to "200,000",
        4 to "50,000",
        5 to "500,000"
    )

    /**
     * Meta-Level Control: Determines whether to trigger compute-intensive deep models (MQTone + YOLOv8n).
     * Evaluates Rules R1, R2, R3, R4, R8.
     * Returns true if deep inference should execute on this frame, false otherwise.
     */
    fun decide(
        hasNote: Boolean,
        quality: Float,
        predClass: String,
        meanLum: Double,
        stdDev: Double
    ): Boolean {
        val r1Pass = hasNote && (stdDev > textureThreshold)
        val r2Pass = (quality >= qualityThreshold) && (predClass == "good" || quality >= 0.40f)

        // 1. In CONFIRMED State: Rule R8 (Dual-Condition Latch Reset)
        if (state == CascadeState.CONFIRMED) {
            val deltaLum = abs(meanLum - latchedMeanLuminance)
            val now = SystemClock.elapsedRealtime()

            // R8a: Spatial texture drops below threshold (banknote removed or placed in pocket/blank table)
            val r8aReset = stdDev <= textureThreshold
            // R8b: Temporal mean luminance shift exceeds transition boundary (note swapped or withdrawn)
            val r8bReset = deltaLum > transitionThreshold
            // R8c: Periodic 1.5s check condition
            val r8cReset = (now - latchedTimestampMs > periodicCheckIntervalMs) && !hasNote

            if (r8aReset || r8bReset || r8cReset) {
                reset()
                return false
            }

            // Suppress deep inference to preserve battery autonomy and prevent thermal surge
            currentRuleStatus = currentRuleStatus.copy(
                r1TexturePass = r1Pass,
                r1StdDev = stdDev,
                r2PhotoPass = r2Pass,
                r2Score = quality,
                r2PredClass = predClass,
                r8LatchActive = true,
                r8LuminanceDelta = deltaLum
            )
            return false
        }

        // 2. In SEARCHING State: Rules R1, R2, R3
        if (state == CascadeState.SEARCHING) {
            if (r1Pass && r2Pass) {
                stableCount++
                if (stableCount >= requiredStableFrames) {
                    // Rule R3: Transition to READY_TO_VERIFY upon K_opt stable frames
                    state = CascadeState.READY_TO_VERIFY
                    verifyBurstCount = 1
                    accumulatedScores.fill(0f)
                    tornFrameCount = 0

                    currentRuleStatus = currentRuleStatus.copy(
                        r1TexturePass = true,
                        r1StdDev = stdDev,
                        r2PhotoPass = true,
                        r2Score = quality,
                        r2PredClass = predClass,
                        r3StablePass = true,
                        r3Count = stableCount,
                        r4BurstCount = 1,
                        r8LatchActive = false,
                        r8LuminanceDelta = 0.0
                    )
                    return true // Open verification burst window
                }
            } else {
                stableCount = 0
            }

            currentRuleStatus = currentRuleStatus.copy(
                r1TexturePass = r1Pass,
                r1StdDev = stdDev,
                r2PhotoPass = r2Pass,
                r2Score = quality,
                r2PredClass = predClass,
                r3StablePass = false,
                r3Count = stableCount,
                r4BurstCount = 0,
                r8LatchActive = false,
                r8LuminanceDelta = 0.0
            )
            return false
        }

        // 3. In READY_TO_VERIFY State: Rule R4 (Burst Budget)
        if (state == CascadeState.READY_TO_VERIFY) {
            verifyBurstCount++
            if (verifyBurstCount > maxVerifyFrames) {
                // Verification attempts exhausted without consensus -> reset to SEARCHING
                reset()
                return false
            }

            currentRuleStatus = currentRuleStatus.copy(
                r1TexturePass = r1Pass,
                r1StdDev = stdDev,
                r2PhotoPass = r2Pass,
                r2Score = quality,
                r2PredClass = predClass,
                r4BurstCount = verifyBurstCount
            )
            return true
        }

        return false
    }

    /**
     * Submits candidate deep inference results to the Temporal Consensus Engine.
     * Evaluates Rules R5 (Filtering), R6 (Consensus Summation), and R7 (Two-Frame Defect Confirmation).
     * Returns true if consensus was reached and state transitioned to CONFIRMED.
     */
    fun onVerificationResult(yolo: YoloResult, currentLum: Double): Boolean {
        if (state != CascadeState.READY_TO_VERIFY) return false

        // Rule R5: Candidate detection filtering
        val confPass = yolo.confidence >= confThreshold

        // Accumulate denomination score evidence (Evidence Accumulation Theory)
        for (c in 0..5) {
            accumulatedScores[c] += yolo.scores[c]
        }

        // Accumulate tear defect persistence evidence
        if (yolo.isTorn && yolo.tornConfidence >= confThreshold) {
            tornFrameCount++
        }

        // Find candidate denomination with maximum accumulated confidence
        var bestId = 0
        var maxScore = accumulatedScores[0]
        for (c in 1..5) {
            if (accumulatedScores[c] > maxScore) {
                maxScore = accumulatedScores[c]
                bestId = c
            }
        }
        val bestDenomStr = denomMap[bestId]

        // Rule R6 & R7 Arbitration:
        // Case A: High-confidence intact banknote confirmed on initial active frame (m = 1, turnaround ~0.56s)
        if (verifyBurstCount == 1 && confPass && maxScore >= 0.70f && tornFrameCount == 0) {
            latchConfirmation(
                denomStr = bestDenomStr,
                denomIdx = bestId,
                confidence = yolo.confidence,
                isTorn = false,
                noteBox = yolo.noteBox,
                tearBoxes = emptyList(),
                currentLum = currentLum
            )
            return true
        }

        // Case B: Full verification burst completed (m == M_verify = 2)
        if (verifyBurstCount >= maxVerifyFrames) {
            if (maxScore >= confThreshold) {
                // Rule R7: Two-frame defect persistence (confirmed iff torn detected in >= 2 frames)
                val defectConfirmed = tornFrameCount >= 2

                latchConfirmation(
                    denomStr = bestDenomStr,
                    denomIdx = bestId,
                    confidence = maxScore / 2.0f,
                    isTorn = defectConfirmed,
                    noteBox = yolo.noteBox,
                    tearBoxes = if (defectConfirmed) yolo.tearBoxes else emptyList(),
                    currentLum = currentLum
                )
                return true
            } else {
                // Insufficient consensus -> safely return to SEARCHING
                reset()
                return false
            }
        }

        // Case C: Continue burst to frame 2 (e.g. verifying defect persistence or accumulating evidence)
        currentRuleStatus = currentRuleStatus.copy(
            r5ConfPass = confPass,
            r6ConsensusWinner = bestDenomStr,
            r6ScoreSum = maxScore,
            r7DefectConfirmed = false,
            r7TornFrameCount = tornFrameCount
        )
        return false
    }

    private fun latchConfirmation(
        denomStr: String?,
        denomIdx: Int,
        confidence: Float,
        isTorn: Boolean,
        noteBox: RectF?,
        tearBoxes: List<RectF>,
        currentLum: Double
    ) {
        state = CascadeState.CONFIRMED
        latchedDenom = denomStr
        latchedDenomIndex = denomIdx
        latchedConfidence = confidence
        latchedIsTorn = isTorn
        latchedNoteBox = noteBox
        latchedTearBoxes = tearBoxes
        latchedMeanLuminance = currentLum
        latchedTimestampMs = SystemClock.elapsedRealtime()

        currentRuleStatus = currentRuleStatus.copy(
            r5ConfPass = true,
            r6ConsensusWinner = denomStr,
            r6ScoreSum = confidence,
            r7DefectConfirmed = isTorn,
            r7TornFrameCount = tornFrameCount,
            r8LatchActive = true,
            r8LuminanceDelta = 0.0
        )
    }

    fun reset() {
        state = CascadeState.SEARCHING
        stableCount = 0
        verifyBurstCount = 0
        accumulatedScores.fill(0f)
        tornFrameCount = 0
        latchedDenom = null
        latchedDenomIndex = null
        latchedConfidence = 0f
        latchedIsTorn = false
        latchedNoteBox = null
        latchedTearBoxes = emptyList()
        latchedMeanLuminance = 0.0
        latchedTimestampMs = 0L

        currentRuleStatus = currentRuleStatus.copy(
            r3StablePass = false,
            r3Count = 0,
            r4BurstCount = 0,
            r6ConsensusWinner = null,
            r6ScoreSum = 0.0f,
            r7DefectConfirmed = false,
            r7TornFrameCount = 0,
            r8LatchActive = false,
            r8LuminanceDelta = 0.0
        )
    }
}
