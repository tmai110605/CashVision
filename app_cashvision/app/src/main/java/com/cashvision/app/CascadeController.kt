package com.cashvision.app

enum class CascadeState {
    SEARCHING,
    READY_TO_VERIFY,
    CONFIRMED
}

class CascadeController(
    private val qualityThreshold: Float = 0.6f,
    private val requiredStableFrames: Int = 3,
    private val maxVerifyFrames: Int = 2
) {
    var state: CascadeState = CascadeState.SEARCHING
        private set

    var stableCount: Int = 0
        private set

    var verifyCount: Int = 0
        private set

    fun decide(hasNote: Boolean, quality: Float): Boolean {
        if (!hasNote) {
            reset()
            return false
        }

        if (state == CascadeState.CONFIRMED) {
            // Banknote confirmed: maintain display result and maximize power saving
            return false
        }

        if (state == CascadeState.SEARCHING) {
            if (quality >= qualityThreshold) {
                stableCount++
                if (stableCount >= requiredStableFrames) {
                    state = CascadeState.READY_TO_VERIFY
                    verifyCount = 1
                    return true
                }
            } else {
                stableCount = 0
            }
            return false
        }

        if (state == CascadeState.READY_TO_VERIFY) {
            verifyCount++
            if (verifyCount > maxVerifyFrames) {
                // Max attempts exhausted without detection -> return to SEARCHING
                reset()
                return false
            }
            return true
        }

        return false
    }

    fun onDetectionResult(hasDetection: Boolean) {
        if (hasDetection) {
            state = CascadeState.CONFIRMED
        } else if (verifyCount >= maxVerifyFrames) {
            reset()
        }
    }

    fun reset() {
        state = CascadeState.SEARCHING
        stableCount = 0
        verifyCount = 0
    }
}
