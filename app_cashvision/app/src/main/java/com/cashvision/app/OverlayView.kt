package com.cashvision.app

import android.content.Context
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.DashPathEffect
import android.graphics.Paint
import android.graphics.RectF
import android.os.SystemClock
import android.util.AttributeSet
import android.view.View
import kotlin.math.cos
import kotlin.math.max
import kotlin.math.min

class OverlayView @JvmOverloads constructor(
    context: Context,
    attrs: AttributeSet? = null,
    defStyleAttr: Int = 0
) : View(context, attrs, defStyleAttr) {

    // Detection Data
    var state: CascadeState = CascadeState.SEARCHING
        set(value) {
            field = value
            postInvalidateOnAnimation()
        }

    var noteBox: RectF? = null
        set(value) {
            field = value
            postInvalidateOnAnimation()
        }

    var tearBoxes: List<RectF> = emptyList()
        set(value) {
            field = value
            postInvalidateOnAnimation()
        }

    var denomination: String? = null
    var confidence: Float = 0f
    var isTorn: Boolean = false
    var lightingClass: String = "good" // "good", "overexposed", "underexposed"
    var qualityScore: Float = 1.0f

    // Paints
    private val cornerPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = Color.parseColor("#00E5FF") // Vivid Cyan
        style = Paint.Style.STROKE
        strokeWidth = 8f
        strokeCap = Paint.Cap.ROUND
    }

    private val guideFramePaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = Color.parseColor("#33FFFFFF")
        style = Paint.Style.STROKE
        strokeWidth = 2.5f
        pathEffect = DashPathEffect(floatArrayOf(16f, 16f), 0f)
    }

    private val noteBoxPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        style = Paint.Style.STROKE
        strokeWidth = 6f
        strokeCap = Paint.Cap.ROUND
    }

    private val noteFillPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        style = Paint.Style.FILL
    }

    private val tearBoxPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = Color.parseColor("#FF1744") // Vivid Alert Red
        style = Paint.Style.STROKE
        strokeWidth = 5f
        strokeCap = Paint.Cap.ROUND
    }

    private val tearFillPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = Color.parseColor("#33FF1744")
        style = Paint.Style.FILL
    }

    private val badgeBgPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = Color.parseColor("#E6111827") // Dark Navy Glass
        style = Paint.Style.FILL
    }

    private val badgeBorderPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        style = Paint.Style.STROKE
        strokeWidth = 2.5f
    }

    private val textPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = Color.WHITE
        textSize = 34f
        isFakeBoldText = true
    }

    private val subTextPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = Color.parseColor("#D1D5DB")
        textSize = 24f
    }

    override fun onDraw(canvas: Canvas) {
        super.onDraw(canvas)
        val w = width.toFloat()
        val h = height.toFloat()
        if (w <= 0 || h <= 0) return

        // 1. Draw Viewfinder Guide Frame & Scanner Corners
        drawViewfinderReticle(canvas, w, h)

        // 2. Draw Detected Banknote Bounding Box
        val curNoteBox = noteBox
        if (curNoteBox != null && (state == CascadeState.READY_TO_VERIFY || state == CascadeState.CONFIRMED)) {
            drawBanknoteBox(canvas, curNoteBox, w, h)
        }

        // 3. Draw Tear Defect Boxes
        if (tearBoxes.isNotEmpty() && (state == CascadeState.READY_TO_VERIFY || state == CascadeState.CONFIRMED)) {
            for (box in tearBoxes) {
                drawTearBox(canvas, box, w, h)
            }
        }

        // Continuously animate during active states
        if (state == CascadeState.SEARCHING || state == CascadeState.READY_TO_VERIFY) {
            postInvalidateOnAnimation()
        }
    }

    private fun drawViewfinderReticle(canvas: Canvas, w: Float, h: Float) {
        // Center aspect ratio 1.85:1 for banknote presentation guide
        val guideW = w * 0.85f
        val guideH = guideW / 1.85f
        val left = (w - guideW) / 2f
        val top = (h - guideH) / 2f - (h * 0.05f) // Slightly higher to accommodate bottom sheet
        val right = left + guideW
        val bottom = top + guideH

        // Dashed bounding rectangle
        canvas.drawRoundRect(left, top, right, bottom, 28f, 28f, guideFramePaint)

        // Dynamic pulsing corner reticles
        val now = SystemClock.uptimeMillis()
        val pulse = 0.7f + 0.3f * (0.5f + 0.5f * cos(now / 350.0).toFloat())

        val cornerColor = when (state) {
            CascadeState.CONFIRMED -> Color.parseColor("#00E676") // Emerald
            CascadeState.READY_TO_VERIFY -> Color.parseColor("#FFD600") // Amber Gold
            CascadeState.SEARCHING -> Color.argb((255 * pulse).toInt(), 0, 229, 255) // Cyan pulse
        }
        cornerPaint.color = cornerColor
        val cornerLen = 42f

        // Top-Left
        canvas.drawLine(left, top, left + cornerLen, top, cornerPaint)
        canvas.drawLine(left, top, left, top + cornerLen, cornerPaint)

        // Top-Right
        canvas.drawLine(right - cornerLen, top, right, top, cornerPaint)
        canvas.drawLine(right, top, right, top + cornerLen, cornerPaint)

        // Bottom-Left
        canvas.drawLine(left, bottom - cornerLen, left, bottom, cornerPaint)
        canvas.drawLine(left, bottom, left + cornerLen, bottom, cornerPaint)

        // Bottom-Right
        canvas.drawLine(right - cornerLen, bottom, right, bottom, cornerPaint)
        canvas.drawLine(right, bottom - cornerLen, right, bottom, cornerPaint)
    }

    private fun drawBanknoteBox(canvas: Canvas, box: RectF, viewW: Float, viewH: Float) {
        val rect = RectF(
            box.left * viewW,
            box.top * viewH,
            box.right * viewW,
            box.bottom * viewH
        )

        val strokeColor = if (isTorn) Color.parseColor("#FF1744") else Color.parseColor("#00E676")
        val fillColor = if (isTorn) Color.parseColor("#1AFF1744") else Color.parseColor("#1A00E676")

        noteBoxPaint.color = strokeColor
        noteFillPaint.color = fillColor

        canvas.drawRoundRect(rect, 20f, 20f, noteFillPaint)
        canvas.drawRoundRect(rect, 20f, 20f, noteBoxPaint)

        // Floating Title Badge
        val title = denomination?.let { "$it ₫" } ?: "Analyzing..."
        val statusText = if (isTorn) "⚠️ TORN DEFECT • ${(confidence * 100).toInt()}%"
                         else "✔ INTACT • ${(confidence * 100).toInt()}%"

        val titleW = textPaint.measureText(title)
        val statusW = subTextPaint.measureText(statusText)
        val badgeW = max(titleW, statusW) + 40f
        val badgeH = 80f

        val badgeLeft = max(16f, min(rect.left, viewW - badgeW - 16f))
        val badgeTop = max(16f, rect.top - badgeH - 12f)
        val badgeRect = RectF(badgeLeft, badgeTop, badgeLeft + badgeW, badgeTop + badgeH)

        badgeBorderPaint.color = strokeColor
        canvas.drawRoundRect(badgeRect, 16f, 16f, badgeBgPaint)
        canvas.drawRoundRect(badgeRect, 16f, 16f, badgeBorderPaint)

        canvas.drawText(title, badgeLeft + 20f, badgeTop + 36f, textPaint)
        canvas.drawText(statusText, badgeLeft + 20f, badgeTop + 68f, subTextPaint)
    }

    private fun drawTearBox(canvas: Canvas, box: RectF, viewW: Float, viewH: Float) {
        val rect = RectF(
            box.left * viewW,
            box.top * viewH,
            box.right * viewW,
            box.bottom * viewH
        )

        canvas.drawRoundRect(rect, 10f, 10f, tearFillPaint)
        canvas.drawRoundRect(rect, 10f, 10f, tearBoxPaint)

        // Mini warning tag
        val tagText = "⚠️ TORN"
        val tagW = subTextPaint.measureText(tagText) + 20f
        val tagH = 40f
        val tagRect = RectF(rect.left, max(8f, rect.top - tagH - 4f), rect.left + tagW, max(8f, rect.top - tagH - 4f) + tagH)

        canvas.drawRoundRect(tagRect, 8f, 8f, badgeBgPaint)
        badgeBorderPaint.color = Color.parseColor("#FF1744")
        canvas.drawRoundRect(tagRect, 8f, 8f, badgeBorderPaint)
        canvas.drawText(tagText, tagRect.left + 10f, tagRect.top + 28f, subTextPaint)
    }
}
