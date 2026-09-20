package com.cashvision.app

import android.content.Context
import android.content.Intent
import android.content.IntentFilter
import android.os.BatteryManager
import kotlin.math.abs

class BatteryMeter(private val context: Context) {

    private val batteryManager = context.getSystemService(Context.BATTERY_SERVICE) as BatteryManager
    private var lastTimestampNanos: Long = System.nanoTime()

    var cumulativeJoules: Double = 0.0
        private set

    /**
     * Retrieve real-time battery voltage (mV) from system Intent
     */
    private fun getBatteryVoltageMv(): Double {
        val intent = context.registerReceiver(null, IntentFilter(Intent.ACTION_BATTERY_CHANGED))
        val voltage = intent?.getIntExtra(BatteryManager.EXTRA_VOLTAGE, -1) ?: -1
        return if (voltage > 0) voltage.toDouble() else 3850.0 // Default to 3.85V if not supported
    }

    /**
     * Measure instantaneous power and accumulate energy consumption
     * @return Triple(Discharge Current mA, Power mW, Accumulated Energy Joules)
     */
    fun sample(): Triple<Double, Double, Double> {
        val nowNanos = System.nanoTime()
        val dtSec = (nowNanos - lastTimestampNanos) / 1_000_000_000.0
        lastTimestampNanos = nowNanos

        // Instantaneous current in micro-amperes (uA)
        val currentMicroAmps = batteryManager.getIntProperty(BatteryManager.BATTERY_PROPERTY_CURRENT_NOW)
        val currentMa = abs(currentMicroAmps) / 1000.0
        val voltageMv = getBatteryVoltageMv()

        // Instantaneous power: P (mW) = (I_mA * V_mV) / 1000.0
        val powerMilliWatts = (currentMa * voltageMv) / 1000.0

        // Accumulated energy: E (Joules) += P (Watts) * dt (seconds)
        if (dtSec in 0.001..2.0) {
            cumulativeJoules += (powerMilliWatts / 1000.0) * dtSec
        }

        return Triple(currentMa, powerMilliWatts, cumulativeJoules)
    }

    fun reset() {
        lastTimestampNanos = System.nanoTime()
        cumulativeJoules = 0.0
    }
}
