package de.konradvoelkel.android.autokorrektur.shared

import androidx.test.platform.app.InstrumentationRegistry
import de.konradvoelkel.android.autokorrektur.ml.ImageProcessor
import de.konradvoelkel.android.autokorrektur.ml.MiGanInference
import de.konradvoelkel.android.autokorrektur.ml.YoloInferenceTFLite

/**
 * Shared, lazily-initialized fixtures for pipeline instrumented tests.
 * Avoids repeated model initialization across multiple test classes.
 */
object PipelineTestFixtures {
    private val appContext get() = InstrumentationRegistry.getInstrumentation().targetContext

    @Volatile private var _yolo: YoloInferenceTFLite? = null
    @Volatile private var _migan: MiGanInference? = null
    @Volatile private var _imageProcessor: ImageProcessor? = null

    fun yolo(): YoloInferenceTFLite {
        var local = _yolo
        if (local == null) {
            synchronized(this) {
                local = _yolo
                if (local == null) {
                    val y = YoloInferenceTFLite(appContext)
                    y.initialize("yolo11s")
                    _yolo = y
                    local = y
                }
            }
        }
        return local!!
    }

    fun migan(): MiGanInference {
        var local = _migan
        if (local == null) {
            synchronized(this) {
                local = _migan
                if (local == null) {
                    val m = MiGanInference(appContext)
                    m.initialize()
                    _migan = m
                    local = m
                }
            }
        }
        return local!!
    }

    fun imageProcessor(): ImageProcessor {
        var local = _imageProcessor
        if (local == null) {
            synchronized(this) {
                local = _imageProcessor
                if (local == null) {
                    val p = ImageProcessor(appContext)
                    _imageProcessor = p
                    local = p
                }
            }
        }
        return local!!
    }

    /** Optional explicit cleanup, typically not required across test process lifetime. */
    fun closeAll() {
        _yolo?.close(); _yolo = null
        _migan?.close(); _migan = null
        _imageProcessor = null
    }
}
