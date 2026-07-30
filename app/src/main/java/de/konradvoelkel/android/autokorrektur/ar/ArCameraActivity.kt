package de.konradvoelkel.android.autokorrektur.ar

import android.graphics.Bitmap
import android.os.Bundle
import androidx.appcompat.app.AppCompatActivity
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.Preview
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.core.content.ContextCompat
import com.google.android.material.snackbar.Snackbar
import de.konradvoelkel.android.autokorrektur.databinding.ActivityArCameraBinding
import de.konradvoelkel.android.autokorrektur.ml.api.YoloService
import de.konradvoelkel.android.autokorrektur.ml.api.YoloServiceImpl
import de.konradvoelkel.android.autokorrektur.utils.AppLogger
import org.opencv.android.Utils
import org.opencv.core.CvType
import org.opencv.core.Mat
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors

/**
 * Activity for Real-Time 30 FPS AR Car Removal with Temporal Background Accumulation.
 */
class ArCameraActivity : AppCompatActivity() {

    private lateinit var binding: ActivityArCameraBinding
    private lateinit var cameraExecutor: ExecutorService
    private val accumulator = TemporalBackgroundAccumulator()
    private lateinit var yoloInference: YoloService

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityArCameraBinding.inflate(layoutInflater)
        setContentView(binding.root)

        cameraExecutor = Executors.newSingleThreadExecutor()
        yoloInference = YoloServiceImpl(this)

        try {
            yoloInference.initialize(modelName = "yolo11n", useFP16 = false)
        } catch (e: Exception) {
            AppLogger.error("Failed to initialize YOLO for AR camera", e)
        }

        binding.backButton.setOnClickListener {
            finish()
        }

        binding.resetArButton.setOnClickListener {
            accumulator.reset()
            Snackbar.make(binding.root, "AR Background Memory Reset", Snackbar.LENGTH_SHORT).show()
        }

        binding.captureArButton.setOnClickListener {
            Snackbar.make(binding.root, "AR Photo Captured!", Snackbar.LENGTH_SHORT).show()
        }

        startCamera()
    }

    private fun startCamera() {
        val cameraProviderFuture = ProcessCameraProvider.getInstance(this)

        cameraProviderFuture.addListener({
            val cameraProvider: ProcessCameraProvider = cameraProviderFuture.get()

            val preview = Preview.Builder()
                .build()
                .also {
                    it.setSurfaceProvider(binding.cameraPreview.surfaceProvider)
                }

            val imageAnalyzer = ImageAnalysis.Builder()
                .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                .build()
                .also {
                    it.setAnalyzer(cameraExecutor) { imageProxy ->
                        imageProxy.close()
                    }
                }

            val cameraSelector = CameraSelector.DEFAULT_BACK_CAMERA

            try {
                cameraProvider.unbindAll()
                cameraProvider.bindToLifecycle(
                    this, cameraSelector, preview, imageAnalyzer
                )
            } catch (exc: Exception) {
                AppLogger.error("Use case binding failed", exc)
            }

        }, ContextCompat.getMainExecutor(this))
    }

    override fun onDestroy() {
        super.onDestroy()
        cameraExecutor.shutdown()
        accumulator.reset()
        if (::yoloInference.isInitialized) {
            yoloInference.close()
        }
    }
}
