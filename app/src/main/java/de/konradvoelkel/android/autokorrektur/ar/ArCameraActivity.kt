package de.konradvoelkel.android.autokorrektur.ar

import android.graphics.Bitmap
import android.os.Bundle
import androidx.appcompat.app.AppCompatActivity
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.Preview
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.core.content.ContextCompat
import androidx.lifecycle.lifecycleScope
import com.google.android.material.snackbar.Snackbar
import de.konradvoelkel.android.autokorrektur.R
import de.konradvoelkel.android.autokorrektur.databinding.ActivityArCameraBinding
import de.konradvoelkel.android.autokorrektur.ml.api.YoloService
import de.konradvoelkel.android.autokorrektur.ml.api.YoloServiceImpl
import de.konradvoelkel.android.autokorrektur.ml.engine.YoloTFLiteEngine
import de.konradvoelkel.android.autokorrektur.utils.AppLogger
import de.konradvoelkel.android.autokorrektur.utils.ImageExportManager
import kotlinx.coroutines.launch
import org.opencv.core.Mat
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors

/**
 * CameraX-driven Augmented Reality activity that performs real-time vehicle segmentation
 * and replaces detected vehicle pixels using temporal background accumulation.
 */
class ArCameraActivity : AppCompatActivity() {

    private lateinit var binding: ActivityArCameraBinding
    private lateinit var cameraExecutor: ExecutorService
    private val accumulator = TemporalBackgroundAccumulator()
    private lateinit var yoloInference: YoloService
    private lateinit var arPipeline: RealtimeArPipeline
    private lateinit var exportManager: ImageExportManager

    private var latestRenderedBitmap: Bitmap? = null

    /**
     * Sets up CameraX view binding, buttons, and launches asynchronous YOLO engine initialization.
     */
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityArCameraBinding.inflate(layoutInflater)
        setContentView(binding.root)

        cameraExecutor = Executors.newSingleThreadExecutor()
        yoloInference = YoloServiceImpl(YoloTFLiteEngine(this))
        arPipeline = RealtimeArPipeline(yoloInference, accumulator)
        exportManager = ImageExportManager(this)

        arPipeline.onFrameRendered = { bitmap, fps ->
            runOnUiThread {
                if (!isDestroyed && !isFinishing) {
                    val prev = latestRenderedBitmap
                    latestRenderedBitmap = bitmap
                    binding.arOverlayView.setImageBitmap(bitmap)
                    binding.arFpsBadge.text = "${fps.toInt().coerceAtLeast(1)} FPS"
                    prev?.recycle()
                } else {
                    bitmap.recycle()
                }
            }
        }

        lifecycleScope.launch {
            try {
                arPipeline.initialize(modelName = "yolo11s")
            } catch (e: Exception) {
                AppLogger.error("Failed to initialize YOLO for AR camera", e)
            }
        }

        binding.backButton.setOnClickListener {
            finish()
        }

        binding.resetArButton.setOnClickListener {
            arPipeline.reset()
            Snackbar.make(binding.root, R.string.msg_ar_reset, Snackbar.LENGTH_SHORT).show()
        }

        binding.captureArButton.setOnClickListener {
            val currentBmp = latestRenderedBitmap
            if (currentBmp != null) {
                val copyBmp = currentBmp.copy(currentBmp.config ?: Bitmap.Config.ARGB_8888, true)
                exportManager.saveImageToGallery(copyBmp)
                Snackbar.make(binding.root, R.string.msg_ar_captured, Snackbar.LENGTH_SHORT).show()
            } else {
                Snackbar.make(binding.root, "No AR frame ready to capture", Snackbar.LENGTH_SHORT).show()
            }
        }

        startCamera()
    }

    private var cameraProvider: ProcessCameraProvider? = null

    private fun startCamera() {
        val cameraProviderFuture = ProcessCameraProvider.getInstance(this)

        cameraProviderFuture.addListener({
            if (isDestroyed || isFinishing) return@addListener

            try {
                val provider: ProcessCameraProvider = cameraProviderFuture.get()
                cameraProvider = provider

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
                            try {
                                val rotation = imageProxy.imageInfo.rotationDegrees
                                val rgbaMat = ArFrameConverter.yuvImageProxyToRgbaMat(imageProxy)
                                val rotatedMat = if (rotation != 0) {
                                    val rot = ArFrameConverter.rotateMat(rgbaMat, rotation)
                                    rgbaMat.release()
                                    rot
                                } else {
                                    rgbaMat
                                }

                                arPipeline.processFrame(rotatedMat)
                                rotatedMat.release()
                            } catch (e: Exception) {
                                AppLogger.error("Error analyzing AR frame", e)
                            } finally {
                                imageProxy.close()
                            }
                        }
                    }

                val cameraSelector = CameraSelector.DEFAULT_BACK_CAMERA

                provider.unbindAll()
                provider.bindToLifecycle(
                    this, cameraSelector, preview, imageAnalyzer
                )
            } catch (exc: Exception) {
                AppLogger.error("Use case binding failed", exc)
            }

        }, ContextCompat.getMainExecutor(this))
    }

    override fun onDestroy() {
        cameraExecutor.shutdown()
        cameraProvider?.unbindAll()
        arPipeline.close()
        latestRenderedBitmap?.recycle()
        latestRenderedBitmap = null
        super.onDestroy()
    }
}
