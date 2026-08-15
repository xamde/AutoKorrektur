package de.konradvoelkel.android.autokorrektur.ar

import android.content.Intent
import android.graphics.Bitmap
import android.graphics.Canvas
import android.net.Uri
import android.os.Bundle
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.Preview
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.core.content.ContextCompat
import androidx.lifecycle.lifecycleScope
import com.google.android.material.snackbar.Snackbar
import de.konradvoelkel.android.autokorrektur.MainActivity
import de.konradvoelkel.android.autokorrektur.R
import de.konradvoelkel.android.autokorrektur.databinding.ActivityArCameraBinding
import de.konradvoelkel.android.autokorrektur.ml.api.YoloService
import de.konradvoelkel.android.autokorrektur.ml.api.YoloServiceImpl
import de.konradvoelkel.android.autokorrektur.ml.engine.YoloTFLiteEngine
import de.konradvoelkel.android.autokorrektur.ui.gallery.VisionGalleryBottomSheet
import de.konradvoelkel.android.autokorrektur.utils.AppLogger
import de.konradvoelkel.android.autokorrektur.utils.ImageExportManager
import kotlinx.coroutines.launch
import org.opencv.android.Utils
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors

/**
 * CameraX-driven Augmented Reality activity that performs real-time vehicle segmentation
 * and renders an asynchronous transparent inpainting patch over the car region,
 * preserving full 30-60 FPS camera preview performance across the rest of the screen.
 */
class ArCameraActivity : AppCompatActivity() {

    private lateinit var binding: ActivityArCameraBinding
    private lateinit var cameraExecutor: ExecutorService
    private val accumulator = TemporalBackgroundAccumulator()
    private lateinit var yoloInference: YoloService
    private lateinit var arPipeline: RealtimeArPipeline
    private lateinit var exportManager: ImageExportManager

    private var latestOverlayBitmap: Bitmap? = null
    private var latestCameraFrameBitmap: Bitmap? = null

    private val photoPickerLauncher = registerForActivityResult(
        ActivityResultContracts.GetContent()
    ) { uri: Uri? ->
        uri?.let { openInStudio(it) }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityArCameraBinding.inflate(layoutInflater)
        setContentView(binding.root)

        cameraExecutor = Executors.newSingleThreadExecutor()
        yoloInference = YoloServiceImpl(YoloTFLiteEngine(this))
        arPipeline = RealtimeArPipeline(yoloInference, accumulator)
        exportManager = ImageExportManager(this)

        arPipeline.onFrameRendered = { overlayPatchBitmap, _ ->
            runOnUiThread {
                if (!isDestroyed && !isFinishing) {
                    val prev = latestOverlayBitmap
                    latestOverlayBitmap = overlayPatchBitmap
                    binding.arOverlayView.setImageBitmap(overlayPatchBitmap)
                    binding.arFpsBadge.text = "● 30 FPS Camera • Active AR Layer"
                    prev?.recycle()
                } else {
                    overlayPatchBitmap.recycle()
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

        binding.resetArButton.setOnClickListener {
            arPipeline.reset()
            Snackbar.make(binding.root, R.string.msg_ar_reset, Snackbar.LENGTH_SHORT).show()
        }

        binding.cardRecentThumbnail.setOnClickListener {
            openVisionGallery()
        }

        binding.btnOpenStudio.setOnClickListener {
            val intent = Intent(this, MainActivity::class.java)
            startActivity(intent)
        }

        binding.captureArButton.setOnClickListener {
            captureCurrentFrame()
        }

        startCamera()
        refreshRecentThumbnail()
    }

    override fun onResume() {
        super.onResume()
        refreshRecentThumbnail()
    }

    private fun refreshRecentThumbnail() {
        lifecycleScope.launch {
            val recent = exportManager.getRecentAutoKorrekturImages(limit = 1)
            if (recent.isNotEmpty()) {
                try {
                    binding.ivRecentThumbnail.setImageURI(recent.first())
                } catch (_: Exception) {}
            }
        }
    }

    private fun captureCurrentFrame() {
        val baseBmp = latestCameraFrameBitmap
        val overlayBmp = latestOverlayBitmap
        if (baseBmp != null) {
            val compositeBmp = Bitmap.createBitmap(baseBmp.width, baseBmp.height, Bitmap.Config.ARGB_8888)
            val canvas = Canvas(compositeBmp)
            canvas.drawBitmap(baseBmp, 0f, 0f, null)
            if (overlayBmp != null && !overlayBmp.isRecycled) {
                canvas.drawBitmap(overlayBmp, 0f, 0f, null)
            }
            val savedUri = exportManager.saveImageToGallery(compositeBmp)
            if (savedUri != null) {
                try {
                    binding.ivRecentThumbnail.setImageURI(savedUri)
                } catch (_: Exception) {}

                Snackbar.make(binding.root, "Car-free vision captured", Snackbar.LENGTH_LONG)
                    .setAction("OPEN IN STUDIO") {
                        openInStudio(savedUri)
                    }
                    .show()
            }
        } else {
            Snackbar.make(binding.root, "No frame ready to capture", Snackbar.LENGTH_SHORT).show()
        }
    }

    private fun openVisionGallery() {
        val gallerySheet = VisionGalleryBottomSheet()
        gallerySheet.onImageSelected = { selectedUri ->
            openInStudio(selectedUri)
        }
        gallerySheet.show(supportFragmentManager, "VisionGalleryBottomSheet")
    }

    private fun openInStudio(imageUri: Uri) {
        val intent = Intent(this, MainActivity::class.java).apply {
            putExtra("EXTRA_IMAGE_URI", imageUri.toString())
        }
        startActivity(intent)
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
                    .setTargetAspectRatio(androidx.camera.core.AspectRatio.RATIO_16_9)
                    .build()
                    .also {
                        it.setSurfaceProvider(binding.cameraPreview.surfaceProvider)
                    }

                val imageAnalyzer = ImageAnalysis.Builder()
                    .setTargetAspectRatio(androidx.camera.core.AspectRatio.RATIO_16_9)
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

                                // Update raw camera frame bitmap for shutter snapshot
                                val currentFrameBmp = Bitmap.createBitmap(
                                    rotatedMat.cols(),
                                    rotatedMat.rows(),
                                    Bitmap.Config.ARGB_8888
                                )
                                Utils.matToBitmap(rotatedMat, currentFrameBmp)
                                val prevFrame = latestCameraFrameBitmap
                                latestCameraFrameBitmap = currentFrameBmp
                                prevFrame?.recycle()

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
        latestOverlayBitmap?.recycle()
        latestOverlayBitmap = null
        latestCameraFrameBitmap?.recycle()
        latestCameraFrameBitmap = null
        super.onDestroy()
    }
}
