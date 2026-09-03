package de.konradvoelkel.android.autokorrektur.video

import android.content.ContentValues
import android.content.Intent
import android.net.Uri
import android.os.Build
import android.os.Bundle
import android.os.Environment
import android.provider.MediaStore
import android.view.View
import androidx.appcompat.app.AppCompatActivity
import androidx.core.content.FileProvider
import androidx.lifecycle.lifecycleScope
import com.google.android.material.snackbar.Snackbar
import de.konradvoelkel.android.autokorrektur.R
import de.konradvoelkel.android.autokorrektur.databinding.ActivityVideoPreviewBinding
import de.konradvoelkel.android.autokorrektur.ml.MiGanInference
import de.konradvoelkel.android.autokorrektur.ml.api.YoloServiceImpl
import de.konradvoelkel.android.autokorrektur.ml.engine.YoloTFLiteEngine
import de.konradvoelkel.android.autokorrektur.utils.AppLogger
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.io.File
import java.io.FileInputStream

/**
 * Activity for displaying captured and HQ-inpainted car-free video snippets,
 * allowing instant Before/After comparison and direct Instagram Reels / Stories sharing.
 */
class VideoPreviewActivity : AppCompatActivity() {

    private lateinit var binding: ActivityVideoPreviewBinding
    private var rawVideoPath: String? = null
    private var processedVideoFile: File? = null
    private var isShowingProcessed = true

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityVideoPreviewBinding.inflate(layoutInflater)
        setContentView(binding.root)

        rawVideoPath = intent.getStringExtra(EXTRA_RAW_VIDEO_PATH)

        binding.btnBack.setOnClickListener {
            finish()
        }

        binding.btnToggleSource.setOnClickListener {
            toggleVideoSource()
        }

        binding.btnSaveGallery.setOnClickListener {
            saveVideoToGallery()
        }

        binding.btnShareReels.setOnClickListener {
            shareToInstagramReels()
        }

        binding.videoView.setOnPreparedListener { mediaPlayer ->
            mediaPlayer.isLooping = true
        }

        val rawPath = rawVideoPath
        if (rawPath != null) {
            val rawFile = File(rawPath)
            if (rawFile.exists()) {
                startVideoProcessing(rawFile)
            } else {
                Snackbar.make(binding.root, getString(R.string.video_error_not_found), Snackbar.LENGTH_LONG).show()
            }
        }
    }

    private fun startVideoProcessing(rawFile: File) {
        binding.processingContainer.visibility = View.VISIBLE
        binding.bottomDock.visibility = View.GONE

        // Play original raw video immediately while background inpainting runs!
        binding.videoView.setVideoPath(rawFile.absolutePath)
        binding.videoView.start()

        lifecycleScope.launch {
            try {
                val yoloService = YoloServiceImpl(YoloTFLiteEngine(this@VideoPreviewActivity))
                val inpaintingEngine = MiGanInference(this@VideoPreviewActivity)
                yoloService.initialize("yolo11s")
                inpaintingEngine.initialize()

                val processor = VideoInpaintProcessor(yoloService, inpaintingEngine)
                val outputDir = File(cacheDir, "processed_videos").apply { mkdirs() }
                val outFile = File(outputDir, "carfree_${System.currentTimeMillis()}.mp4")

                val result = processor.processVideo(
                    inputFile = rawFile,
                    outputFile = outFile,
                    maxFps = 15
                ) { stage, percent ->
                    runOnUiThread {
                        binding.progressBar.progress = percent
                        binding.tvProgressStage.text = stage
                        binding.tvProgressPercent.text = "$percent%"
                    }
                }

                processedVideoFile = result.outputFile
                isShowingProcessed = true

                binding.processingContainer.visibility = View.GONE
                binding.bottomDock.visibility = View.VISIBLE
                binding.btnToggleSource.text = getString(R.string.video_btn_car_free)

                // Switch player to processed car-free video
                binding.videoView.setVideoPath(result.outputFile.absolutePath)
                binding.videoView.start()

                Snackbar.make(binding.root, getString(R.string.video_msg_ready), Snackbar.LENGTH_SHORT).show()

                yoloService.close()
                inpaintingEngine.close()
            } catch (e: Exception) {
                AppLogger.error("Video inpainting failed", e)
                binding.processingContainer.visibility = View.GONE
                binding.bottomDock.visibility = View.VISIBLE
                Snackbar.make(binding.root, getString(R.string.video_error_inpainting, e.message), Snackbar.LENGTH_LONG).show()
            }
        }
    }

    private fun toggleVideoSource() {
        val currentPosition = binding.videoView.currentPosition
        if (isShowingProcessed) {
            val rawPath = rawVideoPath ?: return
            binding.videoView.setVideoPath(rawPath)
            binding.videoView.seekTo(currentPosition)
            binding.videoView.start()
            binding.btnToggleSource.text = getString(R.string.video_btn_before)
            isShowingProcessed = false
        } else {
            val proc = processedVideoFile ?: return
            binding.videoView.setVideoPath(proc.absolutePath)
            binding.videoView.seekTo(currentPosition)
            binding.videoView.start()
            binding.btnToggleSource.text = getString(R.string.video_btn_car_free)
            isShowingProcessed = true
        }
    }

    private fun saveVideoToGallery() {
        val targetFile = if (isShowingProcessed) processedVideoFile else rawVideoPath?.let { File(it) }
        if (targetFile == null || !targetFile.exists()) {
            Snackbar.make(binding.root, getString(R.string.video_error_none_to_save), Snackbar.LENGTH_SHORT).show()
            return
        }

        lifecycleScope.launch(Dispatchers.IO) {
            try {
                val filename = "AutoKorrektur_${System.currentTimeMillis()}.mp4"
                val contentValues = ContentValues().apply {
                    put(MediaStore.Video.Media.DISPLAY_NAME, filename)
                    put(MediaStore.Video.Media.MIME_TYPE, "video/mp4")
                    if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
                        put(MediaStore.Video.Media.RELATIVE_PATH, Environment.DIRECTORY_MOVIES + "/AutoKorrektur")
                        put(MediaStore.Video.Media.IS_PENDING, 1)
                    }
                }

                val uri = contentResolver.insert(MediaStore.Video.Media.EXTERNAL_CONTENT_URI, contentValues)
                if (uri != null) {
                    contentResolver.openOutputStream(uri)?.use { out ->
                        FileInputStream(targetFile).use { input ->
                            input.copyTo(out)
                        }
                    }

                    if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
                        contentValues.clear()
                        contentValues.put(MediaStore.Video.Media.IS_PENDING, 0)
                        contentResolver.update(uri, contentValues, null, null)
                    }

                    withContext(Dispatchers.Main) {
                        Snackbar.make(binding.root, getString(R.string.video_msg_saved), Snackbar.LENGTH_LONG).show()
                    }
                }
            } catch (e: Exception) {
                AppLogger.error("Failed to save video to gallery", e)
                withContext(Dispatchers.Main) {
                    Snackbar.make(binding.root, getString(R.string.video_error_save), Snackbar.LENGTH_SHORT).show()
                }
            }
        }
    }

    private fun shareToInstagramReels() {
        val targetFile = processedVideoFile ?: rawVideoPath?.let { File(it) } ?: return
        try {
            val contentUri: Uri = FileProvider.getUriForFile(
                this,
                "${applicationContext.packageName}.fileprovider",
                targetFile
            )

            val shareIntent = Intent(Intent.ACTION_SEND).apply {
                type = "video/mp4"
                putExtra(Intent.EXTRA_STREAM, contentUri)
                addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION)
                setPackage("com.instagram.android")
            }

            if (shareIntent.resolveActivity(packageManager) != null) {
                startActivity(shareIntent)
            } else {
                val genericChooser = Intent.createChooser(
                    Intent(Intent.ACTION_SEND).apply {
                        type = "video/mp4"
                        putExtra(Intent.EXTRA_STREAM, contentUri)
                        addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION)
                    },
                    "Video teilen via"
                )
                startActivity(genericChooser)
            }
        } catch (e: Exception) {
            AppLogger.error("Failed to share video", e)
            Snackbar.make(binding.root, getString(R.string.video_error_share, e.message), Snackbar.LENGTH_SHORT).show()
        }
    }

    override fun onDestroy() {
        super.onDestroy()
        binding.videoView.stopPlayback()
    }

    companion object {
        const val EXTRA_RAW_VIDEO_PATH = "extra_raw_video_path"
    }
}
