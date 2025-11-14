package de.konradvoelkel.android.autokorrektur

import android.Manifest
import android.annotation.SuppressLint
import android.app.Activity
import android.app.AlertDialog
import android.content.ContentValues
import android.content.Intent
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.ImageDecoder
import android.graphics.Paint
import android.net.Uri
import android.os.Build
import android.os.Bundle
import android.os.Environment
import android.provider.MediaStore
import android.util.Log
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import android.widget.ArrayAdapter
import android.widget.ImageView
import android.widget.LinearLayout
import android.widget.SeekBar
import android.widget.TextView
import androidx.activity.result.contract.ActivityResultContracts
import androidx.core.content.ContextCompat
import androidx.core.content.FileProvider
import androidx.core.graphics.createBitmap
import androidx.core.view.isVisible
import androidx.fragment.app.Fragment
import androidx.fragment.app.viewModels
import androidx.lifecycle.lifecycleScope
import com.google.android.material.snackbar.Snackbar
import de.konradvoelkel.android.autokorrektur.databinding.FragmentFirstBinding
import de.konradvoelkel.android.autokorrektur.viewmodel.MainViewModel
import de.konradvoelkel.android.autokorrektur.viewmodel.MainViewModelFactory
import de.konradvoelkel.android.autokorrektur.ml.ImageProcessor
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import de.konradvoelkel.android.autokorrektur.ml.MiGanInference
import de.konradvoelkel.android.autokorrektur.ml.api.YoloService
import de.konradvoelkel.android.autokorrektur.ml.api.YoloServiceImpl
import de.konradvoelkel.android.autokorrektur.ml.config.YoloConfig
import de.konradvoelkel.android.autokorrektur.utils.AppLogger
import de.konradvoelkel.android.autokorrektur.utils.MaskOverlayUtils
import org.opencv.android.Utils
import org.opencv.core.Mat
import java.io.File
import java.io.FileOutputStream
import java.io.IOException
import java.io.OutputStream
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale

/**
 * Data class to store batch processing results for CSV export
 */
data class BatchProcessingResult(
    val originalImageName: String,
    val processingTimeMs: Long,
    val maskUpscale: Float,
    val scoreThreshold: Float,
    val downscaleMp: String,
    val segmentationModel: String,
    val success: Boolean,
    val errorMessage: String? = null
)

/**
 * Main fragment for the AutoKorrektur app, mimicking the web app functionality.
 */
@SuppressLint("DefaultLocale,SetTextI18n")
class FirstFragment : Fragment() {

    private var _binding: FragmentFirstBinding? = null
    private val binding get() = _binding!!

    private var selectedImageUri: Uri? = null
    private var selectedImageUris: MutableList<Uri> = mutableListOf()
    private var resultImageUri: Uri? = null
    private var processedBitmap: Bitmap? = null
    private var processedBitmaps: MutableList<Bitmap> = mutableListOf()
    private var photoFile: File? = null
    private var batchProcessingResults: MutableList<BatchProcessingResult> = mutableListOf()

    private val viewModel: MainViewModel by viewModels {
        MainViewModelFactory(
            ImageProcessor(requireContext()),
            YoloServiceImpl(requireContext()),
            MiGanInference(requireContext())
        )
    }

    // ML inference objects
    private lateinit var imageProcessor: ImageProcessor
    private lateinit var yoloInference: YoloService
    private lateinit var miGanInference: MiGanInference

    // Activity result launcher for image selection
    private val selectImageLauncher = registerForActivityResult(
        ActivityResultContracts.StartActivityForResult()
    ) { result ->
        AppLogger.debug("Gallery selection result received with code: ${result.resultCode}")

        when (result.resultCode) {
            Activity.RESULT_OK -> {
                val data = result.data
                if (data != null) {
                    val uri = data.data
                    if (uri != null) {
                        AppLogger.info("Gallery image selected successfully: $uri")
                        try {
                            selectedImageUri = uri
                            displayImage(uri, "Original")
                            binding.startInference.isEnabled = true
                            AppLogger.debug("Gallery image displayed successfully")
                        } catch (e: Exception) {
                            AppLogger.error("Error displaying selected gallery image", e)
                            Snackbar.make(
                                binding.root,
                                "Error displaying selected image: ${e.message}",
                                Snackbar.LENGTH_LONG
                            ).show()
                        }
                    } else {
                        AppLogger.error("Gallery selection returned null URI")
                        Snackbar.make(
                            binding.root,
                            "Failed to get image from gallery - no image data received",
                            Snackbar.LENGTH_LONG
                        ).show()
                    }
                } else {
                    AppLogger.error("Gallery selection returned null data")
                    Snackbar.make(
                        binding.root,
                        "Failed to get image from gallery - no data received",
                        Snackbar.LENGTH_LONG
                    ).show()
                }
            }

            Activity.RESULT_CANCELED -> {
                AppLogger.info("Gallery selection was canceled by user")
            }

            else -> {
                AppLogger.error("Gallery selection failed with result code: ${result.resultCode}")
                Snackbar.make(
                    binding.root,
                    "Failed to select image from gallery",
                    Snackbar.LENGTH_LONG
                ).show()
            }
        }
    }

    // Activity result launcher for camera
    private val takePictureLauncher = registerForActivityResult(
        ActivityResultContracts.StartActivityForResult()
    ) { result ->
        if (result.resultCode == Activity.RESULT_OK) {
            selectedImageUri?.let { uri ->
                displayImage(uri, "Original")
                binding.startInference.isEnabled = true
            }
        }
    }

    // Permission request launcher for camera
    private val cameraPermissionLauncher = registerForActivityResult(
        ActivityResultContracts.RequestPermission()
    ) { isGranted ->
        if (isGranted) {
            launchCamera()
        } else {
            AppLogger.warn("Camera permission denied by user")
            Snackbar.make(
                binding.root,
                "Camera permission is required to take photos",
                Snackbar.LENGTH_LONG
            ).show()
        }
    }

    // Permission request launcher for storage
    private val storagePermissionLauncher = registerForActivityResult(
        ActivityResultContracts.RequestPermission()
    ) { isGranted ->
        if (isGranted) {
            launchGallery()
        } else {
            AppLogger.warn("Storage permission denied by user")
            Snackbar.make(
                binding.root,
                "Storage permission is required to select photos",
                Snackbar.LENGTH_LONG
            ).show()
        }
    }

    // Multiple image picker launcher for batch processing
    private val multipleImagePickerLauncher = registerForActivityResult(
        ActivityResultContracts.GetMultipleContents()
    ) { uris ->
        if (uris.isNotEmpty()) {
            selectedImageUris.clear()
            selectedImageUris.addAll(uris)
            AppLogger.info("Selected ${uris.size} images for batch processing")

            // Display first few images as preview
            clearImagesContainer()
            uris.take(3).forEachIndexed { index, uri ->
                displayImage(uri, "Image ${index + 1}")
            }

            if (uris.size > 3) {
                Snackbar.make(
                    binding.root,
                    "Selected ${uris.size} images. Showing first 3 as preview.",
                    Snackbar.LENGTH_LONG
                ).show()
            }

            binding.startInference.isEnabled = true
            binding.startInference.text =
                getString(R.string.start_batch_processing_images, uris.size)
        } else {
            AppLogger.info("No images selected for batch processing")
        }
    }

    override fun onCreateView(
        inflater: LayoutInflater, container: ViewGroup?,
        savedInstanceState: Bundle?
    ): View {
        _binding = FragmentFirstBinding.inflate(inflater, container, false)
        return binding.root
    }

    companion object {
        private const val TAG = "FirstFragment"

        init {
            // This block is called only once when the class is first loaded.
            // It's the recommended place to load native libraries.
            try {
                // The modern and correct way to load the OpenCV library.
                System.loadLibrary("opencv_java4")
                // You can use your AppLogger here if it's accessible statically.
                Log.d(TAG, "OpenCV native library loaded successfully.")
            } catch (e: UnsatisfiedLinkError) {
                // This will catch errors if the .so file is missing.
                Log.e(TAG, "Failed to load OpenCV native library!", e)
                // Handle the error appropriately. Maybe show a dialog to the user.
            }
        }
    }

    override fun onViewCreated(view: View, savedInstanceState: Bundle?) {
        super.onViewCreated(view, savedInstanceState)

        // The OpenCV initialization is now handled in the companion object's init block.
        // You can remove the old try-catch block for OpenCVLoader.initDebug().
        AppLogger.debug("View created. OpenCV should be loaded.")

        // Initialize ML inference objects
        try {
            AppLogger.debug("Creating ML inference objects")
            imageProcessor = ImageProcessor(requireContext())
            yoloInference = YoloServiceImpl(requireContext())
            miGanInference = MiGanInference(requireContext())
            AppLogger.info("ML inference objects created successfully")
        } catch (e: Exception) {
            AppLogger.error("Failed to create ML inference objects", e)
            Snackbar.make(
                binding.root,
                "Failed to initialize ML components: ${e.message}",
                Snackbar.LENGTH_LONG
            ).show()
        }

        setupUI()
        observeViewModel()
    }

    private fun observeViewModel() {
        viewModel.uiState.observe(viewLifecycleOwner) { state ->
            when (state) {
                is de.konradvoelkel.android.autokorrektur.viewmodel.UiState.Success -> {
                    processedBitmap = state.processedBitmap
                    val tempFile = File(requireContext().cacheDir, "processed_image.jpg")
                    val outputStream = java.io.FileOutputStream(tempFile)
                    processedBitmap?.compress(Bitmap.CompressFormat.JPEG, 100, outputStream)
                    outputStream.close()
                    resultImageUri = FileProvider.getUriForFile(
                        requireContext(),
                        "${requireContext().packageName}.fileprovider",
                        tempFile
                    )
                    resultImageUri?.let { resultUri ->
                        addImageToContainer(resultUri, "Result (ONNX Processed)")
                    }
                    val maskBitmap = state.maskBitmap
                    val tempMaskFile = File(requireContext().cacheDir, "mask_image.jpg")
                    val maskOutputStream = java.io.FileOutputStream(tempMaskFile)
                    maskBitmap.compress(Bitmap.CompressFormat.JPEG, 100, maskOutputStream)
                    maskOutputStream.close()
                    val maskUri = FileProvider.getUriForFile(
                        requireContext(),
                        "${requireContext().packageName}.fileprovider",
                        tempMaskFile
                    )
                    displayImage(maskUri, "Mask Processed")
                    createMaskOverlay(selectedImageUri!!, state.maskBitmap)
                }

                is de.konradvoelkel.android.autokorrektur.viewmodel.UiState.Error -> {
                    Snackbar.make(binding.root, state.message, Snackbar.LENGTH_LONG).show()
                }
            }
        }

        viewModel.processing.observe(viewLifecycleOwner) { processing ->
            binding.startInference.isEnabled = !processing
            binding.startInference.text = if (processing) {
                getString(R.string.processing)
            } else {
                getString(R.string.start)
            }
        }

        viewModel.batchUiState.observe(viewLifecycleOwner) { state ->
            when (state) {
                is de.konradvoelkel.android.autokorrektur.viewmodel.BatchUiState.Progress -> {
                    binding.startInference.text =
                        "Processing batch (${state.progress}/${state.total})..."
                }

                is de.konradvoelkel.android.autokorrektur.viewmodel.BatchUiState.Success -> {
                    processedBitmaps.clear()
                    processedBitmaps.addAll(state.results)
                    finalizeBatchProcessing()
                }

                is de.konradvoelkel.android.autokorrektur.viewmodel.BatchUiState.Error -> {
                    Snackbar.make(binding.root, state.message, Snackbar.LENGTH_LONG).show()
                }
            }
        }
    }

    private fun setupUI() {
        // Setup file select button
        binding.fileSelect.setOnClickListener {
            if (binding.batchMode.isChecked) {
                // Launch multiple image picker for batch processing
                AppLogger.info("Launching multiple image selection for batch processing")
                multipleImagePickerLauncher.launch("image/*")
            } else {
                selectImage()
            }
        }

        // Setup start inference button
        binding.startInference.setOnClickListener {
            if (binding.batchMode.isChecked) {
                performBatchProcessing()
            } else {
                performOnnxInference()
            }
        }

        // Setup download button
        binding.download.setOnClickListener {
            processedBitmap?.let { bitmap ->
                // Save the processed image to gallery
                val savedUri = saveImageToGallery(bitmap)
                if (savedUri != null) {
                    Snackbar.make(binding.root, "Image saved to gallery", Snackbar.LENGTH_SHORT)
                        .show()
                }
            } ?: run {
                AppLogger.warn("Download attempted but no processed image available")
                Snackbar.make(
                    binding.root,
                    "No processed image to download. Run inference first.",
                    Snackbar.LENGTH_SHORT
                ).show()
            }
        }

        // Setup options button
        binding.optionsButton.setOnClickListener {
            toggleOptionsPanel()
        }

        // Setup sliders
        setupSliders()

        // Setup spinners
        setupSpinners()

        // Setup switches
        binding.batchMode.setOnCheckedChangeListener { _, isChecked ->
            if (isChecked) {
                // Batch mode enabled
                binding.startInference.isEnabled = selectedImageUris.isNotEmpty()
                binding.startInference.text = if (selectedImageUris.isNotEmpty()) {
                    "Start Batch Processing (${selectedImageUris.size} images)"
                } else {
                    "Start Batch Processing"
                }
                binding.fileSelect.text = getString(R.string.select_multiple_images)
            } else {
                // Single mode enabled
                binding.startInference.isEnabled = selectedImageUri != null
                binding.startInference.text = getString(R.string.start)
                binding.fileSelect.text = getString(R.string.select_image)
                // Clear batch selections when switching to single mode
                selectedImageUris.clear()
            }
        }

        // Setup continue mode switch
        binding.continueWithResult.setOnCheckedChangeListener { _, isChecked ->
            if (isChecked && processedBitmap == null && !binding.batchMode.isChecked) {
                Snackbar.make(
                    binding.root,
                    "No previous result available. Process an image first to enable continue mode.",
                    Snackbar.LENGTH_LONG
                ).show()
                binding.continueWithResult.isChecked = false
            }
        }
    }

    private fun setupSliders() {
        // Mask Upscale slider
        binding.maskUpscale.setOnSeekBarChangeListener(object : SeekBar.OnSeekBarChangeListener {

            override fun onProgressChanged(seekBar: SeekBar?, progress: Int, fromUser: Boolean) {
                val value = (1 + progress * 0.01).toFloat()
                binding.maskUpscaleVal.text = String.format("%.2f", value)
            }

            override fun onStartTrackingTouch(seekBar: SeekBar?) {}

            override fun onStopTrackingTouch(seekBar: SeekBar?) {}
        })


        // Score Threshold slider
        binding.scoreThreshold.setOnSeekBarChangeListener(object : SeekBar.OnSeekBarChangeListener {
            override fun onProgressChanged(seekBar: SeekBar?, progress: Int, fromUser: Boolean) {
                val value = progress * 0.01
                binding.scoreThresholdVal.text = String.format("%.2f", value)
            }

            override fun onStartTrackingTouch(seekBar: SeekBar?) {}

            override fun onStopTrackingTouch(seekBar: SeekBar?) {}
        })
    }

    private fun setupSpinners() {
        // Downscale spinner
        val downscaleOptions = arrayOf(
            "No Scaling", "0.5 MP", "1 MP", "2 MP", "3 MP",
            "4 MP", "5 MP", "6 MP", "7 MP", "8 MP", "9 MP", "10 MP"
        )
        val downscaleAdapter = ArrayAdapter(
            requireContext(),
            android.R.layout.simple_spinner_item,
            downscaleOptions
        )
        downscaleAdapter.setDropDownViewResource(android.R.layout.simple_spinner_dropdown_item)
        binding.downscaleMP.adapter = downscaleAdapter

        // Segmentation Model spinner
        val segModelOptions = arrayOf("Yolo11-Nano", "Yolo11-Small", "Yolo11-Medium")
        val segModelAdapter = ArrayAdapter(
            requireContext(),
            android.R.layout.simple_spinner_item,
            segModelOptions
        )
        segModelAdapter.setDropDownViewResource(android.R.layout.simple_spinner_dropdown_item)
        binding.segModel.adapter = segModelAdapter
        binding.segModel.setSelection(1) // Default to Yolo11-Small
    }

    private fun selectImage() {
        val options = arrayOf(
            getString(R.string.take_photo),
            getString(R.string.choose_from_gallery),
            getString(R.string.cancel)
        )

        AlertDialog.Builder(requireContext())
            .setTitle(R.string.photo_selection_title)
            .setItems(options) { _, which ->
                when (which) {
                    0 -> takePhoto()
                    1 -> chooseFromGallery()
                    // Cancel does nothing
                }
            }
            .show()
    }

    private fun takePhoto() {
        when {
            ContextCompat.checkSelfPermission(
                requireContext(),
                Manifest.permission.CAMERA
            ) == PackageManager.PERMISSION_GRANTED -> {
                launchCamera()
            }

            else -> {
                cameraPermissionLauncher.launch(Manifest.permission.CAMERA)
            }
        }
    }

    private fun launchCamera() {
        try {
            photoFile = createImageFile()
            photoFile?.also {
                selectedImageUri = FileProvider.getUriForFile(
                    requireContext(),
                    "${requireContext().packageName}.fileprovider",
                    it
                )
                val takePictureIntent = Intent(MediaStore.ACTION_IMAGE_CAPTURE)
                takePictureIntent.putExtra(MediaStore.EXTRA_OUTPUT, selectedImageUri)
                takePictureLauncher.launch(takePictureIntent)
            }
        } catch (ex: Exception) {
            AppLogger.error("Error creating image file for camera", ex)
            Snackbar.make(
                binding.root,
                "Error creating image file: ${ex.message}",
                Snackbar.LENGTH_LONG
            ).show()
        }
    }

    private fun chooseFromGallery() {
        val readPermission = if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU) {
            Manifest.permission.READ_MEDIA_IMAGES
        } else {
            Manifest.permission.READ_EXTERNAL_STORAGE
        }

        when {
            ContextCompat.checkSelfPermission(
                requireContext(),
                readPermission
            ) == PackageManager.PERMISSION_GRANTED -> {
                launchGallery()
            }

            else -> {
                storagePermissionLauncher.launch(readPermission)
            }
        }
    }

    private fun launchGallery() {
        try {
            AppLogger.debug("Launching gallery picker")
            val intent = Intent(Intent.ACTION_GET_CONTENT).apply {
                type = "image/*"
                addCategory(Intent.CATEGORY_OPENABLE)
                addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION)
            }
            selectImageLauncher.launch(intent)
            AppLogger.debug("Gallery picker launched successfully")
        } catch (e: Exception) {
            AppLogger.error("Error launching gallery picker", e)
            Snackbar.make(
                binding.root,
                "Error opening gallery: ${e.message}",
                Snackbar.LENGTH_LONG
            ).show()
        }
    }

    private fun createImageFile(): File {
        val timeStamp = SimpleDateFormat("yyyyMMdd_HHmmss", Locale.getDefault()).format(Date())
        val storageDir = requireContext().getExternalFilesDir(Environment.DIRECTORY_PICTURES)
        return File.createTempFile(
            "JPEG_${timeStamp}_",
            ".jpg",
            storageDir
        )
    }

    // Decode a downscaled bitmap for preview display to avoid Canvas too-large issues
    private fun decodePreviewBitmap(uri: Uri, targetW: Int, targetH: Int): Bitmap {
        val source = ImageDecoder.createSource(requireContext().contentResolver, uri)
        return ImageDecoder.decodeBitmap(source) { decoder, info, _ ->
            val srcW = info.size.width
            val srcH = info.size.height
            // Compute scale to fit into target box while preserving aspect ratio
            val scale = kotlin.math.min(
                targetW.toFloat() / srcW.toFloat(),
                targetH.toFloat() / srcH.toFloat()
            ).coerceAtMost(1f)
            val outW = kotlin.math.max(1, (srcW * scale).toInt())
            val outH = kotlin.math.max(1, (srcH * scale).toInt())
            // Prefer software allocation to reduce GPU memory pressure for previews
            decoder.allocator = ImageDecoder.ALLOCATOR_SOFTWARE
            decoder.isMutableRequired = false
            decoder.setTargetSize(outW, outH)
        }
    }

    private fun displayImage(uri: Uri, label: String) {
        clearImagesContainer()

        val imageView = ImageView(context)
        imageView.layoutParams = LinearLayout.LayoutParams(
            LinearLayout.LayoutParams.MATCH_PARENT,
            400
        )
        imageView.scaleType = ImageView.ScaleType.FIT_CENTER

        // Decode a downscaled preview to avoid Canvas too-large crashes
        val metrics = resources.displayMetrics
        val targetW = metrics.widthPixels
        val targetH = 400 // matches the fixed layout height in px
        val preview = decodePreviewBitmap(uri, targetW, targetH)
        imageView.setImageBitmap(preview)

        val textView = TextView(context)
        textView.text = label
        textView.textAlignment = View.TEXT_ALIGNMENT_CENTER

        val container = LinearLayout(context)
        container.orientation = LinearLayout.VERTICAL
        container.layoutParams = LinearLayout.LayoutParams(
            LinearLayout.LayoutParams.MATCH_PARENT,
            LinearLayout.LayoutParams.WRAP_CONTENT
        )
        container.addView(imageView)
        container.addView(textView)

        binding.imagesContainer.addView(container)
    }

    private fun addImageToContainer(uri: Uri, @Suppress("SameParameterValue") label: String) {
        val imageView = ImageView(context)
        imageView.layoutParams = LinearLayout.LayoutParams(
            LinearLayout.LayoutParams.MATCH_PARENT,
            400
        )
        imageView.scaleType = ImageView.ScaleType.FIT_CENTER

        val metrics = resources.displayMetrics
        val targetW = metrics.widthPixels
        val targetH = 400
        val preview = decodePreviewBitmap(uri, targetW, targetH)
        imageView.setImageBitmap(preview)

        val textView = TextView(context)
        textView.text = label
        textView.textAlignment = View.TEXT_ALIGNMENT_CENTER

        val container = LinearLayout(context)
        container.orientation = LinearLayout.VERTICAL
        container.layoutParams = LinearLayout.LayoutParams(
            LinearLayout.LayoutParams.MATCH_PARENT,
            LinearLayout.LayoutParams.WRAP_CONTENT
        )
        container.addView(imageView)
        container.addView(textView)

        binding.imagesContainer.addView(container)
    }

    private fun clearImagesContainer() {
        binding.imagesContainer.removeAllViews()
    }

    private fun toggleOptionsPanel() {
        if (binding.optionsPanel.isVisible) {
            binding.optionsPanel.visibility = View.GONE
        } else {
            binding.optionsPanel.visibility = View.VISIBLE
        }
    }

    private fun performOnnxInference() {
        AppLogger.info("Starting ONNX inference")

        // Check if an image is selected for single processing
        if (selectedImageUri == null) {
            AppLogger.warn("No image selected for inference")
            Snackbar.make(binding.root, "Please select an image first", Snackbar.LENGTH_SHORT)
                .show()
            return
        }
        viewModel.performOnnxInference(
            selectedImageUri = selectedImageUri,
            resultImageUri = resultImageUri,
            continueWithResult = binding.continueWithResult.isChecked,
            segModel = binding.segModel.selectedItem.toString().lowercase(),
            downscaleMp = getDownscaleMpFromSpinner(),
            maskUpscale = getMaskUpscaleFromSlider(),
            scoreThreshold = getScoreThresholdFromSlider()
        )
    }

    private fun performBatchProcessing() {
        AppLogger.info("Starting batch processing for ${selectedImageUris.size} images")

        // Clear previous results
        batchProcessingResults.clear()
        processedBitmaps.clear()

        // Disable UI and show processing state
        binding.startInference.isEnabled = false
        binding.fileSelect.isEnabled = false
        binding.batchMode.isEnabled = false

        // Clear the images container
        clearImagesContainer()

        viewModel.performBatchProcessing(
            uris = selectedImageUris,
            segModel = binding.segModel.selectedItem.toString(),
            downscaleMp = getDownscaleMpFromSpinner(),
            maskUpscale = getMaskUpscaleFromSlider(),
            scoreThreshold = getScoreThresholdFromSlider()
        )
    }

    private fun finalizeBatchProcessing() {
        val successCount = batchProcessingResults.count { it.success }
        val totalCount = batchProcessingResults.size

        AppLogger.info("Batch processing completed: $successCount/$totalCount images processed successfully")

        // Re-enable UI
        binding.startInference.isEnabled = true
        binding.startInference.text = "Start Batch Processing (${selectedImageUris.size} images)"
        binding.fileSelect.isEnabled = true
        binding.batchMode.isEnabled = true

        // Show results summary
        val message =
            "Batch processing completed!\n$successCount/$totalCount images processed successfully"
        Snackbar.make(binding.root, message, Snackbar.LENGTH_LONG)
            .setAction("Export CSV") {
                exportBatchResultsToCSV()
            }.show()

        // Display first few processed images
        processedBitmaps.take(3).forEachIndexed { index, bitmap ->
            try {
                val tempFile = File(requireContext().cacheDir, "batch_result_${index}.jpg")
                val outputStream = FileOutputStream(tempFile)
                bitmap.compress(Bitmap.CompressFormat.JPEG, 100, outputStream)
                outputStream.close()

                val resultUri = FileProvider.getUriForFile(
                    requireContext(),
                    "${requireContext().packageName}.fileprovider",
                    tempFile
                )
                displayImage(resultUri, "Result ${index + 1}")
            } catch (e: Exception) {
                AppLogger.error("Error displaying batch result ${index + 1}", e)
            }
        }
    }

    private fun exportBatchResultsToCSV() {
        if (batchProcessingResults.isEmpty()) {
            Snackbar.make(binding.root, "No batch results to export", Snackbar.LENGTH_SHORT).show()
            return
        }

        try {
            val csvContent = StringBuilder()
            csvContent.append("Image Name,Processing Time (ms),Mask Upscale,Score Threshold,Downscale MP,Segmentation Model,Success,Error Message\n")

            batchProcessingResults.forEach { result ->
                csvContent.append("${result.originalImageName},")
                csvContent.append("${result.processingTimeMs},")
                csvContent.append("${result.maskUpscale},")
                csvContent.append("${result.scoreThreshold},")
                csvContent.append("${result.downscaleMp},")
                csvContent.append("${result.segmentationModel},")
                csvContent.append("${result.success},")
                csvContent.append("${result.errorMessage ?: ""}\n")
            }

            val timestamp = SimpleDateFormat("yyyyMMdd_HHmmss", Locale.getDefault()).format(Date())
            val fileName = "autokorrektur_batch_results_$timestamp.csv"

            val csvFile = File(
                requireContext().getExternalFilesDir(Environment.DIRECTORY_DOCUMENTS),
                fileName
            )
            csvFile.writeText(csvContent.toString())

            AppLogger.info("CSV exported to: ${csvFile.absolutePath}")
            Snackbar.make(binding.root, "CSV exported to: ${csvFile.name}", Snackbar.LENGTH_LONG)
                .show()

        } catch (e: Exception) {
            AppLogger.error("Failed to export CSV", e)
            Snackbar.make(binding.root, "Failed to export CSV: ${e.message}", Snackbar.LENGTH_LONG)
                .show()
        }
    }

    private fun createMaskOverlay(originalUri: Uri, maskBitmap: Bitmap) {
        try {
            AppLogger.debug("Creating mask overlay visualization")

            // Load original image as bitmap using the recommended ImageDecoder
            val source = ImageDecoder.createSource(requireContext().contentResolver, originalUri)
            val originalBitmap = ImageDecoder.decodeBitmap(source)

            // Create a mutable copy of the original bitmap
            val overlayBitmap = originalBitmap.copy(Bitmap.Config.ARGB_8888, true)

            // Build an overlay that is transparent everywhere except masked area (car)
            val overlayMaskBitmap = MaskOverlayUtils
                .createRedOverlayBitmap(
                    maskBitmap,
                    overlayBitmap.width,
                    overlayBitmap.height,
                    threshold = 128,
                    alpha = 128
                )

            // Draw the overlay onto the original image
            val overlayCanvas = Canvas(overlayBitmap)
            val paint = Paint(Paint.ANTI_ALIAS_FLAG)
            overlayCanvas.drawBitmap(overlayMaskBitmap, 0f, 0f, paint)

            // Recycle temporary bitmaps
            overlayMaskBitmap.recycle()
            maskBitmap.recycle()

            // Save overlay image to temporary file
            val tempOverlayFile = File(requireContext().cacheDir, "mask_overlay.jpg")
            val overlayOutputStream = FileOutputStream(tempOverlayFile)
            overlayBitmap.compress(Bitmap.CompressFormat.JPEG, 100, overlayOutputStream)
            overlayOutputStream.close()

            // Get URI for the overlay image
            val overlayUri = FileProvider.getUriForFile(
                requireContext(),
                "${requireContext().packageName}.fileprovider",
                tempOverlayFile
            )

            // Display the overlay image
            displayImage(overlayUri, "Mask Overlay")

            AppLogger.debug("Mask overlay created and displayed successfully")

        } catch (e: Exception) {
            AppLogger.error("Error creating mask overlay", e)
            // Don't show error to user as this is an additional feature
        }
    }

    /**
     * Gets the downscale megapixels value from the spinner.
     */
    private fun getDownscaleMpFromSpinner(): Float? {
        val selectedItem = binding.downscaleMP.selectedItem.toString()
        return when (selectedItem) {
            "No Scaling" -> null
            "0.5 MP" -> 0.5f
            "1 MP" -> 1.0f
            "2 MP" -> 2.0f
            "3 MP" -> 3.0f
            "4 MP" -> 4.0f
            "5 MP" -> 5.0f
            "6 MP" -> 6.0f
            "7 MP" -> 7.0f
            "8 MP" -> 8.0f
            "9 MP" -> 9.0f
            "10 MP" -> 10.0f
            else -> null
        }
    }

    /**
     * Gets the mask upscale factor from the slider.
     */
    private fun getMaskUpscaleFromSlider(): Float {
        return (1 + binding.maskUpscale.progress * 0.01).toFloat()
    }

    /**
     * Gets the score threshold from the slider.
     */
    private fun getScoreThresholdFromSlider(): Float {
        return (binding.scoreThreshold.progress * 0.01).toFloat()
    }

    /**
     * Gets the downshift factor from the slider.
     */
    private fun getDownshiftFromSlider(): Float {
        return 0.0f
    }

    /**
     * Saves the processed bitmap to the gallery
     */
    private fun saveImageToGallery(bitmap: Bitmap): Uri? {
        try {
            val filename = "AutoKorrektur_${System.currentTimeMillis()}.jpg"
            var fos: OutputStream?
            var imageUri: Uri?

            // For Android 10 (Q) and above
            val contentValues = ContentValues().apply {
                put(MediaStore.MediaColumns.DISPLAY_NAME, filename)
                put(MediaStore.MediaColumns.MIME_TYPE, "image/jpeg")
                put(MediaStore.MediaColumns.RELATIVE_PATH, Environment.DIRECTORY_PICTURES)
            }

            val contentResolver = requireContext().contentResolver
            imageUri = contentResolver.insert(
                MediaStore.Images.Media.EXTERNAL_CONTENT_URI,
                contentValues
            )
            fos = imageUri?.let { contentResolver.openOutputStream(it) }

            fos?.use {
                bitmap.compress(Bitmap.CompressFormat.JPEG, 90, it)
                AppLogger.info("Image saved to gallery successfully")
            }

            return imageUri
        } catch (e: Exception) {
            AppLogger.error("Error saving image to gallery", e)
            Snackbar.make(
                binding.root,
                "Error saving image: ${e.message}",
                Snackbar.LENGTH_LONG
            ).show()
            return null
        }
    }

    override fun onDestroyView() {
        super.onDestroyView()

        // Clean up ML inference objects
        if (::yoloInference.isInitialized) {
            yoloInference.close()
        }
        if (::miGanInference.isInitialized) {
            miGanInference.close()
        }

        _binding = null
    }
}
