package de.konradvoelkel.android.autokorrektur

import android.Manifest
import android.app.Activity
import android.content.ContentValues
import android.content.Intent
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.net.Uri
import android.os.Build
import android.os.Bundle
import android.os.Environment
import android.provider.MediaStore
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
import androidx.fragment.app.Fragment
import com.google.android.material.snackbar.Snackbar
import de.konradvoelkel.android.autokorrektur.databinding.FragmentFirstBinding
import de.konradvoelkel.android.autokorrektur.ml.ImageProcessor
import de.konradvoelkel.android.autokorrektur.ml.YoloInferenceTFLite
import de.konradvoelkel.android.autokorrektur.ml.MiGanInference
import de.konradvoelkel.android.autokorrektur.utils.AppLogger
import org.opencv.android.Utils
import java.io.File
import java.io.FileOutputStream
import java.io.IOException
import java.io.OutputStream
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale
import androidx.core.view.isVisible
import androidx.core.graphics.createBitmap
import org.opencv.core.Mat

/**
 * Data class to store batch processing results for CSV export
 */
data class BatchProcessingResult(
    val originalImageName: String,
    val processingTimeMs: Long,
    val maskUpscale: Float,
    val scoreThreshold: Float,
    val downshift: Float,
    val downscaleMp: String,
    val segmentationModel: String,
    val success: Boolean,
    val errorMessage: String? = null
)

/**
 * Main fragment for the AutoKorrektur app, mimicking the web app functionality.
 */
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
    private var isProcessingBatch = false

    // ML inference objects
    private lateinit var imageProcessor: ImageProcessor
    private lateinit var yoloInference: YoloInferenceTFLite
    private lateinit var miGanInference: MiGanInference
    private var mlComponentsInitialized = false

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
            binding.startInference.text = "Start Batch Processing (${uris.size} images)"
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

    override fun onViewCreated(view: View, savedInstanceState: Bundle?) {
        super.onViewCreated(view, savedInstanceState)

        // Check OpenCV initialization
        try {
            AppLogger.debug("Checking OpenCV initialization")
            if (!org.opencv.android.OpenCVLoader.initDebug()) {
                AppLogger.error("OpenCV initialization failed")
                Snackbar.make(binding.root, "OpenCV initialization failed. Some features may not work.", Snackbar.LENGTH_LONG).show()
            } else {
                AppLogger.info("OpenCV initialized successfully")
            }
        } catch (e: Exception) {
            AppLogger.error("OpenCV initialization check failed", e)
            // Continue anyway, as OpenCV might be statically linked
        }

        // Initialize ML inference objects
        try {
            AppLogger.debug("Creating ML inference objects")
            imageProcessor = ImageProcessor(requireContext())
            yoloInference = YoloInferenceTFLite(requireContext())
            miGanInference = MiGanInference(requireContext())
            mlComponentsInitialized = true
            AppLogger.info("ML inference objects created successfully")
        } catch (e: Exception) {
            AppLogger.error("Failed to create ML inference objects", e)
            mlComponentsInitialized = false
            Snackbar.make(binding.root, "Failed to initialize ML components: ${e.message}", Snackbar.LENGTH_LONG).show()
        }

        setupUI()
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
            performOnnxInference()
        }

        // Setup download button
        binding.download.setOnClickListener {
            processedBitmap?.let { bitmap ->
                // Save the processed image to gallery
                val savedUri = saveImageToGallery(bitmap)
                if (savedUri != null) {
                    Snackbar.make(binding.root, "Image saved to gallery", Snackbar.LENGTH_SHORT).show()
                }
            } ?: run {
                AppLogger.warn("Download attempted but no processed image available")
                Snackbar.make(binding.root, "No processed image to download. Run inference first.", Snackbar.LENGTH_SHORT).show()
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
                binding.fileSelect.text = "Select Multiple Images"
            } else {
                // Single mode enabled
                binding.startInference.isEnabled = selectedImageUri != null
                binding.startInference.text = "Start"
                binding.fileSelect.text = "Select Image"
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

        // Mask Downshift slider
        binding.downshift.setOnSeekBarChangeListener(object : SeekBar.OnSeekBarChangeListener {
            override fun onProgressChanged(seekBar: SeekBar?, progress: Int, fromUser: Boolean) {
                val value = progress * 0.001
                binding.downshiftVal.text = String.format("%.3f", value)
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

        android.app.AlertDialog.Builder(requireContext())
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
            val intent = Intent(Intent.ACTION_PICK, MediaStore.Images.Media.EXTERNAL_CONTENT_URI)
            intent.type = "image/*"
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

    private fun displayImage(uri: Uri, label: String) {
        clearImagesContainer()

        val imageView = ImageView(context)
        imageView.layoutParams = LinearLayout.LayoutParams(
            LinearLayout.LayoutParams.MATCH_PARENT,
            400
        )
        imageView.scaleType = ImageView.ScaleType.FIT_CENTER
        imageView.setImageURI(uri)

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

        // Check if ML components are initialized
        if (!mlComponentsInitialized) {
            AppLogger.error("ML components not initialized")
            Snackbar.make(binding.root, "ML components not initialized. Please restart the app.", Snackbar.LENGTH_LONG).show()
            return
        }

        // Route to batch or single processing based on mode
        if (binding.batchMode.isChecked) {
            if (selectedImageUris.isEmpty()) {
                AppLogger.warn("No images selected for batch processing")
                Snackbar.make(binding.root, "Please select images for batch processing first", Snackbar.LENGTH_SHORT).show()
                return
            }
            performBatchProcessing()
        } else {
            // Check if an image is selected for single processing
            if (selectedImageUri == null) {
                AppLogger.warn("No image selected for inference")
                Snackbar.make(binding.root, "Please select an image first", Snackbar.LENGTH_SHORT).show()
                return
            }
            performSingleImageInference()
        }
    }

    private fun performSingleImageInference() {

        // Disable the button and show processing state
        binding.startInference.isEnabled = false
        binding.startInference.text = "Processing..."

        // Clear the images container
        clearImagesContainer()

        // Display the original image or previous result based on continue mode
        val inputUri = if (binding.continueWithResult.isChecked && resultImageUri != null) {
            resultImageUri!!
        } else {
            selectedImageUri!!
        }

        val inputLabel = if (binding.continueWithResult.isChecked && resultImageUri != null) {
            "Previous Result (Input)"
        } else {
            "Original"
        }

        displayImage(inputUri, inputLabel)

        // Perform ONNX inference in a background thread
        Thread {
            try {
                AppLogger.debug("Starting background inference thread")

                selectedImageUri?.let { uri ->
                    AppLogger.debug("Processing image URI: $uri")

                    // Initialize ML inference objects if not already done
                    try {
                        AppLogger.debug("Checking if YOLO inference needs initialization")
                        yoloInference.initialize()
                        AppLogger.debug("YOLO inference initialized successfully")

                        AppLogger.debug("Checking if Mi-GAN inference needs initialization")
                        miGanInference.initialize()
                        AppLogger.debug("Mi-GAN inference initialized successfully")

                        AppLogger.info("All ML inference objects initialized successfully")
                    } catch (e: IOException) {
                        AppLogger.error("IOException during ML initialization", e)
                        throw Exception("Failed to load ML models from assets: ${e.message}", e)
                    } catch (e: RuntimeException) {
                        AppLogger.error("RuntimeException during ML initialization", e)
                        throw Exception("Runtime error during ML initialization: ${e.message}", e)
                    } catch (e: Exception) {
                        AppLogger.error("Unexpected exception during ML initialization", e)
                        throw Exception("Failed to initialize ML models: ${e.message}", e)
                    }

                    // Get UI parameters
                    val downscaleMp = getDownscaleMpFromSpinner()
                    val maskUpscale = getMaskUpscaleFromSlider()
                    val scoreThreshold = getScoreThresholdFromSlider()
                    val downshift = getDownshiftFromSlider()

                    AppLogger.debug("Parameters - downscaleMp: $downscaleMp, maskUpscale: $maskUpscale, scoreThreshold: $scoreThreshold, downshift: $downshift")

                    // Step 1: Process input image
                    AppLogger.debug("Step 1: Processing input image")
                    // Determine input URI based on continue mode
                    val processingUri = if (binding.continueWithResult.isChecked && resultImageUri != null) {
                        resultImageUri!!
                    } else {
                        uri
                    }

                    val processedImage = try {
                        imageProcessor.processInputImage(
                            uri = processingUri,
                            modelWidth = 640,  // YOLO model input width
                            modelHeight = 640, // YOLO model input height
                            downscaleMp = downscaleMp
                        )
                    } catch (e: Exception) {
                        AppLogger.error("Error processing input image", e)
                        throw Exception("Failed to process input image: ${e.message}", e)
                    }
                    AppLogger.debug("Input image processed successfully")

                    // Step 2: Run YOLO inference to get segmentation mask
                    AppLogger.debug("Step 2: Running YOLO inference")
                    val maskMat = try {
                        yoloInference.inferYolo(
                            transformedMat = processedImage.transformedMat,
                            xRatio = processedImage.xRatio,
                            yRatio = processedImage.yRatio,
                            modelWidth = 640,
                            modelHeight = 640,
                            upscaleFactor = maskUpscale,
                            scoreThreshold = scoreThreshold,
                            downshiftFactor = downshift
                        )
                    } catch (e: Exception) {
                        AppLogger.error("Error during YOLO inference", e)
                        throw Exception("YOLO inference failed: ${e.message}", e)
                    }
                    AppLogger.debug("YOLO inference completed successfully")

                    // Display the mask on UI thread
                    if (isAdded && !isDetached) {
                        requireActivity().runOnUiThread {
                            try {
                                if (!isAdded || isDetached) {
                                    AppLogger.warn("Fragment not attached, skipping mask display")
                                    return@runOnUiThread
                                }

                                AppLogger.debug("Creating mask bitmap")
                                val maskBitmap = createBitmap(maskMat.cols(), maskMat.rows())
                                Utils.matToBitmap(maskMat, maskBitmap)

                                val tempMaskFile = File(requireContext().cacheDir, "mask_image.jpg")
                                val maskOutputStream = FileOutputStream(tempMaskFile)
                                maskBitmap.compress(Bitmap.CompressFormat.JPEG, 100, maskOutputStream)
                                maskOutputStream.close()

                                val maskUri = FileProvider.getUriForFile(
                                    requireContext(),
                                    "${requireContext().packageName}.fileprovider",
                                    tempMaskFile
                                )
                                displayImage(maskUri, "Mask")

                                // Create and display mask overlay on original image
                                createMaskOverlay(processingUri, maskMat)

                                AppLogger.debug("Mask displayed successfully")
                            } catch (e: Exception) {
                                AppLogger.error("Error displaying mask", e)
                            }
                        }
                    } else {
                        AppLogger.warn("Fragment not attached, skipping mask display")
                    }

                    // Step 3: Run Mi-GAN inference for inpainting
                    AppLogger.debug("Step 3: Running Mi-GAN inference")
                    val resultMat = try {
                        miGanInference.inferMiGan(
                            imageMat = processedImage.transformedMat,
                            maskMat = maskMat
                        )
                    } catch (e: Exception) {
                        AppLogger.error("Error during Mi-GAN inference", e)
                        throw Exception("Mi-GAN inference failed: ${e.message}", e)
                    }
                    AppLogger.debug("Mi-GAN inference completed successfully")

                    // Convert result to bitmap and display on UI thread
                    if (isAdded && !isDetached) {
                        requireActivity().runOnUiThread {
                            try {
                                if (!isAdded || isDetached) {
                                    AppLogger.warn("Fragment not attached, skipping result display")
                                    return@runOnUiThread
                                }

                                AppLogger.debug("Creating result bitmap")
                                processedBitmap = createBitmap(resultMat.cols(), resultMat.rows())
                                Utils.matToBitmap(resultMat, processedBitmap!!)

                                // Create a temporary file to display the processed image
                                val tempFile = File(requireContext().cacheDir, "processed_image.jpg")
                                val outputStream = FileOutputStream(tempFile)
                                processedBitmap?.compress(Bitmap.CompressFormat.JPEG, 100, outputStream)
                                outputStream.close()

                                // Get URI for the processed image
                                resultImageUri = FileProvider.getUriForFile(
                                    requireContext(),
                                    "${requireContext().packageName}.fileprovider",
                                    tempFile
                                )

                                // Display the processed image
                                resultImageUri?.let { resultUri ->
                                    displayImage(resultUri, "Result (ONNX Processed)")
                                }

                                binding.startInference.isEnabled = true
                                binding.startInference.text = "Start"

                                Snackbar.make(binding.root, "ONNX inference completed successfully", Snackbar.LENGTH_SHORT).show()
                                AppLogger.info("Inference completed successfully")
                            } catch (e: Exception) {
                                AppLogger.error("Error displaying result", e)
                                binding.startInference.isEnabled = true
                                binding.startInference.text = "Start"
                                Snackbar.make(binding.root, "Error displaying result: ${e.message}", Snackbar.LENGTH_LONG).show()
                            }
                        }
                    } else {
                        AppLogger.warn("Fragment not attached, skipping result display")
                    }
                } ?: run {
                    AppLogger.error("selectedImageUri is null in background thread")
                    if (isAdded && !isDetached) {
                        requireActivity().runOnUiThread {
                            if (!isAdded || isDetached) {
                                AppLogger.warn("Fragment not attached, skipping null URI error display")
                                return@runOnUiThread
                            }
                            binding.startInference.isEnabled = true
                            binding.startInference.text = "Start"
                            Snackbar.make(binding.root, "No image selected", Snackbar.LENGTH_SHORT).show()
                        }
                    } else {
                        AppLogger.warn("Fragment not attached, skipping null URI error display")
                    }
                }
            } catch (e: Exception) {
                AppLogger.error("Error during inference", e)

                // Handle errors on UI thread
                if (isAdded && !isDetached) {
                    requireActivity().runOnUiThread {
                        if (!isAdded || isDetached) {
                            AppLogger.warn("Fragment not attached, skipping error display")
                            return@runOnUiThread
                        }

                        binding.startInference.isEnabled = true
                        binding.startInference.text = "Start"

                        val errorMessages = when {
                            e.message?.contains("Failed to create YOLO session") == true -> {
                                Pair("YOLO Model Loading Failed", "The YOLO model could not be loaded. This might be due to:\n• Corrupted model file\n• Insufficient memory\n• Incompatible model format\n\nCheck logcat for detailed error information.\n\nOriginal error: ${e.message}")
                            }
                            e.message?.contains("Failed to create NMS session") == true -> {
                                Pair("NMS Model Loading Failed", "The NMS model could not be loaded. Check logcat for details.\n\nOriginal error: ${e.message}")
                            }
                            e.message?.contains("Failed to create mask session") == true -> {
                                Pair("Mask Model Loading Failed", "The Mask model could not be loaded. Check logcat for details.\n\nOriginal error: ${e.message}")
                            }
                            e.message?.contains("model") == true -> {
                                Pair("Model Loading Failed", "One or more ML models failed to load. Check logcat for detailed error information.\n\nOriginal error: ${e.message}")
                            }
                            e.message?.contains("OpenCV") == true -> {
                                Pair("OpenCV Error", "OpenCV initialization failed. This might indicate a problem with image processing.\n\nOriginal error: ${e.message}")
                            }
                            e.message?.contains("ONNX") == true -> {
                                Pair("ONNX Runtime Error", "ONNX Runtime encountered an error during model execution.\n\nOriginal error: ${e.message}")
                            }
                            e.message?.contains("initialize") == true -> {
                                Pair("Initialization Failed", e.message ?: "Unknown initialization error")
                            }
                            else -> {
                                Pair("Inference Error", "An error occurred during inference processing.\n\nOriginal error: ${e.message}")
                            }
                        }

                        val shortMessage = errorMessages.first
                        val detailedMessage = errorMessages.second

                        // Log the error details and show user-friendly message
                        AppLogger.error("Inference error: $shortMessage - $detailedMessage")

                        // Snackbar for detailed information
                        Snackbar.make(
                            binding.root,
                            detailedMessage as CharSequence,
                            Snackbar.LENGTH_INDEFINITE
                        ).setAction("Dismiss") {
                            // Snackbar will be dismissed
                        }.show()
                    }
                } else {
                    AppLogger.warn("Fragment not attached, skipping error display")
                }
            }
        }.start()
    }

    private fun performBatchProcessing() {
        AppLogger.info("Starting batch processing for ${selectedImageUris.size} images")

        // Clear previous results
        batchProcessingResults.clear()
        processedBitmaps.clear()
        isProcessingBatch = true

        // Disable UI and show processing state
        binding.startInference.isEnabled = false
        binding.startInference.text = "Processing batch (0/${selectedImageUris.size})..."
        binding.fileSelect.isEnabled = false
        binding.batchMode.isEnabled = false

        // Clear the images container
        clearImagesContainer()

        // Start batch processing in background thread
        Thread {
            try {
                // Initialize ML inference objects
                try {
                    yoloInference.initialize()
                    miGanInference.initialize()
                    AppLogger.info("ML inference objects initialized for batch processing")
                } catch (e: Exception) {
                    AppLogger.error("Failed to initialize ML objects for batch processing", e)
                    throw Exception("Failed to initialize ML models: ${e.message}", e)
                }

                // Get UI parameters once for all images
                val downscaleMp = getDownscaleMpFromSpinner()
                val maskUpscale = getMaskUpscaleFromSlider()
                val scoreThreshold = getScoreThresholdFromSlider()
                val downshift = getDownshiftFromSlider()
                val segModel = binding.segModel.selectedItem.toString()

                AppLogger.debug("Batch parameters - downscaleMp: $downscaleMp, maskUpscale: $maskUpscale, scoreThreshold: $scoreThreshold, downshift: $downshift, segModel: $segModel")

                // Process each image
                selectedImageUris.forEachIndexed { index, uri ->
                    if (!isProcessingBatch) {
                        AppLogger.info("Batch processing cancelled")
                        return@Thread
                    }

                    val startTime = System.currentTimeMillis()
                    val imageName = "Image_${index + 1}"

                    // Update progress on UI thread
                    if (isAdded && !isDetached) {
                        requireActivity().runOnUiThread {
                            binding.startInference.text = "Processing batch (${index + 1}/${selectedImageUris.size})..."
                        }
                    }

                    try {
                        AppLogger.debug("Processing image ${index + 1}/${selectedImageUris.size}: $uri")

                        // Step 1: Process input image
                        val processedImage = imageProcessor.processInputImage(
                            uri = uri,
                            modelWidth = 640,
                            modelHeight = 640,
                            downscaleMp = downscaleMp
                        )

                        // Step 2: Run YOLO inference
                        val maskMat = yoloInference.inferYolo(
                            transformedMat = processedImage.transformedMat,
                            xRatio = processedImage.xRatio,
                            yRatio = processedImage.yRatio,
                            modelWidth = 640,
                            modelHeight = 640,
                            upscaleFactor = maskUpscale,
                            scoreThreshold = scoreThreshold,
                            downshiftFactor = downshift
                        )

                        // Step 3: Run Mi-GAN inference
                        val resultMat = miGanInference.inferMiGan(
                            imageMat = processedImage.transformedMat,
                            maskMat = maskMat
                        )

                        // Convert result to bitmap
                        val resultBitmap = createBitmap(resultMat.cols(), resultMat.rows())
                        Utils.matToBitmap(resultMat, resultBitmap)
                        processedBitmaps.add(resultBitmap)

                        val processingTime = System.currentTimeMillis() - startTime

                        // Record successful result
                        batchProcessingResults.add(
                            BatchProcessingResult(
                                originalImageName = imageName,
                                processingTimeMs = processingTime,
                                maskUpscale = maskUpscale,
                                scoreThreshold = scoreThreshold,
                                downshift = downshift,
                                downscaleMp = downscaleMp?.toString() ?: "No Scaling",
                                segmentationModel = segModel,
                                success = true
                            )
                        )

                        AppLogger.debug("Successfully processed image ${index + 1} in ${processingTime}ms")

                    } catch (e: Exception) {
                        val processingTime = System.currentTimeMillis() - startTime
                        AppLogger.error("Error processing image ${index + 1}: ${e.message}", e)

                        // Record failed result
                        batchProcessingResults.add(
                            BatchProcessingResult(
                                originalImageName = imageName,
                                processingTimeMs = processingTime,
                                maskUpscale = maskUpscale,
                                scoreThreshold = scoreThreshold,
                                downshift = downshift,
                                downscaleMp = downscaleMp?.toString() ?: "No Scaling",
                                segmentationModel = segModel,
                                success = false,
                                errorMessage = e.message
                            )
                        )
                    }
                }

                // Update UI on completion
                if (isAdded && !isDetached) {
                    requireActivity().runOnUiThread {
                        finalizeBatchProcessing()
                    }
                }

            } catch (e: Exception) {
                AppLogger.error("Batch processing failed", e)
                if (isAdded && !isDetached) {
                    requireActivity().runOnUiThread {
                        binding.startInference.isEnabled = true
                        binding.startInference.text = "Start Batch Processing"
                        binding.fileSelect.isEnabled = true
                        binding.batchMode.isEnabled = true
                        isProcessingBatch = false

                        Snackbar.make(
                            binding.root,
                            "Batch processing failed: ${e.message}",
                            Snackbar.LENGTH_LONG
                        ).show()
                    }
                }
            }
        }.start()
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
        isProcessingBatch = false

        // Show results summary
        val message = "Batch processing completed!\n$successCount/$totalCount images processed successfully"
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
            csvContent.append("Image Name,Processing Time (ms),Mask Upscale,Score Threshold,Downshift,Downscale MP,Segmentation Model,Success,Error Message\n")

            batchProcessingResults.forEach { result ->
                csvContent.append("${result.originalImageName},")
                csvContent.append("${result.processingTimeMs},")
                csvContent.append("${result.maskUpscale},")
                csvContent.append("${result.scoreThreshold},")
                csvContent.append("${result.downshift},")
                csvContent.append("${result.downscaleMp},")
                csvContent.append("${result.segmentationModel},")
                csvContent.append("${result.success},")
                csvContent.append("${result.errorMessage ?: ""}\n")
            }

            val timestamp = SimpleDateFormat("yyyyMMdd_HHmmss", Locale.getDefault()).format(Date())
            val fileName = "autokorrektur_batch_results_$timestamp.csv"

            val csvFile = File(requireContext().getExternalFilesDir(Environment.DIRECTORY_DOCUMENTS), fileName)
            csvFile.writeText(csvContent.toString())

            AppLogger.info("CSV exported to: ${csvFile.absolutePath}")
            Snackbar.make(binding.root, "CSV exported to: ${csvFile.name}", Snackbar.LENGTH_LONG).show()

        } catch (e: Exception) {
            AppLogger.error("Failed to export CSV", e)
            Snackbar.make(binding.root, "Failed to export CSV: ${e.message}", Snackbar.LENGTH_LONG).show()
        }
    }

    private fun createMaskOverlay(originalUri: Uri, maskMat: Mat) {
        try {
            AppLogger.debug("Creating mask overlay visualization")

            // Load original image as bitmap
            val originalBitmap = MediaStore.Images.Media.getBitmap(requireContext().contentResolver, originalUri)

            // Create a mutable copy of the original bitmap
            val overlayBitmap = originalBitmap.copy(Bitmap.Config.ARGB_8888, true)

            // Convert mask to bitmap
            val maskBitmap = createBitmap(maskMat.cols(), maskMat.rows())
            Utils.matToBitmap(maskMat, maskBitmap)

            // Scale mask bitmap to match original image size
            val scaledMaskBitmap = Bitmap.createScaledBitmap(
                maskBitmap, 
                overlayBitmap.width, 
                overlayBitmap.height, 
                true
            )

            // Create overlay by applying red tint to mask areas
            val overlayCanvas = android.graphics.Canvas(overlayBitmap)
            val paint = android.graphics.Paint().apply {
                alpha = 128 // 50% transparency
                colorFilter = android.graphics.PorterDuffColorFilter(
                    android.graphics.Color.RED,
                    android.graphics.PorterDuff.Mode.SRC_ATOP
                )
            }

            // Draw the scaled mask with red tint
            overlayCanvas.drawBitmap(scaledMaskBitmap, 0f, 0f, paint)

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
        return (binding.downshift.progress * 0.001).toFloat()
    }

    /**
     * Saves the processed bitmap to the gallery
     */
    private fun saveImageToGallery(bitmap: Bitmap): Uri? {
        try {
            val filename = "AutoKorrektur_${System.currentTimeMillis()}.jpg"
            var fos: OutputStream? = null
            var imageUri: Uri? = null

            // For Android 10 (Q) and above
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
                val contentValues = ContentValues().apply {
                    put(MediaStore.MediaColumns.DISPLAY_NAME, filename)
                    put(MediaStore.MediaColumns.MIME_TYPE, "image/jpeg")
                    put(MediaStore.MediaColumns.RELATIVE_PATH, Environment.DIRECTORY_PICTURES)
                }

                val contentResolver = requireContext().contentResolver
                imageUri = contentResolver.insert(MediaStore.Images.Media.EXTERNAL_CONTENT_URI, contentValues)
                fos = imageUri?.let { contentResolver.openOutputStream(it) }
            } else {
                // For older Android versions
                val imagesDir = Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_PICTURES)
                val image = File(imagesDir, filename)
                fos = FileOutputStream(image)
                imageUri = Uri.fromFile(image)
            }

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
