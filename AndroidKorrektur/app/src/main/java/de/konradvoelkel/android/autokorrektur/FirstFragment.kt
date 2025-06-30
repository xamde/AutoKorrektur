package de.konradvoelkel.android.autokorrektur

import android.Manifest
import android.app.Activity
import android.content.ContentValues
import android.content.Intent
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Canvas
import android.graphics.ColorMatrix
import android.graphics.ColorMatrixColorFilter
import android.graphics.Paint
import android.net.Uri
import android.os.Build
import android.os.Bundle
import android.os.Environment
import android.provider.MediaStore
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import android.widget.AdapterView
import android.widget.ArrayAdapter
import android.widget.ImageView
import android.widget.LinearLayout
import android.widget.SeekBar
import android.widget.TextView
import android.widget.Toast
import androidx.activity.result.contract.ActivityResultContracts
import androidx.core.content.ContextCompat
import androidx.core.content.FileProvider
import androidx.fragment.app.Fragment
import com.google.android.material.snackbar.Snackbar
import de.konradvoelkel.android.autokorrektur.databinding.FragmentFirstBinding
import java.io.File
import java.io.FileOutputStream
import java.io.InputStream
import java.io.OutputStream
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale

/**
 * Main fragment for the AutoKorrektur app, mimicking the web app functionality.
 */
class FirstFragment : Fragment() {

    private var _binding: FragmentFirstBinding? = null
    private val binding get() = _binding!!

    private var selectedImageUri: Uri? = null
    private var resultImageUri: Uri? = null
    private var processedBitmap: Bitmap? = null
    private var photoFile: File? = null

    // Activity result launcher for image selection
    private val selectImageLauncher = registerForActivityResult(
        ActivityResultContracts.StartActivityForResult()
    ) { result ->
        if (result.resultCode == Activity.RESULT_OK) {
            result.data?.data?.let { uri ->
                selectedImageUri = uri
                displayImage(uri, "Original")
                binding.startInference.isEnabled = true
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
            Toast.makeText(
                requireContext(),
                "Camera permission is required to take photos",
                Toast.LENGTH_LONG
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
            Toast.makeText(
                requireContext(),
                "Storage permission is required to select photos",
                Toast.LENGTH_LONG
            ).show()
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

        setupUI()
    }

    private fun setupUI() {
        // Setup file select button
        binding.fileSelect.setOnClickListener {
            if (binding.batchMode.isChecked) {
                // Multiple image selection would be implemented here
                Toast.makeText(context, "Multiple image selection not implemented in this mockup", Toast.LENGTH_SHORT).show()
            } else {
                selectImage()
            }
        }

        // Setup start inference button
        binding.startInference.setOnClickListener {
            mockInference()
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
                Toast.makeText(context, "No processed image to download. Run inference first.", Toast.LENGTH_SHORT).show()
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
            binding.startInference.isEnabled = selectedImageUri != null && !isChecked
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
            Toast.makeText(
                requireContext(),
                "Error creating image file: ${ex.message}",
                Toast.LENGTH_LONG
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
        val intent = Intent(Intent.ACTION_PICK, MediaStore.Images.Media.EXTERNAL_CONTENT_URI)
        selectImageLauncher.launch(intent)
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
        if (binding.optionsPanel.visibility == View.VISIBLE) {
            binding.optionsPanel.visibility = View.GONE
        } else {
            binding.optionsPanel.visibility = View.VISIBLE
        }
    }

    private fun mockInference() {
        // This is a mockup, so we'll just simulate the inference process
        binding.startInference.isEnabled = false
        binding.startInference.text = "Processing..."

        // Clear the images container
        clearImagesContainer()

        // Display the original image
        selectedImageUri?.let { uri ->
            displayImage(uri, "Original")
        }

        // Simulate processing delay
        binding.root.postDelayed({
            // Display a mock "mask" image
            selectedImageUri?.let { uri ->
                displayImage(uri, "Mask")
            }

            // Process the image by inverting colors
            selectedImageUri?.let { uri ->
                // Process the image (invert colors)
                processedBitmap = invertImageColors(uri)

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
                    displayImage(resultUri, "Result (Inverted Colors)")
                }
            }

            binding.startInference.isEnabled = true
            binding.startInference.text = "Start"

            Snackbar.make(binding.root, "Inference completed with color inversion", Snackbar.LENGTH_SHORT).show()
        }, 2000) // 2-second delay to simulate processing
    }

    /**
     * Inverts the colors of the given image
     */
    private fun invertImageColors(uri: Uri): Bitmap? {
        try {
            // Get the input stream from the URI
            val inputStream: InputStream? = requireContext().contentResolver.openInputStream(uri)

            // Decode the input stream into a bitmap
            val originalBitmap = BitmapFactory.decodeStream(inputStream)
            inputStream?.close()

            // Create a mutable copy of the bitmap
            val resultBitmap = originalBitmap.copy(Bitmap.Config.ARGB_8888, true)

            // Create a canvas to draw on the new bitmap
            val canvas = Canvas(resultBitmap)

            // Create a color matrix that inverts colors
            val colorMatrix = ColorMatrix().apply {
                set(floatArrayOf(
                    -1f, 0f, 0f, 0f, 255f,
                    0f, -1f, 0f, 0f, 255f,
                    0f, 0f, -1f, 0f, 255f,
                    0f, 0f, 0f, 1f, 0f
                ))
            }

            // Create a paint object with the color matrix
            val paint = Paint().apply {
                colorFilter = ColorMatrixColorFilter(colorMatrix)
            }

            // Draw the bitmap with the inverted colors
            canvas.drawBitmap(resultBitmap, 0f, 0f, paint)

            return resultBitmap
        } catch (e: Exception) {
            Toast.makeText(
                requireContext(),
                "Error processing image: ${e.message}",
                Toast.LENGTH_LONG
            ).show()
            return null
        }
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
                Toast.makeText(
                    requireContext(),
                    "Image saved to gallery",
                    Toast.LENGTH_SHORT
                ).show()
            }

            return imageUri
        } catch (e: Exception) {
            Toast.makeText(
                requireContext(),
                "Error saving image: ${e.message}",
                Toast.LENGTH_LONG
            ).show()
            return null
        }
    }

    override fun onDestroyView() {
        super.onDestroyView()
        _binding = null
    }
}
