package de.konradvoelkel.android.autokorrektur

import android.app.Activity
import android.content.Intent
import android.net.Uri
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
import androidx.core.content.FileProvider
import androidx.fragment.app.Fragment
import com.google.android.material.snackbar.Snackbar
import de.konradvoelkel.android.autokorrektur.databinding.FragmentFirstBinding
import java.io.File
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
            resultImageUri?.let {
                Toast.makeText(context, "Download functionality would save the image to gallery", Toast.LENGTH_SHORT).show()
            } ?: run {
                Toast.makeText(context, "No result image to download", Toast.LENGTH_SHORT).show()
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

            // Display a mock "result" image
            selectedImageUri?.let { uri ->
                resultImageUri = uri
                displayImage(uri, "Result")
            }

            binding.startInference.isEnabled = true
            binding.startInference.text = "Start"

            Snackbar.make(binding.root, "Inference completed (mockup)", Snackbar.LENGTH_SHORT).show()
        }, 2000) // 2-second delay to simulate processing
    }

    override fun onDestroyView() {
        super.onDestroyView()
        _binding = null
    }
}
