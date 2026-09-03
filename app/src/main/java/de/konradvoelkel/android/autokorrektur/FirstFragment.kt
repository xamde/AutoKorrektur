package de.konradvoelkel.android.autokorrektur

import android.Manifest
import android.app.Activity
import android.app.AlertDialog
import android.content.Intent
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.ImageDecoder
import android.graphics.Paint
import android.net.Uri
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
import androidx.fragment.app.activityViewModels
import androidx.lifecycle.Lifecycle
import androidx.lifecycle.lifecycleScope
import androidx.lifecycle.repeatOnLifecycle
import com.google.android.material.snackbar.Snackbar
import de.konradvoelkel.android.autokorrektur.ar.ArCameraActivity
import de.konradvoelkel.android.autokorrektur.manager.ConsentManager
import de.konradvoelkel.android.autokorrektur.manager.QuotaManager
import de.konradvoelkel.android.autokorrektur.databinding.FragmentFirstBinding
import de.konradvoelkel.android.autokorrektur.ui.model.MainUiProperties
import de.konradvoelkel.android.autokorrektur.ui.model.MainUiState
import de.konradvoelkel.android.autokorrektur.utils.AppLogger
import de.konradvoelkel.android.autokorrektur.utils.BitmapMemoryUtils
import de.konradvoelkel.android.autokorrektur.model.InpaintingQualityMode
import de.konradvoelkel.android.autokorrektur.utils.ImageExportManager
import de.konradvoelkel.android.autokorrektur.utils.InstagramExportUtils
import de.konradvoelkel.android.autokorrektur.utils.MaskOverlayUtils
import kotlinx.coroutines.flow.collectLatest
import kotlinx.coroutines.launch
import org.opencv.android.Utils
import java.io.File
import java.io.FileOutputStream
import java.io.IOException
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale
import androidx.core.content.edit

/**
 * Main fragment for the AutoKorrektur app, now refactored to use MainViewModel and MainUiState.
 */
class FirstFragment : Fragment() {

    private var _binding: FragmentFirstBinding? = null
    private val binding get() = _binding!!

    private val viewModel: MainViewModel by activityViewModels()
    private lateinit var exportManager: ImageExportManager
    private lateinit var instagramDelegate: de.konradvoelkel.android.autokorrektur.ui.delegate.InstagramExportDelegate
    private lateinit var batchUiDelegate: de.konradvoelkel.android.autokorrektur.ui.delegate.BatchUiDelegate

    // Activity result launcher for image selection
    private val selectImageLauncher = registerForActivityResult(
        ActivityResultContracts.StartActivityForResult()
    ) { result ->
        if (result.resultCode == Activity.RESULT_OK) {
            val uri = result.data?.data
            if (uri != null) {
                onImageSelected(uri)
            } else {
                showSnackbar(getString(R.string.error_gallery_failed))
            }
        }
    }

    private var pendingCameraUri: Uri? = null

    private var displayBeforeBmp: Bitmap? = null
    private var displayAfterBmp: Bitmap? = null
    private var combinedMaskBmp: Bitmap? = null
    private var decodedBrushBmp: Bitmap? = null

    // Activity result launcher for camera
    private val takePictureLauncher = registerForActivityResult(
        ActivityResultContracts.TakePicture()
    ) { success ->
        if (success) {
            pendingCameraUri?.let { uri ->
                onImageSelected(uri)
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
            showSnackbar(getString(R.string.error_camera_permission_required))
        }
    }

    // Multiple image picker launcher for batch processing
    private val multipleImagePickerLauncher = registerForActivityResult(
        ActivityResultContracts.GetMultipleContents()
    ) { uris ->
        if (uris.isNotEmpty()) {
            viewModel.setSelectedImageUris(uris)
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
            try {
                org.opencv.android.OpenCVLoader.initLocal()
                Log.d(TAG, "OpenCV native library loaded successfully.")
            } catch (e: Exception) {
                Log.e(TAG, "Failed to load OpenCV native library!", e)
            }
        }
    }

    override fun onViewCreated(view: View, savedInstanceState: Bundle?) {
        super.onViewCreated(view, savedInstanceState)
        exportManager = ImageExportManager(requireContext())
        instagramDelegate = de.konradvoelkel.android.autokorrektur.ui.delegate.InstagramExportDelegate(requireContext(), exportManager) { showSnackbar(it) }
        batchUiDelegate = de.konradvoelkel.android.autokorrektur.ui.delegate.BatchUiDelegate(requireContext(), exportManager) { showSnackbar(it) }
        applyFeatureFlags()
        setupUI()
        observeViewModel()

        val extraUriString = requireActivity().intent?.getStringExtra("EXTRA_IMAGE_URI")
        val intentData = requireActivity().intent?.data
        val targetUri = when {
            extraUriString != null -> android.net.Uri.parse(extraUriString)
            intentData != null -> intentData
            else -> null
        }
        targetUri?.let { uri ->
            onImageSelected(uri)
        }
    }

    /**
     * Records the newly selected/captured image and, when this tier offers no real inpainting
     * engine choice (see [autoStartInferenceEnabled]), immediately kicks off processing instead
     * of waiting for a manual Start tap — see docs/MVP_FEATURE_FLAG_PLAN.md §1 ("no screen the
     * user must configure before their first result").
     */
    private fun onImageSelected(uri: Uri) {
        viewModel.setSelectedImageUri(uri)
        if (autoStartInferenceEnabled) {
            startInferenceNow()
        }
    }

    /**
     * True when this build's tier offers no real choice between inpainting engines (only Fast
     * On-Device is available — see [applyFeatureFlags]), so there's nothing for the user to
     * decide before processing starts.
     */
    private val autoStartInferenceEnabled: Boolean
        get() = !BuildConfig.FEATURE_HIGH_RES_PROGRESSIVE && !BuildConfig.FEATURE_CLOUD_SDXL

    private fun startInferenceNow() {
        viewModel.startInference(
            downscaleMp = getDownscaleMpFromSpinner(),
            maskUpscale = getMaskUpscaleFromSlider(),
            scoreThreshold = getScoreThresholdFromSlider(),
            useServerSdxl = binding.useSdxl.isChecked,
            downshift = getDownshiftFromSlider(),
            segModel = binding.segModel.selectedItem.toString()
        )
    }

    private fun observeViewModel() {
        viewLifecycleOwner.lifecycleScope.launch {
            viewLifecycleOwner.repeatOnLifecycle(Lifecycle.State.STARTED) {
                launch {
                    viewModel.uiState.collectLatest { state ->
                        handleUiState(state)
                    }
                }
                launch {
                    viewModel.properties.collectLatest { props ->
                        updateUiFromProperties(props)
                    }
                }
            }
        }
    }

    private fun handleUiState(state: MainUiState) {
        when (state) {
            is MainUiState.Idle -> {
                binding.beforeAfterSliderView.visibility = View.GONE
                binding.imagesContainer.visibility = View.VISIBLE
                updateInferenceButtonState(viewModel.properties.value)
                binding.fileSelect.isEnabled = true
                binding.batchMode.isEnabled = true
            }

            is MainUiState.Loading -> {
                val statusText = getString(R.string.loading_status, state.stage, state.percent)
                if (autoStartInferenceEnabled) {
                    binding.processingStatusText.visibility = View.VISIBLE
                    binding.processingStatusText.text = statusText
                } else {
                    binding.startInference.isEnabled = false
                    binding.startInference.text = statusText
                }
                binding.fileSelect.isEnabled = false
                binding.batchMode.isEnabled = false

                if (state.intermediateInpaintedBitmap != null && !viewModel.properties.value.isBatchMode) {
                    val displayAfter = BitmapMemoryUtils.createScaledBitmapForDisplay(
                        state.intermediateInpaintedBitmap, maxDimension = 1440
                    )
                    displayAfterBmp?.recycle()
                    displayAfterBmp = displayAfter
                    binding.beforeAfterSliderView.updateAfterBitmap(displayAfter)
                    binding.beforeAfterSliderView.visibility = View.VISIBLE
                    binding.imagesContainer.visibility = View.GONE
                }
            }

            is MainUiState.Success -> {
                val result = state.result
                val inpainted = result.inpaintedBitmap
                if (inpainted != null) {
                    if (viewModel.properties.value.isBatchMode) {
                        addImageToContainer(inpainted, getString(R.string.label_result))
                        if (viewModel.properties.value.batchProcessingResults.size == viewModel.properties.value.selectedImageUris.size) {
                            finalizeBatchProcessing()
                        }
                    } else {
                        displayBeforeBmp?.recycle()
                        displayAfterBmp?.recycle()
                        combinedMaskBmp?.recycle()

                        val displayBefore =
                            BitmapMemoryUtils.createScaledBitmapForDisplay(
                                result.originalBitmap, maxDimension = 1440
                            )
                        val displayAfter =
                            BitmapMemoryUtils.createScaledBitmapForDisplay(
                                inpainted, maxDimension = 1440
                            )
                        
                        displayBeforeBmp = displayBefore
                        displayAfterBmp = displayAfter

                        binding.beforeAfterSliderView.setBitmaps(displayBefore, displayAfter)
                        binding.beforeAfterSliderView.visibility = View.VISIBLE

                        // Render intermediate vehicle mask preview with a semi-transparent red overlay
                        val overlay = MaskOverlayUtils.createRedOverlayBitmap(
                            result.maskBitmap,
                            displayBefore.width,
                            displayBefore.height
                        )
                        val combinedMask = Bitmap.createBitmap(displayBefore.width, displayBefore.height, Bitmap.Config.ARGB_8888)
                        val canvas = Canvas(combinedMask)
                        canvas.drawBitmap(displayBefore, 0f, 0f, null)
                        canvas.drawBitmap(overlay, 0f, 0f, null)
                        overlay.recycle()

                        combinedMaskBmp = combinedMask

                        binding.imagesContainer.removeAllViews()
                        addImageToContainer(combinedMask, getString(R.string.label_mask))
                        binding.imagesContainer.visibility = View.VISIBLE
                    }
                } else {
                    showSnackbar(getString(R.string.error_no_processed_image))
                    viewModel.clearState()
                }
                updateInferenceButtonState(viewModel.properties.value)
                binding.fileSelect.isEnabled = true
                binding.batchMode.isEnabled = true
            }

            is MainUiState.Error -> {
                showSnackbar(state.message)
                viewModel.clearState()
            }
        }
    }

    private fun updateUiFromProperties(props: MainUiProperties) {
        if (props.isBatchMode) {
            clearImagesContainer()
            props.selectedImageUris.take(3).forEachIndexed { index, uri ->
                displayImage(uri, getString(R.string.label_image_numbered, index + 1))
            }
            binding.fileSelect.text = getString(R.string.select_multiple_images)
            // Camera capture doesn't fit batch multi-select; fileSelect alone (as the
            // multi-image gallery picker) takes over the whole row.
            binding.btnTakePhoto.visibility = View.GONE
        } else {
            clearImagesContainer()
            props.selectedImageUri?.let { displayImage(it, getString(R.string.label_original)) }
            binding.fileSelect.text = getString(R.string.select_image)
            binding.btnTakePhoto.visibility = View.VISIBLE
        }
        if (viewModel.uiState.value !is MainUiState.Success) {
            binding.beforeAfterSliderView.visibility = View.GONE
            binding.imagesContainer.visibility = View.VISIBLE
        }
        updateInferenceButtonState(props)
        binding.beforeAfterSliderView.setSliderPosition(props.sliderPosition)

        when (props.qualityMode) {
            InpaintingQualityMode.CLOUD_SDXL -> {
                binding.chipCloudSdxl.isChecked = true
                binding.tvPrivacyBadge.visibility = View.VISIBLE
                binding.tvQuotaBadge.visibility = View.VISIBLE
            }
            InpaintingQualityMode.HIGH_RES_PROGRESSIVE -> {
                binding.chipHighResProgressive.isChecked = true
                binding.tvPrivacyBadge.visibility = View.GONE
                binding.tvQuotaBadge.visibility = View.GONE
            }
            InpaintingQualityMode.FAST_PREVIEW -> {
                binding.chipFastPreview.isChecked = true
                binding.tvPrivacyBadge.visibility = View.GONE
                binding.tvQuotaBadge.visibility = View.GONE
            }
        }
    }

    private fun updateInferenceButtonState(props: MainUiProperties) {
        // Not currently processing (Idle/Success) — nothing to show in the auto-start tier's
        // status slot; startInference itself stays permanently hidden there (applyFeatureFlags).
        binding.processingStatusText.visibility = View.GONE
        if (props.isBatchMode) {
            binding.startInference.isEnabled = props.selectedImageUris.isNotEmpty()
            binding.startInference.text = if (props.selectedImageUris.isNotEmpty()) {
                getString(R.string.start_batch_processing_images, props.selectedImageUris.size)
            } else {
                getString(R.string.btn_start_batch_processing)
            }
        } else {
            binding.startInference.isEnabled = props.selectedImageUri != null
            binding.startInference.text = getString(R.string.start)
        }
    }

    /**
     * Hides UI entry points for features not present in this build's tier
     * (see docs/MVP_FEATURE_FLAG_PLAN.md).
     */
    private fun applyFeatureFlags() {
        if (!BuildConfig.FEATURE_LIVE_AR) {
            binding.arLiveModeButton.visibility = View.GONE
        }
        if (!BuildConfig.FEATURE_CLOUD_SDXL) {
            binding.chipCloudSdxl.visibility = View.GONE
        }
        if (!BuildConfig.FEATURE_HIGH_RES_PROGRESSIVE) {
            binding.chipHighResProgressive.visibility = View.GONE
        }
        if (!BuildConfig.FEATURE_MANUAL_MASK_BRUSH) {
            binding.btnMaskBrush.visibility = View.GONE
        }
        if (!BuildConfig.FEATURE_BATCH_PROCESSING) {
            binding.multipleImagesRow.visibility = View.GONE
        }
        if (!BuildConfig.FEATURE_EVALUATION_MODE) {
            binding.evaluationModeContainer.visibility = View.GONE
        }
        // Fast On-Device is always available; High-Res and Cloud are the only other options.
        // A chip-group "choice" between one always-checked pill and nothing else isn't a real
        // choice, so don't show it as one — this is exactly the plan's own "no engine picker
        // unless there's something to pick" principle (docs/MVP_FEATURE_FLAG_PLAN.md §1).
        if (!BuildConfig.FEATURE_HIGH_RES_PROGRESSIVE && !BuildConfig.FEATURE_CLOUD_SDXL) {
            binding.premiumEditCard.visibility = View.GONE
        }
        // No engine choice to make before starting (see autoStartInferenceEnabled) means the
        // manual Start button isn't needed either — processing begins as soon as an image is
        // selected (onImageSelected), and processingStatusText carries progress feedback instead.
        if (autoStartInferenceEnabled) {
            binding.startInference.visibility = View.GONE
        }
        // The Options panel's only feature not already covered by a dedicated flag above is
        // batch processing — with it off, the panel has nothing left worth surfacing an entry
        // point for.
        if (!BuildConfig.FEATURE_BATCH_PROCESSING) {
            binding.optionsButton.visibility = View.GONE
        }
    }

    private fun setupUI() {
        binding.chipGroupQuality.setOnCheckedStateChangeListener { _, checkedIds ->
            when {
                checkedIds.contains(R.id.chipCloudSdxl) -> {
                    viewModel.setQualityMode(InpaintingQualityMode.CLOUD_SDXL)
                    binding.useSdxl.isChecked = true
                    binding.tvPrivacyBadge.visibility = View.VISIBLE
                    binding.tvQuotaBadge.visibility = View.VISIBLE
                }
                checkedIds.contains(R.id.chipHighResProgressive) -> {
                    viewModel.setQualityMode(InpaintingQualityMode.HIGH_RES_PROGRESSIVE)
                    binding.useSdxl.isChecked = false
                    binding.tvPrivacyBadge.visibility = View.GONE
                    binding.tvQuotaBadge.visibility = View.GONE
                }
                else -> {
                    viewModel.setQualityMode(InpaintingQualityMode.FAST_PREVIEW)
                    binding.useSdxl.isChecked = false
                    binding.tvPrivacyBadge.visibility = View.GONE
                    binding.tvQuotaBadge.visibility = View.GONE
                }
            }
        }

        binding.fileSelect.setOnClickListener {
            if (binding.batchMode.isChecked) {
                multipleImagePickerLauncher.launch("image/*")
            } else {
                chooseFromGallery()
            }
        }

        binding.btnTakePhoto.setOnClickListener {
            takePhoto()
        }

        binding.btnMaskBrush.setOnClickListener {
            val uri = viewModel.properties.value.selectedImageUri
            if (uri != null) {
                try {
                    val bitmap = if (android.os.Build.VERSION.SDK_INT >= android.os.Build.VERSION_CODES.P) {
                        ImageDecoder.decodeBitmap(ImageDecoder.createSource(requireContext().contentResolver, uri))
                    } else {
                        @Suppress("DEPRECATION")
                        MediaStore.Images.Media.getBitmap(requireContext().contentResolver, uri)
                    }
                    
                    val mutableBitmap = bitmap.copy(Bitmap.Config.ARGB_8888, true)
                    if (bitmap != mutableBitmap) {
                        bitmap.recycle()
                    }
                    decodedBrushBmp?.recycle()
                    decodedBrushBmp = mutableBitmap

                    val currentSuccess = viewModel.uiState.value as? MainUiState.Success
                    val initialMask = currentSuccess?.result?.maskBitmap
                    val sheet = de.konradvoelkel.android.autokorrektur.ui.brush.MaskBrushBottomSheet.newInstance(
                        sourceBitmap = mutableBitmap,
                        initialMaskBitmap = initialMask
                    )
                    sheet.onMaskApplied = { _ ->
                        showSnackbar("Maske angepasst 🖌️ Inpainting wird gestartet...")
                        startInferenceNow()
                    }
                    sheet.show(childFragmentManager, "MaskBrushBottomSheet")
                } catch (e: Exception) {
                    showSnackbar("Fehler beim Laden des Bildes: ${e.message}")
                }
            } else {
                showSnackbar("Bitte zuerst ein Bild auswählen")
            }
        }

        binding.startInference.setOnClickListener {
            startInferenceNow()
        }

        binding.download.setOnClickListener {
            val state = viewModel.uiState.value
            if (state is MainUiState.Success) {
                state.result.inpaintedBitmap?.let { bitmap ->
                    val savedUri = exportManager.saveImageToGallery(bitmap)
                    if (savedUri != null) {
                        showSnackbar(getString(R.string.msg_image_saved))
                    }
                }
            } else {
                showSnackbar(getString(R.string.error_no_processed_image))
            }
        }

        binding.exportInstagram.setOnClickListener {
            exportInstagramGraphic()
        }

        binding.arLiveModeButton.setOnClickListener {
            if (ContextCompat.checkSelfPermission(requireContext(), Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED) {
                val intent = Intent(requireContext(), ArCameraActivity::class.java)
                startActivity(intent)
            } else {
                cameraPermissionLauncher.launch(Manifest.permission.CAMERA)
            }
        }

        binding.optionsButton.setOnClickListener {
            toggleOptionsPanel()
        }

        setupSliders()
        setupSpinners()

        binding.batchMode.setOnCheckedChangeListener { _, isChecked ->
            viewModel.setBatchMode(isChecked)
        }

        binding.useSdxl.setOnCheckedChangeListener { _, isChecked ->
            if (isChecked) {
                val consentManager = ConsentManager(requireContext())
                val quotaManager = QuotaManager(requireContext())
                val remaining = quotaManager.getRemainingDailyQuota()
                if (!consentManager.isConsentGranted()) {
                    val view = LayoutInflater.from(requireContext()).inflate(R.layout.dialog_gdpr_consent, null)
                    val checkbox = view.findViewById<android.widget.CheckBox>(R.id.rememberChoiceCheckbox)
                    AlertDialog.Builder(requireContext())
                        .setTitle(R.string.premium_edit_title)
                        .setView(view)
                        .setPositiveButton(R.string.btn_accept) { _, _ ->
                            if (checkbox.isChecked) {
                                consentManager.setConsentGranted(true)
                            }
                        }
                        .setNegativeButton(R.string.cancel) { _, _ ->
                            binding.useSdxl.isChecked = false
                        }
                        .show()
                }
            }
        }

        binding.continueWithResult.setOnCheckedChangeListener { _, isChecked ->
            if (isChecked && viewModel.uiState.value !is MainUiState.Success && !binding.batchMode.isChecked) {
                showSnackbar(getString(R.string.error_no_previous_result))
                binding.continueWithResult.isChecked = false
            }
        }
    }

    private fun setupSliders() {
        binding.maskUpscale.setOnSeekBarChangeListener(object : SeekBar.OnSeekBarChangeListener {
            override fun onProgressChanged(seekBar: SeekBar?, progress: Int, fromUser: Boolean) {
                val value = (1 + progress * 0.01).toFloat()
                binding.maskUpscaleVal.text = String.format("%.2f", value)
            }
            override fun onStartTrackingTouch(seekBar: SeekBar?) {}
            override fun onStopTrackingTouch(seekBar: SeekBar?) {}
        })

        binding.downshift.setOnSeekBarChangeListener(object : SeekBar.OnSeekBarChangeListener {
            override fun onProgressChanged(seekBar: SeekBar?, progress: Int, fromUser: Boolean) {
                val value = progress * 0.001
                binding.downshiftVal.text = String.format("%.3f", value)
            }
            override fun onStartTrackingTouch(seekBar: SeekBar?) {}
            override fun onStopTrackingTouch(seekBar: SeekBar?) {}
        })

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
        val downscaleAdapter = ArrayAdapter.createFromResource(
            requireContext(),
            R.array.downscale_options,
            android.R.layout.simple_spinner_item
        )
        downscaleAdapter.setDropDownViewResource(android.R.layout.simple_spinner_dropdown_item)
        binding.downscaleMP.adapter = downscaleAdapter

        val segModelAdapter = ArrayAdapter.createFromResource(
            requireContext(),
            R.array.yolo_model_options,
            android.R.layout.simple_spinner_item
        )
        segModelAdapter.setDropDownViewResource(android.R.layout.simple_spinner_dropdown_item)
        binding.segModel.adapter = segModelAdapter
        binding.segModel.setSelection(1)
    }

    private fun takePhoto() {
        if (ContextCompat.checkSelfPermission(
                requireContext(),
                Manifest.permission.CAMERA
            ) == PackageManager.PERMISSION_GRANTED
        ) {
            launchCamera()
        } else {
            cameraPermissionLauncher.launch(Manifest.permission.CAMERA)
        }
    }

    private fun launchCamera() {
        try {
            val file = createImageFile()
            val uri = FileProvider.getUriForFile(
                requireContext(),
                "${requireContext().packageName}.fileprovider",
                file
            )
            pendingCameraUri = uri
            takePictureLauncher.launch(uri)
        } catch (ex: Exception) {
            AppLogger.error(getString(R.string.error_create_file_failed), ex)
            showSnackbar(getString(R.string.error_create_file_message, ex.message))
        }
    }

    private val pickVisualMediaLauncher =
        registerForActivityResult(ActivityResultContracts.PickVisualMedia()) { uri ->
        if (uri != null) {
            onImageSelected(uri)
        }
    }

    private fun chooseFromGallery() {
        if (ActivityResultContracts.PickVisualMedia.isPhotoPickerAvailable(requireContext())) {
            pickVisualMediaLauncher.launch(
                androidx.activity.result.PickVisualMediaRequest(
                    ActivityResultContracts.PickVisualMedia.ImageOnly
                )
            )
        } else {
            val intent = Intent(Intent.ACTION_GET_CONTENT).apply { type = "image/*" }
            selectImageLauncher.launch(intent)
        }
    }

    private fun createImageFile(): File {
        val timeStamp = SimpleDateFormat("yyyyMMdd_HHmmss", Locale.getDefault()).format(Date())
        val storageDir = requireContext().getExternalFilesDir(Environment.DIRECTORY_PICTURES)
        return File.createTempFile("JPEG_${timeStamp}_", ".jpg", storageDir)
    }

    private fun decodePreviewBitmap(uri: Uri, targetW: Int, targetH: Int): Bitmap {
        return BitmapMemoryUtils.decodeSampledBitmapFromUri(
            context = requireContext(),
            uri = uri,
            maxDimension = kotlin.math.max(targetW, targetH)
        )
    }

    private fun displayImage(uri: Uri, label: String) {
        val imageView = ImageView(context).apply {
            layoutParams = LinearLayout.LayoutParams(LinearLayout.LayoutParams.MATCH_PARENT, 400)
            scaleType = ImageView.ScaleType.FIT_CENTER
            val preview = decodePreviewBitmap(uri, resources.displayMetrics.widthPixels, 400)
            setImageBitmap(preview)
        }
        val textView = TextView(context).apply {
            text = label
            textAlignment = View.TEXT_ALIGNMENT_CENTER
        }
        val container = LinearLayout(context).apply {
            orientation = LinearLayout.VERTICAL
            layoutParams = LinearLayout.LayoutParams(
                LinearLayout.LayoutParams.MATCH_PARENT,
                LinearLayout.LayoutParams.WRAP_CONTENT
            )
            addView(imageView)
            addView(textView)
        }
        binding.imagesContainer.addView(container)
    }

    private fun addImageToContainer(bitmap: Bitmap, label: String) {
        val imageView = ImageView(context).apply {
            layoutParams = LinearLayout.LayoutParams(LinearLayout.LayoutParams.MATCH_PARENT, 400)
            scaleType = ImageView.ScaleType.FIT_CENTER
            setImageBitmap(bitmap)
        }
        val textView = TextView(context).apply {
            text = label
            textAlignment = View.TEXT_ALIGNMENT_CENTER
        }
        val container = LinearLayout(context).apply {
            orientation = LinearLayout.VERTICAL
            layoutParams = LinearLayout.LayoutParams(
                LinearLayout.LayoutParams.MATCH_PARENT,
                LinearLayout.LayoutParams.WRAP_CONTENT
            )
            addView(imageView)
            addView(textView)
        }
        binding.imagesContainer.addView(container)
    }

    private fun showSnackbar(message: String) {
        Snackbar.make(binding.root, message, Snackbar.LENGTH_LONG).show()
    }

    private fun clearImagesContainer() {
        binding.imagesContainer.removeAllViews()
        binding.beforeAfterSliderView.visibility = View.GONE
        binding.imagesContainer.visibility = View.VISIBLE
    }

    private fun toggleOptionsPanel() {
        binding.optionsPanel.visibility =
            if (binding.optionsPanel.isVisible) View.GONE else View.VISIBLE
    }

    private fun finalizeBatchProcessing() {
        val results = viewModel.properties.value.batchProcessingResults
        val successCount = results.count { it.success }
        val message = getString(R.string.msg_batch_completed, successCount, results.size)
        Snackbar.make(binding.root, message, Snackbar.LENGTH_LONG)
            .setAction(R.string.btn_export_csv) { batchUiDelegate.showCsvExportDialog(results) }
            .show()
    }

    private fun exportInstagramGraphic() {
        val state = viewModel.uiState.value
        if (state !is MainUiState.Success) {
            showSnackbar(getString(R.string.error_export_instagram_no_image))
            return
        }
        val inpainted = state.result.inpaintedBitmap ?: return
        val currentOriginalBitmap = state.result.originalBitmap

        if (!BuildConfig.FEATURE_EXTRA_EXPORT_LAYOUTS) {
            // Only one export shape exists in this tier — the bottom sheet's layout/ratio
            // picker would be a choice with a single option, i.e. not a real choice, so skip
            // straight to composing and sharing it (docs/MVP_FEATURE_FLAG_PLAN.md §1).
            viewLifecycleOwner.lifecycleScope.launch {
                try {
                    val uri = de.konradvoelkel.android.autokorrektur.utils.InstagramExportUtils
                        .exportSplitCardForSharing(requireContext(), currentOriginalBitmap, inpainted)
                    de.konradvoelkel.android.autokorrektur.utils.InstagramExportUtils
                        .shareImage(requireContext(), uri, getString(R.string.share_chooser_title))
                } catch (e: Exception) {
                    showSnackbar(getString(R.string.export_error_failed, e.message))
                }
            }
            return
        }

        val sheet = de.konradvoelkel.android.autokorrektur.ui.export.InstagramExportBottomSheet.newInstance(
            before = currentOriginalBitmap,
            after = inpainted
        )
        sheet.show(childFragmentManager, "InstagramExportBottomSheet")
    }

    private fun getDownscaleMpFromSpinner(): Float? {
        val selectedItem = binding.downscaleMP.selectedItem.toString()
        val noScaling = resources.getStringArray(R.array.downscale_options)[0]
        return when (selectedItem) {
            noScaling -> null
            else -> selectedItem.replace(" MP", "").toFloatOrNull()
        }
    }

    private fun getMaskUpscaleFromSlider() = (1 + binding.maskUpscale.progress * 0.01).toFloat()
    private fun getScoreThresholdFromSlider() = (binding.scoreThreshold.progress * 0.01).toFloat()
    private fun getDownshiftFromSlider() = (binding.downshift.progress * 0.001).toFloat()

    override fun onDestroyView() {
        super.onDestroyView()
        _binding = null
        displayBeforeBmp?.recycle()
        displayBeforeBmp = null
        displayAfterBmp?.recycle()
        displayAfterBmp = null
        combinedMaskBmp?.recycle()
        combinedMaskBmp = null
        decodedBrushBmp?.recycle()
        decodedBrushBmp = null
    }
}
