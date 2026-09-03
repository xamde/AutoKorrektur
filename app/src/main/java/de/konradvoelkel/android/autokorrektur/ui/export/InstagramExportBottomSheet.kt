package de.konradvoelkel.android.autokorrektur.ui.export

import android.content.Intent
import android.graphics.Bitmap
import android.net.Uri
import android.os.Bundle
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import android.widget.Toast
import androidx.core.content.FileProvider
import androidx.lifecycle.lifecycleScope
import com.google.android.material.bottomsheet.BottomSheetDialogFragment
import de.konradvoelkel.android.autokorrektur.R
import de.konradvoelkel.android.autokorrektur.databinding.BottomSheetInstagramExportBinding
import de.konradvoelkel.android.autokorrektur.utils.AppLogger
import de.konradvoelkel.android.autokorrektur.utils.InstagramExportUtils
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.io.File

/**
 * Bottom Sheet Dialog presenting Instagram Multi-Layout export options:
 * - Side-by-Side Split Image
 * - 2-Slide Swipe Carousel Pair
 * - Animated Split-Sweep MP4 Video for Reels / Stories
 */
class InstagramExportBottomSheet : BottomSheetDialogFragment() {

    private var _binding: BottomSheetInstagramExportBinding? = null
    private val binding get() = _binding!!

    var beforeBitmap: Bitmap? = null
    var afterBitmap: Bitmap? = null

    override fun onCreateView(
        inflater: LayoutInflater,
        container: ViewGroup?,
        savedInstanceState: Bundle?
    ): View {
        _binding = BottomSheetInstagramExportBinding.inflate(inflater, container, false)
        return binding.root
    }

    override fun onViewCreated(view: View, savedInstanceState: Bundle?) {
        super.onViewCreated(view, savedInstanceState)
        applyFeatureFlags()

        binding.btnExportInstagram.setOnClickListener {
            performExport()
        }
    }

    /**
     * MVP tier ships exactly one export shape (the split card, one aspect ratio) — see
     * docs/MVP_FEATURE_FLAG_PLAN.md §2. With only one layout and one ratio possible, the
     * picker rows themselves aren't a real choice either — hide the whole row (label + chip
     * group), not just the extra chips inside it, so `core` doesn't show a single-option
     * "choice" that can't do anything except stay selected.
     */
    private fun applyFeatureFlags() {
        if (!de.konradvoelkel.android.autokorrektur.BuildConfig.FEATURE_EXTRA_EXPORT_LAYOUTS) {
            binding.tvExportLayoutTypeLabel.visibility = View.GONE
            binding.chipGroupLayout.visibility = View.GONE
            binding.tvExportAspectRatioLabel.visibility = View.GONE
            binding.chipGroupRatio.visibility = View.GONE
        }
    }

    private fun performExport() {
        val before = beforeBitmap
        val after = afterBitmap
        if (before == null || after == null) {
            Toast.makeText(requireContext(), getString(R.string.export_error_not_ready), Toast.LENGTH_SHORT).show()
            dismiss()
            return
        }

        val ratio = when (binding.chipGroupRatio.checkedChipId) {
            R.id.chipRatio45 -> InstagramExportUtils.AspectRatio.PORTRAIT_4_5
            R.id.chipRatio916 -> InstagramExportUtils.AspectRatio.STORY_9_16
            else -> InstagramExportUtils.AspectRatio.SQUARE_1_1
        }

        binding.btnExportInstagram.isEnabled = false
        binding.btnExportInstagram.text = getString(R.string.export_btn_exporting)

        lifecycleScope.launch(Dispatchers.IO) {
            try {
                when (binding.chipGroupLayout.checkedChipId) {
                    R.id.chipCarouselPair -> {
                        val (slide1, slide2) = InstagramExportUtils.createCarouselPair(before, after, ratio)
                        val uri1 = InstagramExportUtils.saveBitmapForSharing(requireContext(), slide1, "autokorrektur_slide1.jpg")
                        val uri2 = InstagramExportUtils.saveBitmapForSharing(requireContext(), slide2, "autokorrektur_slide2.jpg")
                        slide1.recycle()
                        slide2.recycle()

                        withContext(Dispatchers.Main) {
                            shareMultipleImages(arrayListOf(uri1, uri2))
                            dismiss()
                        }
                    }

                    R.id.chipAnimatedVideo -> {
                        val videoDir = File(requireContext().cacheDir, "export_videos").apply { mkdirs() }
                        val videoFile = File(videoDir, "autokorrektur_anim_${System.currentTimeMillis()}.mp4")
                        InstagramExportUtils.createAnimatedSweepVideo(
                            beforeBitmap = before,
                            afterBitmap = after,
                            outputFile = videoFile,
                            ratio = ratio
                        )

                        val videoUri = FileProvider.getUriForFile(
                            requireContext(),
                            "${requireContext().packageName}.fileprovider",
                            videoFile
                        )

                        withContext(Dispatchers.Main) {
                            shareVideo(videoUri)
                            dismiss()
                        }
                    }

                    else -> { // Split Card
                        val imageUri = InstagramExportUtils.exportSplitCardForSharing(requireContext(), before, after, ratio)

                        withContext(Dispatchers.Main) {
                            InstagramExportUtils.shareImage(requireContext(), imageUri, getString(R.string.share_chooser_title))
                            dismiss()
                        }
                    }
                }
            } catch (e: Exception) {
                AppLogger.error("Instagram export failed", e)
                withContext(Dispatchers.Main) {
                    Toast.makeText(requireContext(), getString(R.string.export_error_failed, e.message), Toast.LENGTH_LONG).show()
                    binding.btnExportInstagram.isEnabled = true
                    binding.btnExportInstagram.text = getString(R.string.btn_export_share)
                }
            }
        }
    }

    private fun shareMultipleImages(uris: ArrayList<Uri>) {
        val intent = Intent(Intent.ACTION_SEND_MULTIPLE).apply {
            type = "image/jpeg"
            putParcelableArrayListExtra(Intent.EXTRA_STREAM, uris)
            addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION)
            setPackage("com.instagram.android")
        }

        if (intent.resolveActivity(requireContext().packageManager) != null) {
            startActivity(intent)
        } else {
            val chooser = Intent.createChooser(
                Intent(Intent.ACTION_SEND_MULTIPLE).apply {
                    type = "image/jpeg"
                    putParcelableArrayListExtra(Intent.EXTRA_STREAM, uris)
                    addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION)
                },
                "Karussell teilen via"
            )
            startActivity(chooser)
        }
    }

    private fun shareVideo(uri: Uri) {
        val intent = Intent(Intent.ACTION_SEND).apply {
            type = "video/mp4"
            putExtra(Intent.EXTRA_STREAM, uri)
            addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION)
            setPackage("com.instagram.android")
        }

        if (intent.resolveActivity(requireContext().packageManager) != null) {
            startActivity(intent)
        } else {
            val chooser = Intent.createChooser(
                Intent(Intent.ACTION_SEND).apply {
                    type = "video/mp4"
                    putExtra(Intent.EXTRA_STREAM, uri)
                    addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION)
                },
                "Animiertes Video teilen via"
            )
            startActivity(chooser)
        }
    }

    override fun onDestroyView() {
        super.onDestroyView()
        _binding = null
    }

    companion object {
        fun newInstance(before: Bitmap, after: Bitmap): InstagramExportBottomSheet {
            return InstagramExportBottomSheet().apply {
                this.beforeBitmap = before
                this.afterBitmap = after
            }
        }
    }
}
