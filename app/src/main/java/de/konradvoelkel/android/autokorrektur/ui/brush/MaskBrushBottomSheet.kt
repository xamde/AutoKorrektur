package de.konradvoelkel.android.autokorrektur.ui.brush

import android.app.Dialog
import android.graphics.Bitmap
import android.os.Bundle
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import android.widget.FrameLayout
import com.google.android.material.bottomsheet.BottomSheetBehavior
import com.google.android.material.bottomsheet.BottomSheetDialog
import com.google.android.material.bottomsheet.BottomSheetDialogFragment
import de.konradvoelkel.android.autokorrektur.R
import de.konradvoelkel.android.autokorrektur.databinding.BottomSheetMaskBrushBinding
import org.opencv.core.Mat

/**
 * Fullscreen bottom sheet modal for interactive manual mask brushing & touch-up.
 */
class MaskBrushBottomSheet : BottomSheetDialogFragment() {

    private var _binding: BottomSheetMaskBrushBinding? = null
    private val binding get() = _binding!!

    var sourceBitmap: Bitmap? = null
    var initialMaskBitmap: Bitmap? = null
    var onMaskApplied: ((Mat) -> Unit)? = null

    override fun onCreateDialog(savedInstanceState: Bundle?): Dialog {
        val dialog = super.onCreateDialog(savedInstanceState) as BottomSheetDialog
        dialog.setOnShowListener {
            val bottomSheet = dialog.findViewById<FrameLayout>(com.google.android.material.R.id.design_bottom_sheet)
            if (bottomSheet != null) {
                val behavior = BottomSheetBehavior.from(bottomSheet)
                behavior.state = BottomSheetBehavior.STATE_EXPANDED
                behavior.isDraggable = false
                bottomSheet.layoutParams.height = ViewGroup.LayoutParams.MATCH_PARENT
            }
        }
        return dialog
    }

    override fun onCreateView(
        inflater: LayoutInflater,
        container: ViewGroup?,
        savedInstanceState: Bundle?
    ): View {
        _binding = BottomSheetMaskBrushBinding.inflate(inflater, container, false)
        return binding.root
    }

    override fun onViewCreated(view: View, savedInstanceState: Bundle?) {
        super.onViewCreated(view, savedInstanceState)

        sourceBitmap?.let { bmp ->
            binding.maskBrushView.setup(bmp, initialMaskBitmap)
        }

        binding.toggleGroupTool.check(R.id.btnToolBrush)
        binding.toggleGroupTool.addOnButtonCheckedListener { _, checkedId, isChecked ->
            if (isChecked) {
                when (checkedId) {
                    R.id.btnToolEraser -> binding.maskBrushView.setToolMode(MaskBrushView.ToolMode.ERASER)
                    else -> binding.maskBrushView.setToolMode(MaskBrushView.ToolMode.BRUSH)
                }
            }
        }

        binding.sliderBrushSize.addOnChangeListener { _, value, _ ->
            binding.maskBrushView.setBrushSize(value)
        }

        binding.btnClearMask.setOnClickListener {
            binding.maskBrushView.clearMask()
        }

        binding.btnApplyMask.setOnClickListener {
            val maskMat = binding.maskBrushView.exportSubtractiveMaskMat()
            onMaskApplied?.invoke(maskMat)
            dismiss()
        }
    }

    override fun onDestroyView() {
        super.onDestroyView()
        _binding = null
    }

    companion object {
        fun newInstance(
            sourceBitmap: Bitmap,
            initialMaskBitmap: Bitmap? = null
        ): MaskBrushBottomSheet {
            return MaskBrushBottomSheet().apply {
                this.sourceBitmap = sourceBitmap
                this.initialMaskBitmap = initialMaskBitmap
            }
        }
    }
}
