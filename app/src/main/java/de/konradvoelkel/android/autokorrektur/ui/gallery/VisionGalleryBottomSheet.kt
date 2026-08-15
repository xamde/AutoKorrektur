package de.konradvoelkel.android.autokorrektur.ui.gallery

import android.net.Uri
import android.os.Bundle
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import android.widget.ImageView
import android.widget.TextView
import androidx.recyclerview.widget.RecyclerView
import com.google.android.material.bottomsheet.BottomSheetDialogFragment
import de.konradvoelkel.android.autokorrektur.R
import de.konradvoelkel.android.autokorrektur.utils.ImageExportManager

/**
 * Bottom sheet displaying the grid of past car-free shots captured with AutoKorrektur.
 */
class VisionGalleryBottomSheet : BottomSheetDialogFragment() {

    var onImageSelected: ((Uri) -> Unit)? = null

    override fun onCreateView(
        inflater: LayoutInflater,
        container: ViewGroup?,
        savedInstanceState: Bundle?
    ): View? {
        return inflater.inflate(R.layout.bottom_sheet_vision_gallery, container, false)
    }

    override fun onViewCreated(view: View, savedInstanceState: Bundle?) {
        super.onViewCreated(view, savedInstanceState)

        val rvGallery = view.findViewById<RecyclerView>(R.id.rvVisionGallery)
        val tvCount = view.findViewById<TextView>(R.id.tvGalleryCount)
        val tvEmpty = view.findViewById<TextView>(R.id.tvEmptyGallery)

        val exportManager = ImageExportManager(requireContext())
        val images = exportManager.getRecentAutoKorrekturImages(limit = 60)

        if (images.isEmpty()) {
            tvEmpty.visibility = View.VISIBLE
            rvGallery.visibility = View.GONE
            tvCount.text = getString(R.string.gallery_zero_shots)
        } else {
            tvEmpty.visibility = View.GONE
            rvGallery.visibility = View.VISIBLE
            tvCount.text = getString(R.string.gallery_n_shots, images.size)

            rvGallery.adapter = VisionGalleryAdapter(images) { selectedUri ->
                dismiss()
                onImageSelected?.invoke(selectedUri)
            }
        }
    }

    private class VisionGalleryAdapter(
        private val images: List<Uri>,
        private val onClick: (Uri) -> Unit
    ) : RecyclerView.Adapter<VisionGalleryAdapter.ViewHolder>() {

        class ViewHolder(view: View) : RecyclerView.ViewHolder(view) {
            val ivThumbnail: ImageView = view.findViewById(R.id.ivGalleryThumbnail)
        }

        override fun onCreateViewHolder(parent: ViewGroup, viewType: Int): ViewHolder {
            val view = LayoutInflater.from(parent.context)
                .inflate(R.layout.item_vision_gallery, parent, false)
            return ViewHolder(view)
        }

        override fun onBindViewHolder(holder: ViewHolder, position: Int) {
            val uri = images[position]
            try {
                holder.ivThumbnail.setImageURI(uri)
            } catch (_: Exception) {}

            holder.itemView.setOnClickListener {
                onClick(uri)
            }
        }

        override fun getItemCount(): Int = images.size
    }
}
