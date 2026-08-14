# AutoKorrektur: Image Pipeline & Tensor Specification

This document provides a comprehensive technical reference for all image properties, color spaces, channel layouts, memory management invariants, and tensor conventions across the Android client, OpenCV processing, YOLO segmentation, and Mi-GAN / SDXL inpainting backends.

---

## 1. Android Bitmap & Color Space Invariants

### 1.1 Bitmap Configurations & Color Space
- **Standard Format**: All in-memory Bitmaps must use `Bitmap.Config.ARGB_8888` with standard sRGB color space.
- **Hardware Bitmaps**: ML models (TFLite, ONNX, OpenCV) cannot read from `Bitmap.Config.HARDWARE`. Ensure all bitmaps are software-backed ARGB_8888.

### 1.2 Ultra HDR & Gainmap Layer Handling (Android 14+ / API 34+)
- **Problem**: In Android 14/15/17 (API 34+), `ImageDecoder` automatically attaches an Ultra HDR `Gainmap` layer to camera captures. When passed to Canvas rasterizers (`drawBitmap`) or OpenCV scaling routines, Android's `libhwui.so` native graphics engine invokes `SkiaCanvas::useGainmapShader`, resulting in a native `SIGSEGV` (null pointer dereference).
- **Rule**: Explicitly strip gainmaps (`bitmap.gainmap = null`) immediately upon loading or decoding in `UriLoader.kt` and `BitmapMemoryUtils.kt`.

### 1.3 Memory Limits & Max Megapixels
- **Max Processing Resolution**: Default limit is 8.0 Megapixels (~2040×1536) in `UriLoader.kt` using power-of-two `inSampleSize` to prevent Out-Of-Memory (OOM) exceptions on large 50MP–200MP camera captures.
- **Display Resolution**: Display Bitmaps in `FirstFragment` and `BeforeAfterSliderView` are capped at 1920px max dimension via `BitmapMemoryUtils.createScaledBitmapForDisplay`.

---

## 2. OpenCV Android SDK Conventions

### 2.1 Native Library Initialization
- **Rule**: `OpenCVLoader.initLocal()` **must** be called in `AutoKorrekturApplication.onCreate()`.
- **Reason**: Lazy initialization inside Fragment companion objects fails to bind JNI symbols to the VM's `FinalizerDaemon` thread. When `CleanableMat` is garbage-collected on background threads, missing JNI symbols trigger an uncatchable `UnsatisfiedLinkError: org.opencv.core.CleanableMat.n_delete(long)`, crashing the application.

### 2.2 Channel Layout & Color Space Conversions
- **`Utils.bitmapToMat`**: Outputs a 4-channel `CV_8UC4` Mat in **RGBA** order (NOT BGRA).
- **RGB Pipeline Mat**: Converted using `Imgproc.cvtColor(bgraMat, rgbMat, Imgproc.COLOR_RGBA2RGB)` to produce `CV_8UC3` in **RGB** order.
- **Mat to Bitmap**: Converted using `Imgproc.cvtColor(rgbMat, rgbaMat, Imgproc.COLOR_RGB2RGBA)` followed by `Utils.matToBitmap`.
- **Multi-channel Normalization**: Always use `Core.split(src, channels)` and `Core.merge(rgb3, dst)` rather than manual buffer iteration when converting arbitrary channel Mats to 3-channel RGB.

---

## 3. YOLO Segmentation Model (TFLite Engine)

| Property | Value | Notes |
| :--- | :--- | :--- |
| **Input Shape** | `[1, 640, 640, 3]` | Float32 tensor |
| **Channel Order** | `RGB` | Normalized to `[0.0, 1.0]` |
| **Stride Alignment** | 32 pixels | Letterboxed to square with black borders (`Scalar(0,0,0)`) |
| **Bounding Box Format** | Normalized `[cx, cy, w, h]` | Normalized to `[0.0, 1.0]` relative to 640×640 grid |
| **Prototypes Output** | `[1, 160, 160, 32]` | Continuous linear mask features |
| **Mask Output Polarity** | `0` = Vehicle, `255` = Background | Grayscale `CV_8UC1` Mat matching original image dimensions |

### 3.1 Mask Assembly Pipeline
1. **Extract Prototypes**: 32 channels of 160×160 continuous linear feature maps.
2. **Crop & Linear Combination**: Multiply detection mask coefficients against prototype crop.
3. **Continuous Upscaling**: Upscale continuous logits to target resolution using `INTER_CUBIC` *before* thresholding to avoid pixelation.
4. **Sigmoid & Threshold**: `1 / (1 + exp(-x))` with threshold `0.4` $\to$ binary 8-bit mask (255 where car is).
5. **Guided Filter**: Edge-preserving refinement guided by the original RGB image to snap mask edges to vehicle contours.
6. **Overlay Assembly**: Subtracted from white background Mat (`overlay.setTo(255)` $\to$ car is `0`, background is `255`).

---

## 4. MI-GAN Inpainting Model (ONNX Runtime)

| Property | Value | Notes |
| :--- | :--- | :--- |
| **Input `image`** | `[1, 3, 512, 512]` | `UINT8` in `CHW` layout (Channel, Height, Width), RGB order |
| **Input `mask`** | `[1, 1, 512, 512]` | `UINT8` in `CHW` layout |
| **Inpainting Mask Values** | `1` = Hole (Vehicle), `0` = Keep (Background) | Inverted from YOLO mask via `Core.bitwise_not` and thresholded to `1.0` binary (`Imgproc.threshold(onnxMask, onnxMask, 127.0, 1.0, THRESH_BINARY)`) |
| **Padding Rule** | Background (`255.0`) | Non-image border padding in `prepareSquareInputs` must be marked as Background to avoid inpainting the padding |
| **Output `result`** | `[1, 3, 512, 512]` | `UINT8` in `CHW` layout, RGB order |

### 4.1 Output Reconstruction & Blending
1. **Reorder CHW to HWC**: Convert ONNX output byte buffer from planar `CHW` to interleaved `HWC` (RGB).
2. **Unpad & Resize**: Resize 512×512 to original square dimension and crop the active region `Rect(0, 0, origWidth, origHeight)`.
3. **Masked Blending**:
   ```kotlin
   val finalBlendedMat = processedImage.clone()
   val carMask = Mat()
   Core.bitwise_not(processedMask, carMask) // carMask has 255 on vehicle
   unpaddedInpainted.copyTo(finalBlendedMat, carMask) // copies inpainting strictly onto car region
   ```

---

## 5. UI Presentation & Viewport Scaling

### 5.1 `BeforeAfterSliderView`
- **Aspect Ratio**: Uses `FIT_CENTER` bounding-box math so landscape (4:3, 16:9) and portrait images are letterboxed with accurate proportions rather than being aggressively cropped.
- **Slider Coordinates**: Normalized `0.0` (all After / Inpainted) to `1.0` (all Before / Original).
- **Floating Labels & Glyphs**: Renders `VORHER` on the left and `NACHHER` on the right with universal `◀  ▶` arrows.

---

## 6. Two-Pass ML Test Verification Invariant
- **Pass 1 (Pre-Inpainting)**: Runs YOLO Segmentation $\to$ Asserts $\ge 1$ vehicle detected on input image.
- **Pass 2 (Post-Inpainting)**: Runs YOLO Detection on output image $\to$ Asserts **0 residual vehicles remain** via `PostInpaintingVehicleAssertionUtils`.
