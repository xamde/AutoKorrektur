# Android On-Device ML Vision & Inpainting Rules

*See complete technical specification in [IMAGE_PIPELINE_SPECIFICATION.md](file:///home/konrad/files/work/__drafts/AutoKorrektur/docs/IMAGE_PIPELINE_SPECIFICATION.md).*

## 1. Segmentation Mask Assembly Invariants
- **Coordinate Space Normalization**: When mapping YOLO proposal boxes to prototype tensors, normalize box coordinates to $[0..1]$ before scaling by prototype grid dimensions to avoid edge-pixel clamping and background over-masking.
- **Continuous Logit Upscaling**: Always upscale continuous logits to target image dimensions using `INTER_CUBIC` or `INTER_LINEAR` before applying sigmoid activation and binary thresholding to prevent staircase aliasing.
- **Morphological Edge Finishing**: Apply `Imgproc.morphologyEx(MORPH_CLOSE)` with an elliptical kernel on high-resolution masks to eliminate internal reflection/glare holes.

## 2. Inpainting & Blending Polarity
- **OpenCV `copyTo` Invariant**: `src.copyTo(dst, mask)` copies pixels where `mask > 0`. If the detection pipeline produces an overlay mask where background is 255 and target is 0, invert the mask via `Core.bitwise_not(mask, targetMask)` before blending.
- **ONNX Inpainting Binary Mask**: The ONNX model `mi-gan-512.onnx` requires a binary mask with `1` on the inpaint hole and `0` on preserved background (`Imgproc.threshold(onnxMask, onnxMask, 127.0, 1.0, THRESH_BINARY)`). Passing `255` bypasses inpainting and produces passthrough output.
- **Color Channel Order**: OpenCV Android `Utils.bitmapToMat` outputs `RGBA` order (not `BGRA`). Convert to 3-channel RGB using `Imgproc.COLOR_RGBA2RGB` and back to Bitmap using `Imgproc.COLOR_RGB2RGBA`.

## 3. Ultra HDR & Native Rendering (Android 14+)
- **Gainmap Stripping**: On API 34+, `ImageDecoder` automatically creates Gainmaps on camera captures (`Bitmap.hasGainmap() == true`). When passed to Canvas or OpenCV rasterizers, Android's `libhwui.so` native renderer segfaults in `SkiaCanvas::useGainmapShader`. Bitmaps must have `gainmap = null` stripped upon decode.
- **OpenCV JNI Lifecycle**: Always call `OpenCVLoader.initLocal()` in `Application.onCreate()`. Initializing inside Fragment companion objects fails to bind JNI symbols to `FinalizerDaemon`, triggering `UnsatisfiedLinkError` during garbage collection.

## 4. EXIF Orientation & Viewport Scaling
- **Upright Normalization**: All camera inputs with EXIF orientation tags must be rotated to true upright orientation upon loading. In-memory Bitmaps passed to UI views (e.g. Before/After split sliders) must share identical upright orientations.
- **Aspect-Ratio Fitting**: Interactive comparison views must use `FIT_CENTER` letterboxing rather than `centerCrop` to preserve the complete scene boundaries.

## 5. Two-Pass ML Test-Driven Development (TDD)
- **Quantitative Benchmark Gates**: Validate segmentation and inpainting changes with automated instrumented tests calculating Mean $IoU$, Dice Similarity Coefficient, and background false-positive masking ratios.
- **Post-Inpainting Zero-Vehicle Invariant**: Every inpainting test must run a second-pass YOLO detection using `PostInpaintingVehicleAssertionUtils.assertNoVehiclesRemain` to ensure zero residual vehicle proposals exist in the final output.
- **Red-Green Verification**: Verify test failure on the un-fixed condition before committing algorithm or threshold changes.
