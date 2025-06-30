# AutoKorrektur Android App Conversion Plan

## Overview
This document outlines the plan to convert the AutoKorrektur web application (car detection and inpainting) into a native Android application for Play Store deployment.

## Current Web Application Analysis

### Technology Stack
- **Frontend**: HTML, CSS, JavaScript
- **ML Inference**: ONNX Runtime Web with WASM backend
- **Computer Vision**: OpenCV.js
- **Models**: 
  - YOLOv11-seg (nano/small/medium) for car detection/segmentation
  - MI-GAN (512x512) for inpainting
  - Additional NMS and mask processing models

### Core Functionality
1. **Image Processing Pipeline**:
   - File input → Preprocessing → YOLO segmentation → Mask processing → MI-GAN inpainting → Result
2. **Features**:
   - Single and batch image processing
   - Configurable parameters (mask upscaling, score threshold, model selection)
   - Mobile optimization (automatic downscaling)
   - Evaluation mode for performance measurement
   - Iterative processing capability

### Key Files Analysis
- `app.js`: Main application logic, model loading, UI controls
- `yoloInference.js`: YOLO segmentation implementation
- `miGanInference.js`: MI-GAN inpainting implementation  
- `processInput.js`: Image preprocessing pipeline
- `model/`: ONNX model files (~100MB total)

## Android Development Approach Options

### Option 1: Native Android (Kotlin/Java) - RECOMMENDED
**Pros:**
- Best performance for ML inference
- Full access to Android APIs
- Better memory management
- Native ONNX Runtime Android support
- OpenCV Android SDK available
- Smaller APK size potential
- Better Play Store optimization

**Cons:**
- Complete rewrite required
- Longer development time
- Need Android development expertise

**Technology Stack:**
- **Language**: Kotlin
- **ML Framework**: ONNX Runtime Android
- **Computer Vision**: OpenCV Android SDK
- **UI**: Jetpack Compose (modern Android UI)
- **Architecture**: MVVM with Repository pattern

### Option 2: React Native
**Pros:**
- Can reuse some JavaScript logic
- Cross-platform potential
- Faster development

**Cons:**
- Limited ONNX Runtime support
- Performance overhead for ML inference
- Complex native module integration needed
- Larger app size

### Option 3: Flutter
**Pros:**
- Good performance
- Modern UI framework
- Cross-platform

**Cons:**
- Limited ONNX Runtime support
- Need to rewrite everything
- Dart language learning curve

### Option 4: Capacitor/Cordova (Web View)
**Pros:**
- Minimal code changes
- Fast conversion

**Cons:**
- Poor performance for ML inference
- Large app size
- Limited native features
- Not suitable for intensive ML workloads

## Recommended Approach: Native Android with Kotlin

### Architecture Design

```
┌─────────────────────────────────────────┐
│                UI Layer                 │
│  (Jetpack Compose Activities/Fragments) │
├─────────────────────────────────────────┤
│              ViewModel Layer            │
│     (Business Logic & State Management) │
├─────────────────────────────────────────┤
│             Repository Layer            │
│        (Data Access & Coordination)     │
├─────────────────────────────────────────┤
│              Service Layer              │
│  ┌─────────────┬─────────────┬─────────┐ │
│  │   Image     │    YOLO     │ MI-GAN  │ │
│  │ Processing  │  Inference  │Inference│ │
│  │   Service   │   Service   │ Service │ │
│  └─────────────┴─────────────┴─────────┘ │
├─────────────────────────────────────────┤
│               Data Layer                │
│  ┌─────────────┬─────────────┬─────────┐ │
│  │   Model     │   OpenCV    │  File   │ │
│  │  Manager    │   Manager   │ Manager │ │
│  └─────────────┴─────────────┴─────────┘ │
└─────────────────────────────────────────┘
```

### Implementation Plan

#### Phase 1: Project Setup & Infrastructure (Week 1-2)
1. **Create Android Project**
   - Set up Kotlin Android project with Jetpack Compose
   - Configure build.gradle for ONNX Runtime and OpenCV dependencies
   - Set up project structure following Android best practices

2. **Dependency Integration**
   - Add ONNX Runtime Android: `implementation 'com.microsoft.onnxruntime:onnxruntime-android:1.16.3'`
   - Add OpenCV Android SDK: `implementation 'org.opencv:opencv-android:4.8.0'`
   - Add Jetpack Compose dependencies
   - Add image processing libraries (Coil for image loading)

3. **Model Asset Management**
   - Add ONNX models to assets folder
   - Implement model loading and caching system
   - Handle large model files (consider downloading on first run)

#### Phase 2: Core Services Implementation (Week 3-5)
1. **Image Processing Service**
   - Port `processInput.js` functionality
   - Implement file reading, preprocessing, resizing
   - Handle Android-specific image formats and permissions

2. **YOLO Inference Service**
   - Port `yoloInference.js` to Kotlin
   - Implement ONNX Runtime Android integration
   - Handle tensor operations and NMS processing

3. **MI-GAN Inference Service**
   - Port `miGanInference.js` to Kotlin
   - Implement inpainting pipeline
   - Optimize memory usage for mobile devices

4. **OpenCV Integration**
   - Set up OpenCV Android SDK
   - Port image manipulation operations
   - Implement mask processing and image transformations

#### Phase 3: UI Implementation (Week 6-7)
1. **Main Activity & Navigation**
   - Implement main screen with image selection
   - Add settings/options screen
   - Implement result viewing screen

2. **Image Processing UI**
   - Image picker integration (Gallery, Camera)
   - Progress indicators for processing
   - Parameter adjustment controls (sliders, dropdowns)

3. **Results Display**
   - Before/after image comparison
   - Download/share functionality
   - Batch processing UI

#### Phase 4: Android-Specific Features (Week 8)
1. **Permissions & File Access**
   - Request camera and storage permissions
   - Handle scoped storage (Android 10+)
   - Implement proper file management

2. **Performance Optimization**
   - Background processing with WorkManager
   - Memory management for large images
   - Battery optimization considerations

3. **User Experience**
   - Loading states and progress indicators
   - Error handling and user feedback
   - Offline functionality

#### Phase 5: Testing & Optimization (Week 9-10)
1. **Testing**
   - Unit tests for core services
   - Integration tests for ML pipeline
   - UI tests with Espresso
   - Performance testing on various devices

2. **Optimization**
   - Model quantization for smaller size
   - Memory usage optimization
   - Processing speed improvements

#### Phase 6: Play Store Preparation (Week 11-12)
1. **App Metadata**
   - App icon and screenshots
   - Store listing description
   - Privacy policy and terms of service

2. **Release Preparation**
   - Code signing and release build
   - ProGuard/R8 optimization
   - APK size optimization

3. **Play Store Submission**
   - Create Play Console account
   - Upload APK and metadata
   - Handle review process

### Technical Implementation Details

#### Model Integration
```kotlin
class ModelManager {
    private var yoloSession: OrtSession? = null
    private var miGanSession: OrtSession? = null
    
    suspend fun loadModels(context: Context) {
        val ortEnvironment = OrtEnvironment.getEnvironment()
        
        // Load YOLO models
        yoloSession = ortEnvironment.createSession(
            context.assets.open("yolo11s-seg.onnx").readBytes()
        )
        
        // Load MI-GAN model
        miGanSession = ortEnvironment.createSession(
            context.assets.open("mi-gan-512.onnx").readBytes()
        )
    }
}
```

#### Image Processing Pipeline
```kotlin
class ImageProcessingService {
    suspend fun processImage(
        imageUri: Uri,
        parameters: ProcessingParameters
    ): ProcessingResult {
        // 1. Load and preprocess image
        val originalMat = loadImageFromUri(imageUri)
        val preprocessedMat = preprocessForYolo(originalMat)
        
        // 2. Run YOLO inference
        val mask = yoloInferenceService.infer(preprocessedMat, parameters)
        
        // 3. Run MI-GAN inference
        val result = miGanInferenceService.infer(originalMat, mask)
        
        return ProcessingResult(original = originalMat, result = result)
    }
}
```

#### UI with Jetpack Compose
```kotlin
@Composable
fun MainScreen(viewModel: MainViewModel) {
    Column {
        ImageSelector(
            onImageSelected = viewModel::selectImage
        )
        
        ParameterControls(
            parameters = viewModel.parameters,
            onParametersChanged = viewModel::updateParameters
        )
        
        ProcessButton(
            enabled = viewModel.canProcess,
            onClick = viewModel::startProcessing
        )
        
        ResultDisplay(
            result = viewModel.processingResult
        )
    }
}
```

### Challenges & Solutions

#### Challenge 1: Large Model Files
**Problem**: ONNX models are ~100MB total, exceeding APK size limits
**Solutions**:
- Use Android App Bundle for dynamic delivery
- Download models on first app launch
- Implement model quantization to reduce size
- Consider cloud-based inference for some models

#### Challenge 2: Memory Management
**Problem**: Processing large images with ML models requires significant memory
**Solutions**:
- Implement image downscaling based on device capabilities
- Use memory-mapped files for models
- Implement proper cleanup of OpenCV Mats and ONNX tensors
- Add memory monitoring and warnings

#### Challenge 3: Processing Performance
**Problem**: Mobile devices have limited computational power
**Solutions**:
- Use NNAPI acceleration when available
- Implement GPU acceleration for OpenCV operations
- Add processing queue for batch operations
- Provide quality vs speed trade-offs

#### Challenge 4: Android Permissions
**Problem**: Need access to camera and storage
**Solutions**:
- Implement proper permission request flow
- Handle scoped storage for Android 10+
- Provide clear explanations for permission needs
- Graceful degradation when permissions denied

### Deployment Strategy

#### APK Size Optimization
- Use Android App Bundle
- Enable ProGuard/R8 code shrinking
- Compress model files
- Remove unused resources

#### Play Store Listing
- **Title**: "AutoKorrektur - Car Removal from Photos"
- **Description**: Focus on urban planning, photography enhancement
- **Keywords**: photo editing, car removal, urban planning, inpainting
- **Category**: Photography
- **Target Audience**: Urban planners, photographers, general users

#### Monetization Options
- Free app with ads
- Premium version without ads
- In-app purchases for additional features
- Subscription for cloud processing

### Timeline Summary
- **Total Duration**: 12 weeks
- **MVP Ready**: Week 8
- **Play Store Ready**: Week 12
- **Team Size**: 2-3 developers (Android, ML, UI/UX)

### Success Metrics
- **Technical**: Processing time < 10 seconds on mid-range devices
- **User Experience**: App rating > 4.0 stars
- **Performance**: Memory usage < 1GB during processing
- **Adoption**: 10K+ downloads in first 3 months

This plan provides a comprehensive roadmap for converting the AutoKorrektur web application into a successful Android app ready for Play Store deployment.
