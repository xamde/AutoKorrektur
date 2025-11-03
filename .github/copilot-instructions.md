# AutoKorrektur - Copilot Instructions

## Repository Overview

AutoKorrektur is an Android application that automatically removes cars from photographs using machine learning. This is a reimplementation of the [AutoKorrektur Web Version](https://github.com/BenB2/AutoKorrektur), based on Till Schellscheidt's Bachelor Thesis "Autokorrektur - Automatisierte Objektersetzung in Fotos".

The app's purpose is to help users visualize cities with fewer cars by processing photos entirely on-device, with no server-side processing required.

## Technology Stack

- **Language**: Kotlin
- **Platform**: Android (minSdk 29, targetSdk 36)
- **Build System**: Gradle with Kotlin DSL
- **Java Version**: JDK 17
- **Key Libraries**:
  - AndroidX Core, AppCompat, ConstraintLayout
  - Material Design Components
  - Navigation Component
  - ONNX Runtime Android
  - TensorFlow Lite
  - OpenCV for Android

### Machine Learning Models

- **Instance Segmentation**: YOLOv11-seg (for detecting cars)
- **Inpainting**: MI-GAN (for removing detected objects and filling in backgrounds)

## Project Structure

```
app/
├── src/
│   ├── main/java/de/konradvoelkel/android/autokorrektur/
│   │   ├── MainActivity.kt - Main activity entry point
│   │   ├── FirstFragment.kt - Primary UI fragment with image processing
│   │   ├── ml/ - Machine learning components
│   │   │   ├── ImageProcessor.kt - Image processing pipeline
│   │   │   ├── YoloInferenceTFLite.kt - YOLO segmentation model
│   │   │   └── MiGanInference.kt - MI-GAN inpainting model
│   │   └── utils/ - Utility classes (logging, debugging)
│   ├── androidTest/ - Instrumentation tests
│   └── test/ - Unit tests
├── build.gradle.kts - App-level build configuration
└── how-to-debug.adoc - Debugging tips
```

## Build and Test

### Building the Project

This is an Android project built with Gradle. Since there's no `gradlew` wrapper checked in, you'll need to use Android Studio or install Gradle separately.

**Using Android Studio**:
1. Open the project in Android Studio
2. Sync Gradle files
3. Build > Make Project

**Using Gradle** (if installed):
```bash
gradle build
```

### Running Tests

**Unit Tests**:
```bash
gradle test
```

**Android Instrumentation Tests**:
```bash
gradle connectedAndroidTest
```

Key test files:
- `app/src/test/java/de/konradvoelkel/android/autokorrektur/ExampleUnitTest.kt`
- `app/src/androidTest/java/de/konradvoelkel/android/autokorrektur/` - Various component tests including ML, UI, and image processing tests

## Code Style

- Follow official Kotlin code style (`kotlin.code.style=official` in gradle.properties)
- Use view binding for UI components
- Package structure follows domain-driven organization (ml, utils, etc.)

## Important Constraints & Known Issues

### Known Bugs (as documented in README)
- Pictures must be taken in landscape mode (portrait mode support planned)
- High-resolution pictures can cause the app to fail (work in progress)

### Technical Constraints
- Minimum SDK: Android 10 (API 29)
- All ML processing happens on-device using ONNX Runtime and TensorFlow Lite
- The app requires significant memory for ML model inference (org.gradle.jvmargs=-Xmx2048m)

### Resource Considerations
- YOLO and MI-GAN models are resource-intensive
- Large images may cause OutOfMemory errors
- Consider image downscaling for processing large inputs

## Licensing

**Critical**: This project is licensed under **GNU AGPLv3** due to the YOLOv11-seg model dependency. Any code changes must comply with AGPLv3 terms:
- All derivative works must also be licensed under AGPLv3
- Source code must be made available
- Network use triggers distribution requirements

## Development Tips

### Debugging
- See `app/how-to-debug.adoc` for debugging tips
- Use `adb` commands for testing file access (documented in how-to-debug.adoc)
- AppLogger is initialized in MainActivity.onCreate() - use it for logging

### Adding Dependencies
- Dependencies are defined in `app/build.gradle.kts`
- Use version catalog (libs.versions.toml) for dependency management
- Ensure any new ML dependencies are compatible with on-device inference

### Testing Strategy
- Write unit tests for pure Kotlin logic
- Use Android instrumentation tests for UI and ML components
- Test with various image sizes and orientations
- Test memory constraints with large images

## Common Pitfalls

1. **Memory Management**: ML models require significant memory. Always test with realistic image sizes.
2. **Image Orientation**: The app currently only supports landscape images.
3. **Build Cache**: Use `org.gradle.configuration-cache=true` is enabled - some plugins may not be compatible.
4. **OpenCV**: Ensure OpenCV native libraries are properly loaded before use.
5. **License Compliance**: Remember AGPLv3 requirements when adding features or dependencies.

## Making Changes

When making changes to this codebase:
1. Maintain compatibility with minSdk 29
2. Test on both small and large images
3. Verify ML model integration still works
4. Run both unit and instrumentation tests
5. Consider memory implications of changes
6. Ensure AGPLv3 license compliance
