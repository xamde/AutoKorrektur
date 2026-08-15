# AutoKorrektur ProGuard / R8 Optimization & Obfuscation Rules

# Preserve ONNX Runtime Native Bindings
-keep class ai.onnxruntime.** { *; }
-dontwarn ai.onnxruntime.**

# Preserve OpenCV Native Bindings
-keep class org.opencv.** { *; }
-dontwarn org.opencv.**

# Preserve Model & Data Binding Classes
-keepclassmembers class * implements androidx.viewbinding.ViewBinding {
    public static *** inflate(...);
    public static *** bind(...);
}

# Preserve TensorFlow Lite Native Bindings
-keep class org.tensorflow.lite.** { *; }
-dontwarn org.tensorflow.lite.**

# Preserve WorkManager Worker Classes (instantiated via reflection)
-keep public class * extends androidx.work.ListenableWorker {
    public <init>(android.content.Context, androidx.work.WorkerParameters);
}

# Preserve Model Data Classes for JSON Serialization
-keepclassmembers class de.konradvoelkel.android.autokorrektur.model.** { *; }
-keepclassmembers class de.konradvoelkel.android.autokorrektur.ml.model.** { *; }

# Preserve OkHttp / Okio Symbols
-keepattributes Signature
-keepattributes *Annotation*
-keepclassmembers class okhttp3.internal.publicsuffix.PublicSuffixDatabase {
    public static final java.lang.String PUBLIC_SUFFIX_RESOURCE;
}
-dontwarn okhttp3.**
-dontwarn okio.**

# Preserve CameraX Video & MediaCodec Classes
-keep class androidx.camera.video.** { *; }
-dontwarn androidx.camera.video.**
-keep class de.konradvoelkel.android.autokorrektur.video.** { *; }

# Preserve Line Numbers and Source Attributes for Crash Reporting
-keepattributes SourceFile,LineNumberTable
-renamesourcefileattribute SourceFile