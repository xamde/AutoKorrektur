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

# Preserve OkHttp / Okio Symbols
-keepattributes Signature
-keepattributes *Annotation*
-keepclassmembers class okhttp3.internal.publicsuffix.PublicSuffixDatabase {
    public static final java.lang.String PUBLIC_SUFFIX_RESOURCE;
}
-dontwarn okhttp3.**
-dontwarn okio.**

# Preserve Line Numbers for Debug Stack Traces
-keepattributes SourceFile,LineNumberTable