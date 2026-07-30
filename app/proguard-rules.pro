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

# Preserve Line Numbers for Debug Stack Traces
-keepattributes SourceFile,LineNumberTable