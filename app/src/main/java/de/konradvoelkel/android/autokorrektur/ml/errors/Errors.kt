package de.konradvoelkel.android.autokorrektur.ml.errors

/** Domain-specific exceptions to make error handling clearer at call sites. */
sealed class YoloException(message: String, cause: Throwable? = null) :
    RuntimeException(message, cause)

class ModelLoadException(message: String, cause: Throwable? = null) : YoloException(message, cause)
class InferenceException(message: String, cause: Throwable? = null) : YoloException(message, cause)
class ShapeMismatchException(message: String, cause: Throwable? = null) :
    YoloException(message, cause)

class ModelNotInitializedException(message: String) : YoloException(message)

/** Exceptions related to inpainting operations. */
open class InpaintException(message: String, cause: Throwable? = null) : RuntimeException(message, cause)

class CloudInferenceException(message: String, cause: Throwable? = null) : InpaintException(message, cause)
class QuotaExceededException(message: String) : InpaintException(message)
