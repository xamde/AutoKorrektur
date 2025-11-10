package de.konradvoelkel.android.autokorrektur.ml.errors

/** Domain-specific exceptions to make error handling clearer at call sites. */
open class YoloException(message: String, cause: Throwable? = null) :
    RuntimeException(message, cause)

class ModelLoadException(message: String, cause: Throwable? = null) : YoloException(message, cause)
class InferenceException(message: String, cause: Throwable? = null) : YoloException(message, cause)
class ShapeMismatchException(message: String, cause: Throwable? = null) :
    YoloException(message, cause)
