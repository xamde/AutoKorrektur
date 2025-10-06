// Simple test script to debug mask conversion issue
// This would help identify the specific problem in the conversion process

import org.opencv.core.*
import org.opencv.imgproc.Imgproc

fun debugMaskConversion() {
    // Simulate the mask conversion process step by step
    println("[DEBUG] === DEBUGGING MASK CONVERSION ===")
    
    // Step 1: Create a simple test mask (160x160)
    val testMask = Mat.zeros(160, 160, CvType.CV_32FC1)
    // Add some test data - create a circle in the center
    val center = Point(80.0, 80.0)
    val radius = 30
    Imgproc.circle(testMask, center, radius, Scalar(0.8), -1)
    
    val beforeSigmoid = Core.minMaxLoc(testMask)
    println("[DEBUG] Test mask before sigmoid: min=${beforeSigmoid.minVal}, max=${beforeSigmoid.maxVal}")
    
    // Step 2: Apply sigmoid (this is where the issue might be)
    applySigmoid(testMask)
    
    val afterSigmoid = Core.minMaxLoc(testMask)
    println("[DEBUG] Test mask after sigmoid: min=${afterSigmoid.minVal}, max=${afterSigmoid.maxVal}")
    
    // Step 3: Apply threshold
    Imgproc.threshold(testMask, testMask, 0.5, 1.0, Imgproc.THRESH_BINARY)
    
    val afterThreshold = Core.minMaxLoc(testMask)
    println("[DEBUG] Test mask after threshold: min=${afterThreshold.minVal}, max=${afterThreshold.maxVal}")
    
    // Step 4: Convert to CV_8UC1
    testMask.convertTo(testMask, CvType.CV_8UC1, 255.0)
    
    val afterConvert = Core.minMaxLoc(testMask)
    println("[DEBUG] Test mask after convert to 8UC1: min=${afterConvert.minVal}, max=${afterConvert.maxVal}")
    
    // Step 5: Test subtraction from white background
    val whiteBg = Mat.ones(160, 160, CvType.CV_8UC1)
    whiteBg.setTo(Scalar(255.0))
    
    val beforeSubtract = Core.minMaxLoc(whiteBg)
    println("[DEBUG] White background before subtract: min=${beforeSubtract.minVal}, max=${beforeSubtract.maxVal}")
    
    Core.subtract(whiteBg, testMask, whiteBg)
    
    val afterSubtract = Core.minMaxLoc(whiteBg)
    println("[DEBUG] Result after subtract: min=${afterSubtract.minVal}, max=${afterSubtract.maxVal}")
    
    // Count black pixels
    val blackMask = Mat()
    Core.inRange(whiteBg, Scalar(0.0), Scalar(10.0), blackMask)
    val blackPixels = Core.countNonZero(blackMask)
    val totalPixels = whiteBg.rows() * whiteBg.cols()
    println("[DEBUG] Black pixels: $blackPixels / $totalPixels (${blackPixels.toDouble() / totalPixels.toDouble() * 100}%)")
    
    blackMask.release()
    testMask.release()
    whiteBg.release()
}

fun applySigmoid(mat: Mat) {
    // Implementation from YoloInferenceTFLite
    val rows = mat.rows()
    val cols = mat.cols()
    
    for (row in 0 until rows) {
        val rowData = FloatArray(cols)
        mat.get(row, 0, rowData)
        
        for (col in 0 until cols) {
            val x = rowData[col]
            rowData[col] = (1.0f / (1.0f + kotlin.math.exp(-x.toDouble()))).toFloat()
        }
        
        mat.put(row, 0, rowData)
    }
}