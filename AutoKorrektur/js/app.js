import {preprocessing, processInputImage} from './processInput.js';
import {inferYolo} from "./yoloInference.js";
import {inferMiGan} from "./miGanInference.js";

window.start = start;
window.downloadResult = downloadResult;
window.dropdown = dropdown;

let inputImageMat;
let transImageMat;
let resultImageMat;
let xRatio;
let yRatio;
const segModelWidth = 640;
const segModelHeight = 640;

let segModel;
let instanceSegSession;
let miGanSession;


showLoadingIcon()



/* --- Setup DOM Elements --- */

/* -- Setup Selection Options -- */
const segmodelSelect = document.getElementById("segModel");
segModel = segmodelSelect.value;
segmodelSelect.addEventListener("change", (e) => {
    segModel = e.target.value;
    console.log(segModel)
    loadModel(segModel)
});

// Configure Downscaling value
const isMobile = /Mobi|Android|iPhone|iPad|iPod/i.test(navigator.userAgent);
const downscaleSelect = document.getElementById("downscaleMP")
if (isMobile) {
    downscaleSelect.value = 2
}


/* -- Setup Slider options -- */
var maskUpscaleSlider = document.getElementById("maskUpscale");
var maskUpscaleVal = document.getElementById("maskUpscaleVal");
maskUpscaleVal.innerHTML = (1 + maskUpscaleSlider.value * 0.01).toFixed(2); // Display the initlial slider value

maskUpscaleSlider.oninput = function () {
    maskUpscaleVal.innerHTML = (1 + this.value * 0.01).toFixed(2);
}

var downshiftSlider = document.getElementById("downshift");
var downshiftVal = document.getElementById("downshiftVal");
downshiftVal.innerHTML = (downshiftSlider.value * 0.001).toFixed(3); // Display the initlial slider value

downshiftSlider.oninput = function () {
    downshiftVal.innerHTML = (this.value * 0.001).toFixed(3);
}

var scoreThresholdSlider = document.getElementById("scoreThreshold");
var scoreThresholdVal = document.getElementById("scoreThresholdVal");
scoreThresholdVal.innerHTML = (scoreThresholdSlider.value * 0.01).toFixed(2) // Display the initlial slider value

scoreThresholdSlider.oninput = function () {
    scoreThresholdVal.innerHTML = (scoreThresholdSlider.value * 0.01).toFixed(2)
};


/* -- Setup Checkbox Options -- */
const evalModeChecker = document.getElementById("evalData")
const continueChecker = document.getElementById("continue")
const batchModeChecker = document.getElementById("batchMode");
updateStartButton(batchModeChecker.checked)

batchModeChecker.addEventListener("change", (e) => {
    updateStartButton(batchModeChecker.checked);
});


/* -- Setup File Input -- */
const fileSelectButton = document.getElementById("fileSelect");
const inputElement = document.getElementById("inputImage");

// enable the fileSelect Custom Button to open Input Element
fileSelectButton.addEventListener("click", (e) => {
        if (inputElement) {
            if (batchModeChecker.checked) {
                inputElement.setAttribute("multiple", "")
            } else {
                inputElement.removeAttribute("multiple")
            }
            inputElement.click();
        }
    },
    false);
// When the input changes, handle the images
inputElement.addEventListener('change', handleFileInputChange);


/* -- Setup ONNX-Runtime Environment -- */
ort.env.wasm.numThreads = 1; // WASM Backend is choosing #Threads
const sessionOptions = {executionProviders: ["wasm"]};
await loadModel(segModel)



/* --- Main Functions --- */

/**
 * Loads the ONNX models for segmentation and inpainting.
 * @param {string} segmentationModelName - The name of the segmentation model.
 * @param {string} inpaintModelName - The name of the inpainting model (default: "mi-gan-512").
 */
async function loadModel(segmentationModel, inpaintModel = "mi-gan-512") {
    showLoadingIcon()
    await new Promise(resolve => setTimeout(resolve, 30));

    console.time("Loading Models")
    const segModelFile = "model/" + segmentationModel + "-seg.onnx";
    const inpModelFile = "model/" + inpaintModel + ".onnx";

    const [yolo, nms, mask] = await Promise.all([
        ort.InferenceSession.create(segModelFile, sessionOptions),
        ort.InferenceSession.create("model/nms-yolov8.onnx", sessionOptions),
        ort.InferenceSession.create("model/mask-yolov8-seg.onnx", sessionOptions),
    ]);

    instanceSegSession = {yolo, nms, mask};
    miGanSession = await ort.InferenceSession.create(inpModelFile, sessionOptions);
    console.timeEnd("Loading Models")
    removeLoadingIcon()
}


/**
 * Handles the change event of the file input element.
 * @param {Event} event - The file input change event.
 */
async function handleFileInputChange(event) {
    const inputFiles = event.target.files;
    if (!inputFiles || inputFiles.length < 1) {
        console.error('Please select at least one file.');
        return;
    }

    if (inputFiles.length >= 1 && batchModeChecker.checked) {
        await handleBatchFileProcessing(inputFiles);
    } else {
        await handleSingleFileProcessing(inputFiles[0]);
    }
}

/**
 * Processes multiple files in batch mode.
 * @param {FileList} inputFiles - The list of files to process.
 */
async function handleBatchFileProcessing(inputFiles) {
    showLoadingIcon()
    await new Promise(resolve => setTimeout(resolve, 30));

    const times = []
    const fileNames = []

    const upscaleFactor = parseFloat(maskUpscaleVal.innerHTML)
    const downshift = parseFloat(downshiftVal.innerHTML)
    const scoreThreshold = parseFloat(scoreThresholdVal.innerHTML)
    for (let i = 0; i < inputFiles.length; i++) { // iterate over all Images
        const [filename, time] = await inferenceStep(inputFiles[i], upscaleFactor, downshift, scoreThreshold, evalModeChecker.checked)
        times.push(time)
        fileNames.push(filename)

        console.log("Step:", i)

    }
    if (evalModeChecker.checked) {
        await downloadCSV(fileNames, times);
    }
    removeLoadingIcon()
}

/**
 * Processes a single input file.
 * @param {File} inputFile - The file to process.
 */
async function handleSingleFileProcessing(inputFile) {
    await clearImagesContainer(); // Clear Images Container
    if (inputImageMat && inputImageMat.isDeleted() === false) {
        console.log("Free Up")
        inputImageMat.delete();
        transImageMat.delete();
    }
    console.time("Processing Input");

    [inputImageMat, transImageMat, xRatio, yRatio] = await processInputImage(inputFile, segModelWidth, segModelHeight, downscaleSelect.value);
    showImageMat(inputImageMat);
    console.timeEnd("Processing Input")
}

/**
 * Performs a single inference step for batchMode, including preprocessing, inference, and result handling.
 * Used in batch processing.
 * @param {File} inputImageFile - The input image file.
 * @param {number} upscaleFactor - The mask upscaling factor.
 * @param {number} downshift - The mask downshift amount.
 * @param {number} scoreThreshold - The score threshold for segmentation.
 * @param {boolean} [evalMode=false] - Whether evaluation mode is active.
 * @returns {Promise<[string, number]>} A tuple containing the filename and elapsed time.
 */
async function inferenceStep(inputImageFile, upscaleFactor, downshift, scoreThreshold, evalMode = false) {
    const startTime = performance.now();
    console.time("Processing Input")

    let [inputImageMat, transImageMat, xRatio, yRatio] = await processInputImage(inputImageFile, segModelWidth, segModelHeight, downscaleSelect.value);
    // showImageMat(inputImageMat);
    console.timeEnd("Processing Input")

    // Auto Start Inference and download Result
    let [result, mask] = await startInference(inputImageMat, transImageMat, xRatio, yRatio, upscaleFactor, downshift, scoreThreshold, evalMode)
    const elapsedTime = performance.now() - startTime;
    const fileName = inputImageFile.name.split(".")[0] + "_m-" + upscaleFactor + "_d-" + downshift + "_s-" + scoreThreshold + "_p-" + downscaleSelect.value

    await downloadMatAsJpeg(result, fileName)
    await new Promise(resolve => setTimeout(resolve, 300));

    if (evalMode) {
        await downloadMatAsJpeg(mask, inputImageFile.name.split(".")[0] + "_mask")
    }
    result.delete();
    mask.delete();
    inputImageMat.delete();
    transImageMat.delete();
    return [fileName, elapsedTime]
}

/**
 * @param {cv.Mat} inputImageMat - The original input image matrix with CV_8UC3 data type .
 * @param {cv.Mat} transImageMat - The transformed image matrix with CV_8UC3 data type, resized for the segmentation model.
 * @param {number} xRatio - The aspect ratio of inputs width to the biggest side of input.
 * @param {number} yRatio - The aspect ratio of inputs height to the biggest side of input.
 * @param {number} [maskUpscale=1.2] - Factor by which the segmentation mask is upscaled.
 * @param {number} [downshift=0.03] - Relative amount (percentage of image height) to shift down the mask.
 * @param {number} [scoreThreshold=0.2] - Confidence threshold for detections in the NMS.
 * @param {boolean} [evalMode=false] - If True, images are not displayed.
 * @returns {Promise<[cv.Mat, cv.Mat]>}  The resulting image matrix with CV_8UC3 data type and the used mask image matrix with CV_8UC1 data type.
 */
async function startInference(inputImageMat, transImageMat, xRatio, yRatio, maskUpscale = 1.2, downshift = 0.03, scoreThreshold = 0.2, evalMode = false) {
    await clearImagesContainer();

    console.time("Segmentaion Inference")
    const mask = await inferYolo(transImageMat, xRatio, yRatio, instanceSegSession, segModelWidth, segModelHeight, maskUpscale, scoreThreshold) // background pixel = 255 object pixel = 0
    console.timeEnd("Segmentaion Inference")

    console.time("Scaling Mask")
    const resizedMask = new cv.Mat();
    cv.resize(mask, resizedMask, new cv.Size(inputImageMat.cols, inputImageMat.rows), 0, 0, cv.INTER_LANCZOS4); // see for variants https://docs.opencv.org/3.4/da/d54/group__imgproc__transform.html#ga5bb5a1fea74ea38e1a5445ca803ff121
    mask.delete();

    const downshiftedMask = await shiftDown(resizedMask, downshift)
    console.timeEnd("Scaling Mask")

    cv.bitwise_and(resizedMask, downshiftedMask, downshiftedMask); // combine the two masks
    resizedMask.delete();

    console.time("MiGan Inference")
    const result = await inferMiGan(inputImageMat, downshiftedMask, miGanSession);
    console.timeEnd("MiGan Inference")

    if (!evalMode) {
        const overlayedImg = await layover(inputImageMat, downshiftedMask);

        await showImageMat(inputImageMat, "Original")
        await showImageMat(overlayedImg, "Mask")
        await showImageMat(result, "Result")
        overlayedImg.delete();
    }

    return [result, downshiftedMask]
}



/* --- Functions for Document Access --- */

/**
 * enables the start of Inference from Document.
 * @returns {Promise<void>}
 */
async function start() {
    showLoadingIcon()

    if (!inputImageMat || !transImageMat || !xRatio || !yRatio) {
        console.error('Please select an image');
        return;
    }

    await new Promise(resolve => setTimeout(resolve, 30));
    console.log("Manual Start")
    const upscaleFactor = parseFloat(maskUpscaleVal.innerHTML)
    const downshift = parseFloat(downshiftVal.innerHTML)
    const scoreThreshold = parseFloat(scoreThresholdVal.innerHTML)

    const [result, mask] = await startInference(inputImageMat, transImageMat, xRatio, yRatio, upscaleFactor, downshift, scoreThreshold)

    mask.delete();
    resultImageMat = result;

    if (continueChecker.checked) {
        safeDeleteMat(transImageMat);
        [transImageMat, xRatio, yRatio] = preprocessing(result, segModelWidth, segModelHeight)
        safeDeleteMat(inputImageMat);
        inputImageMat = result
    }
    removeLoadingIcon()
}

/**
 * Downloads the current result image.
 */
async function downloadResult() {
    if (resultImageMat && !resultImageMat.isDeleted()) {
        await downloadMatAsJpeg(resultImageMat, "result");
    } else {
        console.warn("No result image available to download.");
        alert("No result image to download. Please run inference first.");
    }
}

/**
 * Toggles the visibility of the dropdown menu.
 */
function dropdown() {
    document.getElementById("myDropdown").classList.toggle("show");
}



/* --- Utility Functions --- */

/**
 * Clears Displayed Images by removing all Elements from the image container.
 */
async function clearImagesContainer() {
    const container = document.getElementById('imagesContainer');
    while (container.firstChild) {
        container.removeChild(container.firstChild);
    }
}

/**
 * Show the image matrix in the image container.
 * @param inputImageMat
 * @param {String} label Label to display under the image.
 * @returns {HTMLCanvasElement} The canvas element that was created.
 */
async function showImageMat(inputImageMat, label = '') {

    const container = document.getElementById('imagesContainer');

    const imageItem = document.createElement('div');
    imageItem.className = 'imageItem';

    // Create a new canvas for this image
    const canvas = document.createElement('canvas');
    const canvasId = 'canvas_' + Date.now() + '_' + Math.floor(Math.random() * 1000);
    canvas.id = canvasId;
    canvas.width = inputImageMat.cols;
    canvas.height = inputImageMat.rows;

    // Add canvas to the container
    imageItem.appendChild(canvas);

    // Add label if provided
    if (label) {
        const labelElement = document.createElement('p');
        labelElement.textContent = label;
        imageItem.appendChild(labelElement);
    }

    container.appendChild(imageItem);

    // Render the image on the canvas
    await cv.imshow(canvasId, inputImageMat);

    return canvas;
}

/**
 * Download a cv.Mat as a JPEG file.
 * @param {cv.Mat} mat The image matrix with CV_8UC3 / CV_8UC1 / CV_8UC4 data type to download.
 * @param filename The name of the resulting file.
 */
async function downloadMatAsJpeg(mat, filename) {
    // Get dimensions and number of channels
    const cols = mat.cols;
    const rows = mat.rows;
    const channels = mat.channels();
    let imageData;
    let clampedData;

    // Convert the cv.Mat data to Uint8ClampedArray
    if (channels === 3) { // Convert RGB to RGBA
        const rgbaMat = new cv.Mat();
        cv.cvtColor(mat, rgbaMat, cv.COLOR_RGB2RGBA);
        clampedData = new Uint8ClampedArray(rgbaMat.data);
        rgbaMat.delete();
    } else if (channels === 1) { // Convert Gray to RGBA
        const rgbaMat = new cv.Mat();
        cv.cvtColor(mat, rgbaMat, cv.COLOR_GRAY2RGBA);
        clampedData = new Uint8ClampedArray(rgbaMat.data);
        rgbaMat.delete();
    } else if (channels === 4) {
        // If already RGBA, use the data directly
        clampedData = new Uint8ClampedArray(mat.data);
    } else {
        console.error("Unsupported number of channels: " + channels);
        return;
    }
    imageData = new ImageData(clampedData, cols, rows);

    // Create an off-screen canvas
    const canvas = document.createElement("canvas");
    canvas.width = cols;
    canvas.height = rows;
    const ctx = canvas.getContext("2d");

    ctx.putImageData(imageData, 0, 0);

    // Create a link to trigger the download
    const link = document.createElement("a");
    link.href = canvas.toDataURL("image/jpeg", 0.9);
    link.download = filename + ".jpeg" || "output.jpeg";
    link.click();

    if (isMobile) {
        await new Promise(resolve => setTimeout(resolve, 1000)); // Timeout for more time to accept Download in iOS
    }
}

/**
 * Generates and triggers the download of a CSV file.
 * @param {string[]} fileNames - Array of filenames.
 * @param {number[]} times - Array of corresponding processing times.
 */
async function downloadCSV(fileNames, times) {

    let csvContent = "Filename,Speed,SegScore,RealScore,KonsScore,NatScore,NewImage,OldImage\n";

    for (let i = 0; i < fileNames.length; i++) {
        csvContent += fileNames[i] + "," + times[i] + "\n";
    }

    const blob = new Blob([csvContent], {type: "text/csv;charset=utf-8;"});
    const link = document.createElement("a");
    if (link.download !== undefined) {
        const url = URL.createObjectURL(blob);
        link.setAttribute("href", url);
        link.setAttribute("download", "results.csv");
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
    }
}

/**
 * Shifts an image down and fills the top with white pixels.
 * @param {cv.Mat} inputMat The input image matrix.
 * @param {number} [shiftAmount=0.025] The amount of pixels to shift down.
 * @returns {cv.Mat} The shifted image matrix.
 */
async function shiftDown(inputMat, shiftAmount = 0.025) {
    if (shiftAmount === 0) {
        return inputMat.clone();
    }
    const originalWidth = inputMat.cols;
    const originalHeight = inputMat.rows;

    shiftAmount = Math.round(originalHeight * shiftAmount);
    // Ensure shift amount is valid
    if (shiftAmount < 0 || shiftAmount >= originalHeight) {
        console.error("Shift amount must be greater than 0 and less than image height");
        return inputMat.clone();
    }

    // Create a new matrix with same size as input
    const shiftedMat = new cv.Mat(originalHeight, originalWidth, inputMat.type());
    let roi = null;
    let destRoi = null;
    try {
        shiftedMat.setTo(new cv.Scalar(255, 255, 255, 255));

        const sourceRect = new cv.Rect(0, 0, originalWidth, originalHeight - shiftAmount);
        const destRect = new cv.Rect(0, shiftAmount, originalWidth, originalHeight - shiftAmount);

        roi = inputMat.roi(sourceRect);
        destRoi = shiftedMat.roi(destRect);
        roi.copyTo(destRoi);

        return shiftedMat;
    } catch (error) {
        console.error("Error in shiftDown:", error);
        if (shiftedMat && !shiftedMat.isDeleted()) {
            shiftedMat.delete();
        }
        return inputMat.clone();
    } finally {
        if (roi && !roi.isDeleted()) {
            roi.delete();
        }
        if (destRoi && !destRoi.isDeleted()) {
            destRoi.delete();
        }
    }
}

/**
 * Layover function to blend the original image with a red mask overlay.
 * @param {cv.Mat} original image matrix with CV_8UC3 data type.
 * @param {cv.Mat} mask image matrix with CV_8UC1 data type.
 * @returns {cv.Mat} Blended image matrix with CV_8UC3 data type.
 */
async function layover(original, mask) {
    const alpha = 0.5;

    let redOverlay = new cv.Mat(original.rows, original.cols, original.type(), new cv.Scalar(255, 0, 0));

    let blended = new cv.Mat();
    cv.addWeighted(original, 1 - alpha, redOverlay, alpha, 0, blended);

    let maskInv = new cv.Mat();
    cv.bitwise_not(mask, maskInv);

    let result = original.clone();
    blended.copyTo(result, maskInv);


    redOverlay.delete();
    blended.delete();
    maskInv.delete();

    return result;
}

/**
 * Safely deletes a cv.Mat object if it exists and is not already deleted.
 * @param {cv.Mat} mat - The cv.Mat object to delete.
 */
function safeDeleteMat(mat) {
    if (mat != null && !mat.isDeleted()) {
        mat.delete();
    }
}


/**
 * Disables or enables the start button based on the provided boolean value.
 * @param bool
 */
function updateStartButton(bool) {
    const startbutton = document.getElementById('startInference');
    if (bool) {
        startbutton.setAttribute("disabled", "");
    } else if (startbutton.hasAttribute("disabled")) {
        startbutton.removeAttribute("disabled");
    }
}

/**
 * Shows a loading icon on the start button and disables it.
 */
function showLoadingIcon() {
    const startbutton = document.getElementById('startInference');
    startbutton.innerHTML = '<i class="fa fa-spinner fa-spin"></i> ';
    startbutton.setAttribute("disabled", "");
}

/**
 * Removes the loading icon from the start button and re-enables it.
 */
function removeLoadingIcon() {

    const startbutton = document.getElementById('startInference');
    startbutton.innerHTML = '<i class="fa fa-play" aria-hidden="true"></i> &nbsp;Start';
    if (!batchModeChecker.checked) {
        startbutton.removeAttribute("disabled");
    }
}
