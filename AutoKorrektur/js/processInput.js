export {processInputImage, preprocessing}

/**
 * @param  file The file Url of the image.
 * @param modelWidth The width of the model.
 * @param modelHeight The height of the model.
 * @param {Number|null} downscale The max Megapixel to be downscaled to or null.
 * @returns {Promise<[cv.Mat, cv.Mat, number, number]>} A promise that resolves to an array containing:
 *   - The original RGB Matrix with CV_8UC3 Data type,
 *   - The transformed RGB cv.Mat with CV_8UC3 Data type,
 *   - xRatio of the Image,
 *   - yRatio of the Image.
 */
async function processInputImage(file, modelWidth, modelHeight, downscale = null) {
    let imageElement = await readImageFile(file);

    const cvImageMat = cv.imread(imageElement);
    imageElement = null
    const cvRgbImageMat = new cv.Mat(cvImageMat.rows, cvImageMat.cols, cv.CV_8UC3);
    cv.cvtColor(cvImageMat, cvRgbImageMat, cv.COLOR_RGBA2RGB, 0);
    cvImageMat.delete();
    const [transImageMat, xRatio, yRatio] = preprocessing(cvRgbImageMat, modelWidth, modelHeight); // Image Letterbox-reshape to model input size

    if (downscale !== null) {
        const currentMegapixels = (cvRgbImageMat.rows * cvRgbImageMat.cols) / 1000000;

        if (currentMegapixels > downscale) {
            const scaleFactor = Math.sqrt(downscale / currentMegapixels);

            const newWidth = Math.round(cvRgbImageMat.cols * scaleFactor);
            const newHeight = Math.round(cvRgbImageMat.rows * scaleFactor);

            cv.resize(cvRgbImageMat, cvRgbImageMat, new cv.Size(newWidth, newHeight), 0, 0, cv.INTER_AREA);

        }
    }
    return [cvRgbImageMat, transImageMat, xRatio, yRatio];
}


/**
 * Reads the image file and return ImageElement.
 * @param file The file Url of the image.
 * @returns {Promise<HTMLImageElement>} The resulting ImageElement.
 */
function readImageFile(file) {
    return new Promise((resolve, reject) => {
        if (file.type && !file.type.startsWith('image/')) {
            console.log('File is not an image.', file.type, file);
            reject('File is not an image.');
            return;
        }
        const img = new Image();
        const reader = new FileReader();
        reader.addEventListener('load', (event) => {
            img.src = event.target.result;
        });
        reader.addEventListener('error', (event) => {
            console.log('Error reading file:', event);
            reject('Error reading file.');
        });
        reader.readAsDataURL(file);
        img.onload = () => resolve(img);
    });
}

/**
 * Preprocessing image to Model Shape
 * @param {cv.Mat} matC3 The RGB image matrix with CV_8UC3 Data type.
 * @param {Number} modelWidth The model input width.
 * @param {Number} modelHeight The model input height.
 * @param {Number} stride The value by which Image dims will be divisible.
 * @return {cv.Mat, Number, Number} The resulting image matrix, xRatio of the Input, yRatio of the Input.
 */
const preprocessing = (matC3, modelWidth, modelHeight, stride = 32) => {

    const [w, h] = divStride(stride, matC3.cols, matC3.rows);
    cv.resize(matC3, matC3, new cv.Size(w, h), 0, 0, cv.INTER_LANCZOS4);

    // padding image to [n x n] dim
    const maxSize = Math.max(matC3.rows, matC3.cols); // get max size from width and height
    const xPad = maxSize - matC3.cols;
    const xRatio = maxSize / matC3.cols;
    const yPad = maxSize - matC3.rows;
    const yRatio = maxSize / matC3.rows;
    const matPad = new cv.Mat(); // new mat for padded image
    cv.copyMakeBorder(matC3, matPad, 0, yPad, 0, xPad, cv.BORDER_CONSTANT); // padding black

    const input = cv.blobFromImage(
        matPad,
        1 / 255.0, // normalize
        new cv.Size(modelWidth, modelHeight), // resize to model input size
        new cv.Scalar(0, 0, 0),
        false, // swapRB
        false // crop
    ); // preprocessing image matrix

    // release mat opencv
    matPad.delete();

    return [input, xRatio, yRatio];
};

/**
 * Get divisible image size by stride
 * @param {Number} stride
 * @param {Number} width
 * @param {Number} height
 * @returns {Number[2]} image size [w, h]
 * @author @Hyuto Wahyu Setianto (yolov8-seg-onnxruntime-web)
 */
const divStride = (stride, width, height) => {
    if (width % stride !== 0) {
        if (width % stride >= stride / 2) width = (Math.floor(width / stride) + 1) * stride;
        else width = Math.floor(width / stride) * stride;
    }
    if (height % stride !== 0) {
        if (height % stride >= stride / 2) height = (Math.floor(height / stride) + 1) * stride;
        else height = Math.floor(height / stride) * stride;
    }
    return [width, height];
};
