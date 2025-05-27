export {inferYolo}

const labels = [
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "airplane",
    "bus",
    "train",
    "truck",
    "boat",
    "traffic light",
    "fire hydrant",
    "stop sign",
    "parking meter",
    "bench",
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
    "backpack",
    "umbrella",
    "handbag",
    "tie",
    "suitcase",
    "frisbee",
    "skis",
    "snowboard",
    "sports ball",
    "kite",
    "baseball bat",
    "baseball glove",
    "skateboard",
    "surfboard",
    "tennis racket",
    "bottle",
    "wine glass",
    "cup",
    "fork",
    "knife",
    "spoon",
    "bowl",
    "banana",
    "apple",
    "sandwich",
    "orange",
    "broccoli",
    "carrot",
    "hot dog",
    "pizza",
    "donut",
    "cake",
    "chair",
    "couch",
    "potted plant",
    "bed",
    "dining table",
    "toilet",
    "tv",
    "laptop",
    "mouse",
    "remote",
    "keyboard",
    "cell phone",
    "microwave",
    "oven",
    "toaster",
    "sink",
    "refrigerator",
    "book",
    "clock",
    "vase",
    "scissors",
    "teddy bear",
    "hair drier",
    "toothbrush",
];

const topAmountPerClass = 100; // top amount of Instances per class
const intersectionOverUnionThreshold = 0.9; // iou threshold
const baseScoreThreshold = 0.2; // score threshold
const numClass = labels.length;

/**
 * Performs YOLO inference for object detection and segmentation.
 * @param {cv.Mat} transImageMat - The transformed image matrix CV_8UC3 data type, resized for the segmentation model.
 * @param {number} xRatio - The aspect ratio of the input's width to the largest side of the input.
 * @param {number} yRatio - The aspect ratio of the input's height to the largest side of the input.
 * @param {object} session - The object containing the inference sessions for YOLO, NMS, and Mask.
 * @param {number} modelWidth - The width of the model for which the image was prepared.
 * @param {number} modelHeight - The height of the model for which the image was prepared.
 * @param {number} [upscaleFactor=1.0] - Factor by which the segmentation mask is upscaled.
 * @param {number} [scoreThreshold=0.2] - Confidence threshold for detections.
 * @param {number[]} [searchinglabels=[2,3,7]] - An array of label indices to search for.
 * @returns {Promise<cv.Mat>} A Promise that resolves to an image matrix with CV_8UC1 data type, representing the resulting mask.
 */
async function inferYolo(transImageMat, xRatio, yRatio, session, modelWidth, modelHeight, upscaleFactor = 1.0, scoreThreshold = baseScoreThreshold, searchinglabels = [2, 3, 7]) {
    const imageTensor = new ort.Tensor("float32", transImageMat.data32F, [1, 3, 640, 640]); // to ort.Tensor
    const maxSize = Math.max(modelWidth, modelHeight);

    /*
    run YOLO Model and get output Arrays.
    out0: segmentation Data [1,116,8400] (8400 possible detections with 116 values [4: boundingBoxes, 80: classProb, 32: maskCoefficients]),
    out1: mask Prototypes [1,32,160,160] (32 Prototypes with 160x160 size)
    */
    const {output0, output1} = await session.yolo.run({images: imageTensor});

    const nmsConfigTensor = new ort.Tensor(
        "float32",
        new Float32Array([
            80, // num class
            topAmountPerClass, // top amount of Instances per class
            intersectionOverUnionThreshold, // iou threshold
            scoreThreshold, // score threshold
        ])
    ); // nms config tensor

    // perform nms and filter boxes : dims [1,topAmPC,116] (116 = 4 (box) + 80 (classes) + 32 (mask coefficients))
    const {selected} = await session.nms.run(
        {detection: output0, config: nmsConfigTensor});

    // create image Mat in greyscale to overlay masks
    const overlay_gray = cv.Mat.ones(modelHeight, modelWidth, cv.CV_8UC1);
    overlay_gray.setTo(new cv.Scalar(255)); // set know values to white


    // looping over detected objects
    for (let idx = 0; idx < selected.dims[1]; idx++) {
        const data = selected.data.slice(idx * selected.dims[2], (idx + 1) * selected.dims[2]); // get rows
        let box = data.slice(0, 4); // get boundingBoxes
        const scores = data.slice(4, 4 + numClass); // get probability scores for each class
        const score = Math.max(...scores); // get highest probability scores
        const label = scores.indexOf(score); // get class id of highest probability scores

        // if detected Object is not in searching labels skip it
        if (!searchinglabels.includes(label)) { continue }

        box = overflowBoxes(
            [
                box[0] - 0.5 * box[2], // before upscale x
                box[1] - 0.5 * box[3], // before upscale y
                box[2], // before upscale w
                box[3], // before upscale h
            ],
            maxSize
        ); // keep boxes in maxSize range

        const [x, y, w, h] = overflowBoxes(
            [
                Math.floor(box[0] * xRatio), // upscale left
                Math.floor(box[1] * yRatio), // upscale top
                Math.floor(box[2] * xRatio), // upscale width
                Math.floor(box[3] * yRatio), // upscale height
            ],
            maxSize
        ); // upscale boxes


        const mask = new ort.Tensor(
            "float32",
            new Float32Array([
                ...box, // original scale box
                ...data.slice(4 + numClass), // mask data
            ])
        ); // mask input

        // Reposition the mask to fit with upscale
        const newX = x - ((w * upscaleFactor) - w) / 2;
        const newY = y - ((h * upscaleFactor) - h) / 2;

        const maskConfig = new ort.Tensor(
            "float32",
            new Float32Array([
                maxSize,
                newX, // upscale x
                newY, // upscale y
                w * upscaleFactor, // upscale width
                h * upscaleFactor, // upscale height
                2, 2, 2, 255, // fixed Color for Mask Model
            ])
        );// Configuration for Mask Model


        const {mask_filter} = await session.mask.run({
            detection: mask,
            mask: output1,
            config: maskConfig,
        }); // run mask calculation


        const mask_mat = cv.matFromArray(
            mask_filter.dims[0],
            mask_filter.dims[1],
            cv.CV_8UC4,
            mask_filter.data
        ); // mask result to Mat

        // Convert RGBA to grayscale
        cv.cvtColor(mask_mat, mask_mat, cv.COLOR_BGRA2GRAY);

        // Threshold to ensure binary values (0 or 255)
        cv.threshold(mask_mat, mask_mat, 1, 255, cv.THRESH_BINARY);


        // substarct mask from overlay so masked area will be black
        cv.subtract(overlay_gray, mask_mat, overlay_gray);

        mask_mat.delete();
    }
    return overlay_gray;

}


/**
 * Handle overflow boxes based on maxSize
 * @param {Number[4]} box box in [x, y, w, h] format
 * @param {Number} maxSize
 * @returns non overflow boxes
 * @author @Hyuto Wahyu Setianto (yolov8-seg-onnxruntime-web)
 */
const overflowBoxes = (box, maxSize) => {
    box[0] = box[0] >= 0 ? box[0] : 0;
    box[1] = box[1] >= 0 ? box[1] : 0;
    box[2] = box[0] + box[2] <= maxSize ? box[2] : maxSize - box[0];
    box[3] = box[1] + box[3] <= maxSize ? box[3] : maxSize - box[1];
    return box;
};



