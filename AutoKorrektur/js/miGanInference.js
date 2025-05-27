export {inferMiGan}

/**
 * Inferences MiGan Model with input image and Mask and returns the resulting image.
 * @param imageMat - The RGB image matrix with CV_8UC3 data type.
 * @param maskMat - The mask image matrix with CV_8UC1 data type.
 * @param model -  The MiGan Ort-Session.
 * @returns {Promise<cv.Mat>} - The resulting RGB image matrix with CV_8UC3 data type.
 */
async function inferMiGan(imageMat,maskMat,model){
  console.log("Inferencing MiGan")
  const imageWidth = imageMat.cols
  const imageHeight = imageMat.rows
  const imageArrCHW = orderInCHW(imageMat)
  const maskArrCHW = orderInCHW(maskMat)

  const imageTensor = new ort.Tensor('uint8', imageArrCHW, [1, 3, imageHeight, imageWidth])
  const maskTensor = new ort.Tensor('uint8', maskArrCHW, [1, 1, imageHeight, imageWidth])

  const output = await model.run({image: imageTensor, mask: maskTensor})

  const outputImageTensor = output[model.outputNames[0]]
  const outputImageHWC = reorderToHWC(outputImageTensor.data, imageWidth, imageHeight)
  const data = new Uint8ClampedArray(outputImageHWC)
  return cv.matFromArray(imageHeight, imageWidth, cv.CV_8UC3, data)



}

/**
 * @param {cv.Mat} img - The RGB image matrix with CV_8UC3 data type.
 * @returns {Uint8Array} - The RGB image array with CHW order.
 * @author lxfater (InpaintWeb)
 */
function orderInCHW(img ) {
  const channels = new cv.MatVector()
  cv.split(img, channels)

  const C = channels.size()
  const H = img.rows
  const W = img.cols

  const chwArray = new Uint8Array(C * H * W)

  for (let c = 0; c < C; c++) {
    const channelData = channels.get(c).data
    for (let h = 0; h < H; h++) {
      for (let w = 0; w < W; w++) {
        chwArray[c * H * W + h * W + w] = channelData[h * W + w]
      }
    }
  }

  channels.delete()
  return chwArray
}

/**
 * Reorders CHW image data into HWC image data and clamps the pixel values to 0-255.
 * @param {Uint8Array} uint8Data - The RGB image array with CHW order.
 * @param {number} width - The width of the image represented by the array.
 * @param {number} height - The height of the image represented by the array.
 * @returns {*[]} - The RGB image array with HWC order.
 * @author lxfater (InpaintWeb)
 */
function reorderToHWC(uint8Data, width, height) {
  const chwToHwcData = []
  const size = width * height

  for (let h = 0; h < height; h++) {
    for (let w = 0; w < width; w++) {
      for (let c = 0; c < 3; c++) {

        const chwIndex = c * size + h * width + w
        const pixelVal = uint8Data[chwIndex]
        let newPiex = pixelVal
        if (pixelVal > 255) {
          newPiex = 255
        } else if (pixelVal < 0) {
          newPiex = 0
        }
        chwToHwcData.push(newPiex)
      }
    }
  }
  return chwToHwcData
}
