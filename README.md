# AutoKorrektur

We re-implemented the [AutoKorrektur Web Version](https://github.com/BenB2/AutoKorrektur) in
Android.
This web version was based on a Bachelor Thesis "Autokorrektur - Automatisierte Objektersetzung in
Fotos" by Till Schellscheidt.

<table>
  <tr>
    <td><img src="media/image_1_with_car_640x640.png" alt="Example before processing" width="400"/></td>
    <td><img src="media/image_1_without_car_640x640.png" alt="Example after processing" width="400"/></td>
  </tr>
</table>

This application is intended to remove cars from pictures to make it easier to imagine a world, in
which the most dangerous animal in cities (cars) are less prevalent.
All processing is done on device.
Usage is free.

## Known Bugs

- Pictures must be taken in landscape mode (we plan to fix this)

## Tech

* ONNX
* OpenCV
* Instance Segmentation: **YOLOv11-seg**
* Inpainting: **MI-GAN**

## Licenses

The licensing of this project is governed by the licenses of some components.

* **YOLOv11-seg:** Licensed under GNU AGPLv3. You must comply with its terms, which may require this
  entire project to be licensed similarly.

Therefore this Project is licensed under the GNU AGPLv3 License. 

