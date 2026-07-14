Pill Crop CLI
=============

This standalone bundle detects every pill in one input image and writes a
separate tightly cropped image for each detected region. It uses the bundled
TorchScript model directly and does not require Ultralytics.

Requirements
------------

- Windows
- Python 3.10 or newer
- Internet access during the initial dependency installation

Setup
-----

1. Extract the entire ZIP file.
2. Open Command Prompt in the extracted pill-crop-cli directory.
3. Run:

   setup.cmd

Usage
-----

   run.cmd --input input.jpg --output output\pill.png

If one pill is detected, the requested output name is used. If multiple pills
are detected, files are named pill_1.png, pill_2.png, and so forth in descending
confidence order.

Options
-------

   run.cmd --help
   run.cmd -i input.jpg -o output\pill.png --confidence 0.40
   run.cmd -i input.jpg -o output\pill.png --device cuda:0

The default model is models\crop_best.torchscript. Use --model to select a
different compatible TorchScript model.

Exit codes
----------

0 = crops saved successfully
1 = input, model, inference, dependency, or output error
2 = no pill detected
