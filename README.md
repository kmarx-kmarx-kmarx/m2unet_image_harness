# m2unet_image_harness

Run a pretrained M2UNet cell segmentation model on input images of any size. This code handles tiling and overlaps to prevent segmentation artifacts from appearing in the final image.

### Installation

After cloning the repo, run `pip install -r requirements.txt` to install the requirements.

Change the constants in `demo_run_on_images.py` to set the path to your image files, batch size, number of images, etc. Then run `python3 demo_run_on_images.py`
