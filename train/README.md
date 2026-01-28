# Training

### Frameworks
We use the following frameworks for training:
- **Text-to-HDR training**: [sd-scripts](https://github.com/kohya-ss/sd-scripts/tree/sd3) (`sd3` branch)
- **RAW-to-HDR training**: [EasyControl](https://github.com/Xiaojiu-z/EasyControl)

And please find them in the `text2hdr` and `raw2hdr` directories respectively.

### HDR Image Loading for Training
To train with HDR images, the key is to modify the dataset loading code to load HDR images and convert them to PU21 color space in the [0, 1] range. The following code demonstrates how to do this:
```python
from utils.util_hdr import read_hdr_image, scale_to_L_peak, pu21_encode_rgb

def load_hdr_image(image_path):
    hdr_image = read_hdr_image(image_path)
    hdr_image, _ = scale_to_L_peak(hdr_image, L_peak=4000.0)
    pu21_image = pu21_encode_rgb(hdr_image)  # Output range: [0, 1]
    return pu21_image
```

The rest of the training process remains the same as standard diffusion model training.

**Important reminder**: Most generative models expect inputs in the [-1, 1] range, so you'll need to normalize the PU21 images (which are in [0, 1]) accordingly before feeding them to the model. Apply the transformation: `normalized_image = pu21_image * 2.0 - 1.0`.

For the corresponding decoding process (converting from PU21 back to HDR), refer to the implementation in `infer_text2hdr.py`, [`process_hdr()`](../infer_text2hdr.py#L73) function.