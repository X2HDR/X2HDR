## Text-to-HDR Test Set
The test prompts are located in [`text2hdr.txt`](text2hdr.txt).


## RAW-to-HDR Test Set

To download the test set used in our paper, please visit this Google Drive [link](https://drive.google.com/file/d/1F-jbA4OhzU2TVUXEOqihdsBFuLBS5q-1/view?usp=sharing). The dataset includes:
- Input RAW images
- Ground truth HDR images
- LDR images (converted from RAW to sRGB space)

This test set is derived from the [SI-HDR](https://www.cl.cam.ac.uk/research/rainbow/projects/sihdr_benchmark/) benchmark.

As the reference HDR images are provided, merging RAW images to obtain the ground truth is unnecessary.

To ensure proper pixel alignment between reference HDR and RAW images when creating (RAW, HDR) pairs for testing, use the following preprocessing code:

```python
import numpy as np
import cv2
import HDRutils

ref = HDRutils.imread('reference/015.exr').astype(np.float32)
raw = HDRutils.imread('raw/015/_07A5960.CR2').astype(np.float32)

# Crop and resize to align with reference
raw = raw[:-6, 48:-48, :]
raw = cv2.resize(raw, (ref.shape[1], ref.shape[0]), interpolation=cv2.INTER_LINEAR)
```

For RAW-to-HDR reconstruction using LEDiff and Bracket Diffusion, use the following preprocessing code:
```python
def lin2srgb(L):
    t = 0.0031308
    a = 0.055
    L = L.clip(0, 1)
    p = np.where(L <= t, L * 12.92, (1 + a) * (L) ** (1 / 2.4) - a)
    return p
```
