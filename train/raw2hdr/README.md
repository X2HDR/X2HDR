# RAW-to-HDR Training

## Framework

This training framework is built upon the [`EasyControl`](https://github.com/Xiaojiu-z/EasyControl) repository.

## Setup

Activate the conda environment created in the root directory:

```shell
conda activate x2hdr
```

## Model Preparation

This training pipeline uses the same FLUX.1 model as the inference pipeline.

Before starting, update the `MODEL_DIR` variable in the `train.sh` script to point to your local FLUX.1 model path.

## Training Dataset

### Data Sources
The datasets are sourced from the following repositories:
- **HDRPS**: [http://markfairchild.org/HDR.html](http://markfairchild.org/HDR.html)
- **RawHDR**: [https://github.com/yunhao-zou/RawHDR](https://github.com/yunhao-zou/RawHDR)
  - *Note: RawHDR images are hosted on BaiduDisk, which may have slow download speeds without a VIP account.*

### Data Processing
1.  **Merge & Align**: We merge RAW images into HDR ground truth using [HDRutils](https://github.com/gfxdisp/HDRUtils). This handles alignment and exposure estimation. You can use the provided `merge_raw.py` script for this step.
    - Sometimes, you may encounter errors when merging RAW images. In this case, you can set `do_align=False` and `estimate_exp=None` in the `merge_raw.py` script.
2.  **Crop & Format**: Random patches are cropped to create `(RAW, HDR)` pairs.
    - Since `HDRutils` processes RAW images and exports them as EXR files, both the input RAW and target HDR images for training must be in **EXR format**.

*For more details, please refer to the paper. Due to size constraints, the processed training dataset is not provided directly.*

### Metadata Format

Prepare a metadata `jsonl` file to index your dataset. Each line should be a JSON object containing the source path, target path, and optional captions:

```json
{"source": "raw_path_1.exr", "target": "target_path_1.exr", "caption": ["caption_1", "caption_2"]}
{"source": "raw_path_2.exr", "target": "target_path_2.exr", "caption": ["caption_1"]}
...
```

**Notes:**
- The `caption` field can be a single string or a list of strings.
- During training, captions are randomly dropped (replaced with an empty string `" "`) to support unconditional reconstruction.
- A sample metadata file is provided in the `examples` folder. Please replace it with your own generated file.

## Training

Run the training script using the following command:

```shell
bash train.sh
```

## Inference

To perform inference using the `diffusers` library, use the `infer_raw2hdr.py` script located in the root directory.
