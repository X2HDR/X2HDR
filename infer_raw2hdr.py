import torch
import argparse
import os
import logging
import cv2
import numpy as np
import HDRutils
from src.pipeline import FluxPipeline
from src.transformer_flux import FluxTransformer2DModel
from src.lora_helper import set_single_lora
from utils.util_hdr import save_hdr_image, read_hdr_image, scale_to_L_peak

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def clear_cache(transformer):
    """Clear the attention processor cache."""
    for name, attn_processor in transformer.attn_processors.items():
        attn_processor.bank_kv.clear()

def preprocess_raw_to_exr(raw_image_path: str, width: int, height: int, output_dir: str):
    """
    Preprocess RAW file to EXR format with resizing.
    
    Args:
        raw_image_path: Path to the RAW file (.CR2, .dng, etc.)
        width: Target width for resizing
        height: Target height for resizing
        output_dir: Directory to save processed EXR file
        
    Returns:
        Path to the processed EXR file
    """
    logger.info(f"Preprocessing RAW file: {raw_image_path}")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Read RAW file using HDRutils
    raw_image = HDRutils.imread(raw_image_path)
    logger.info(f"Loaded RAW image with shape: {raw_image.shape}")
    
    # Ensure image is float32 format and positive values
    if raw_image.dtype != np.float32:
        raw_image = raw_image.astype(np.float32)
    raw_image = np.maximum(raw_image, 0.0)
    
    # Resize to target dimensions
    resized_image = cv2.resize(raw_image, (width, height), interpolation=cv2.INTER_LINEAR)
    logger.info(f"Resized image to: {resized_image.shape}")
    
    # Save as EXR file in output directory
    base_name = os.path.splitext(os.path.basename(raw_image_path))[0]
    exr_path = os.path.join(output_dir, f"{base_name}_processed.exr")
    save_hdr_image(resized_image, exr_path)
    logger.info(f"Saved processed EXR file to: {exr_path}")
    
    return exr_path

def setup_pipeline(model_id: str, lora_path: str, width: int, height: int, device: str = "cuda"):
    """Initialize and setup the FLUX pipeline with LoRA weights."""
    logger.info("Loading pipeline...")
    pipe = FluxPipeline.from_pretrained(model_id, torch_dtype=torch.bfloat16, device=device)
    transformer = FluxTransformer2DModel.from_pretrained(
        model_id, 
        subfolder="transformer",
        torch_dtype=torch.bfloat16, 
        device=device
    )
    pipe.transformer = transformer
    pipe.to(device)

    # Load LoRA model
    set_single_lora(pipe.transformer, lora_path, lora_weights=[1], cond_width=width, cond_height=height)
    logger.info("Pipeline loaded successfully!")
    
    return pipe


def process_raw_to_hdr(
    pipe: FluxPipeline,
    raw_image_path: str,
    prompt: str,
    width: int,
    height: int,
    guidance_scale: float,
    num_inference_steps: int,
    max_sequence_length: int,
    seed: int,
    output_dir: str,
    input_is_raw: bool = True
):
    """Process a single raw HDR image to generate output HDR."""
    logger.info(f"Processing raw image: {raw_image_path}")
    logger.info(f"Prompt: {prompt}")
    logger.info(f"Parameters: size={width}x{height}, guidance={guidance_scale}, steps={num_inference_steps}, seed={seed}")
    
    # Check if source file exists
    if not os.path.exists(raw_image_path):
        logger.error(f"Raw image file not found: {raw_image_path}")
        return
    
    # Check if input is a RAW file and preprocess if needed
    _, ext = os.path.splitext(raw_image_path)
    raw_extensions = ['.cr2', '.CR2', '.dng', '.DNG', '.nef', '.NEF', '.arw', '.ARW', '.raw', '.RAW']
    
    if ext in raw_extensions:
        logger.info("Detected RAW file format, preprocessing...")
        # Preprocess RAW file to EXR
        processed_exr_path = preprocess_raw_to_exr(raw_image_path, width, height, output_dir)
        # Load the processed EXR image
        spatial_image = read_hdr_image(processed_exr_path)
    else:
        # Load the HDR image directly (assuming it's already EXR format)
        logger.info("Loading EXR file directly...")
        spatial_image = read_hdr_image(raw_image_path)
    
    # Run pipeline to generate HDR image
    logger.info("Running inference...")
    image_output, hdr_image = pipe(
        prompt,
        width=width,
        height=height,
        guidance_scale=guidance_scale,
        num_inference_steps=num_inference_steps,
        max_sequence_length=max_sequence_length,
        generator=torch.Generator("cpu").manual_seed(seed),
        spatial_images=[spatial_image],
        subject_images=[],
        cond_width=width,
        cond_height=height,
        hdr_mode=True,
        input_is_raw=input_is_raw,
    )

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Save output LDR image
    output_ldr_image = image_output.images[0]
    output_ldr_path = os.path.join(output_dir, 'output_ldr.png')
    output_ldr_image.save(output_ldr_path)
    logger.info(f"Saved output LDR to: {output_ldr_path}")
    
    # Save output HDR image
    output_hdr_path = os.path.join(output_dir, 'output_hdr.exr')
    hdr_image_scaled, _ = scale_to_L_peak(hdr_image, L_peak=32.0)
    save_hdr_image(hdr_image_scaled, output_hdr_path)
    logger.info(f"Saved output HDR to: {output_hdr_path}")
    
    # Clear cache
    clear_cache(pipe.transformer)
    logger.info("Processing completed successfully!")


def main():
    parser = argparse.ArgumentParser(description="FLUX Raw to HDR conversion with LoRA")
    parser.add_argument("--model_id", type=str, default="models/Flux", help="Path to the FLUX model")
    parser.add_argument("--lora_path", type=str, default="models/raw2hdr_lora.safetensors", help="Path to LoRA weights")
    parser.add_argument("--raw_image", type=str, required=True, help="Path to the input raw HDR image (.CR2, .dng, etc.) or EXR file")
    parser.add_argument("--prompt", type=str, default=" ", help="Prompt for generation")
    parser.add_argument("--output_dir", type=str, default="output", help="Output directory")
    parser.add_argument("--width", type=int, default=512, help="Image width")
    parser.add_argument("--height", type=int, default=512, help="Image height")
    parser.add_argument("--guidance_scale", type=float, default=3.5, help="Guidance scale")
    parser.add_argument("--num_inference_steps", type=int, default=30, help="Number of inference steps")
    parser.add_argument("--max_sequence_length", type=int, default=512, help="Maximum sequence length")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--input_is_raw", action="store_true", default=True, help="Whether input is HDR format")
    
    args = parser.parse_args()
    
    # Setup pipeline
    pipe = setup_pipeline(args.model_id, args.lora_path, args.width, args.height)
    
    # Process the single raw image
    process_raw_to_hdr(
        pipe,
        args.raw_image,
        args.prompt,
        args.width,
        args.height,
        args.guidance_scale,
        args.num_inference_steps,
        args.max_sequence_length,
        args.seed,
        args.output_dir,
        args.input_is_raw
    )
    
    logger.info("Done!")


if __name__ == "__main__":
    main()
