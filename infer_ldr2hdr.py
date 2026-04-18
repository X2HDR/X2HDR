import argparse
import logging
import os

import torch
from PIL import Image

from src.lora_helper import set_single_lora
from src.pipeline import FluxPipeline
from src.transformer_flux import FluxTransformer2DModel
from utils.util_hdr import get_luminance_percentile, save_hdr_image

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def clear_cache(transformer):
    """Clear the attention processor cache."""
    for name, attn_processor in transformer.attn_processors.items():
        attn_processor.bank_kv.clear()


def scale_hdr_to_peak(hdr_image, target_peak=16.0):
    """Scale HDR output based on a luminance percentile peak."""
    highest_luminance = get_luminance_percentile(hdr_image, percentile=99.5)
    if highest_luminance > 0:
        scale = target_peak / highest_luminance
        scale = min(scale, 1.0)
        return hdr_image * scale
    return hdr_image


def setup_pipeline(model_id: str, lora_path: str, width: int, height: int, device: str = "cuda"):
    """Initialize and setup the FLUX pipeline with LoRA weights."""
    logger.info("Loading pipeline...")
    pipe = FluxPipeline.from_pretrained(model_id, torch_dtype=torch.bfloat16, device=device)
    transformer = FluxTransformer2DModel.from_pretrained(
        model_id,
        subfolder="transformer",
        torch_dtype=torch.bfloat16,
        device=device,
    )
    pipe.transformer = transformer
    pipe.to(device)

    set_single_lora(pipe.transformer, lora_path, lora_weights=[1], cond_width=width, cond_height=height)
    logger.info("Pipeline loaded successfully!")

    return pipe


def process_ldr_to_hdr(
    pipe: FluxPipeline,
    ldr_image_path: str,
    prompt: str,
    width: int,
    height: int,
    guidance_scale: float,
    num_inference_steps: int,
    max_sequence_length: int,
    seed: int,
    output_dir: str,
):
    """Process a single LDR image to generate output HDR."""
    logger.info(f"Processing LDR image: {ldr_image_path}")
    logger.info(f"Prompt: {prompt}")
    logger.info(f"Parameters: size={width}x{height}, guidance={guidance_scale}, steps={num_inference_steps}, seed={seed}")

    if not os.path.exists(ldr_image_path):
        logger.error(f"LDR image file not found: {ldr_image_path}")
        return

    spatial_image = Image.open(ldr_image_path).convert("RGB")
    spatial_image = spatial_image.resize((width, height))

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
        input_is_raw=False,
    )

    os.makedirs(output_dir, exist_ok=True)

    output_ldr_image = image_output.images[0]
    output_ldr_path = os.path.join(output_dir, "output_ldr.png")
    output_ldr_image.save(output_ldr_path)
    logger.info(f"Saved output LDR to: {output_ldr_path}")

    output_hdr_path = os.path.join(output_dir, "output_hdr.exr")
    hdr_image_scaled = scale_hdr_to_peak(hdr_image, target_peak=16.0)
    save_hdr_image(hdr_image_scaled, output_hdr_path)
    logger.info(f"Saved output HDR to: {output_hdr_path}")

    clear_cache(pipe.transformer)
    logger.info("Processing completed successfully!")


def main():
    parser = argparse.ArgumentParser(description="FLUX LDR to HDR conversion with LoRA")
    parser.add_argument("--model_id", type=str, default="models/Flux", help="Path to the FLUX model")
    parser.add_argument("--lora_path", type=str, default="models/ldr2hdr_lora.safetensors", help="Path to LoRA weights")
    parser.add_argument("--ldr_image", type=str, required=True, help="Path to the input LDR image")
    parser.add_argument("--prompt", type=str, default=" ", help="Prompt for generation")
    parser.add_argument("--output_dir", type=str, default="output", help="Output directory")
    parser.add_argument("--width", type=int, default=512, help="Image width")
    parser.add_argument("--height", type=int, default=512, help="Image height")
    parser.add_argument("--guidance_scale", type=float, default=3.5, help="Guidance scale")
    parser.add_argument("--num_inference_steps", type=int, default=30, help="Number of inference steps")
    parser.add_argument("--max_sequence_length", type=int, default=512, help="Maximum sequence length")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()

    pipe = setup_pipeline(args.model_id, args.lora_path, args.width, args.height)

    process_ldr_to_hdr(
        pipe,
        args.ldr_image,
        args.prompt,
        args.width,
        args.height,
        args.guidance_scale,
        args.num_inference_steps,
        args.max_sequence_length,
        args.seed,
        args.output_dir,
    )

    logger.info("Done!")


if __name__ == "__main__":
    main()