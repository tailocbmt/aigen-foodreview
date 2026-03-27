import os
import pandas as pd
import torch
import gc
from diffusers import FluxPipeline, FluxTransformer2DModel, GGUFQuantizationConfig, StableDiffusion3Pipeline, SD3Transformer2DModel, GGUFQuantizationConfig
from PIL import Image
from tqdm import tqdm

# --- Configuration ---
OUTPUT_DIR = "generated_evons"
CSV_OUTPUT_NAME = "evons_generated_images.csv"
IMAGE_MODEL_ID = "https://huggingface.co/city96/stable-diffusion-3.5-large-turbo-gguf/blob/main/sd3.5_large_turbo-Q8_0.gguf"
OLLAMA_MODEL = "llava:7b"  # Added Llava model configuration

os.makedirs(OUTPUT_DIR, exist_ok=True)


def initialize_sd_models():
    """Initializes the SD3.5 Turbo pipeline using the Q8_0 GGUF model."""
    print("Loading the SD3.5 Large Turbo GGUF transformer...")

    # 1. Load the transformer using the SD3-specific single file loader
    transformer = SD3Transformer2DModel.from_single_file(
        IMAGE_MODEL_ID,
        quantization_config=GGUFQuantizationConfig(
            compute_dtype=torch.bfloat16),
        torch_dtype=torch.bfloat16,
    )

    print("Loading the rest of the SD3.5 pipeline...")

    # 2. Load the pipeline using the SD3 base model, passing in our custom GGUF transformer
    pipe = StableDiffusion3Pipeline.from_pretrained(
        "stabilityai/stable-diffusion-3.5-large-turbo",
        transformer=transformer,
        torch_dtype=torch.bfloat16
    )

    # Move directly to the RTX 3090's VRAM
    print("Moving model to GPU...")
    pipe.to("cuda")

    return pipe


def initialize_flux_models():
    """Initializes the FLUX pipeline using the Q8_0 GGUF model."""
    print("Loading the 12.7GB GGUF transformer...")

    # NOTE: You can use the Hugging Face URL directly, OR if you already downloaded
    # the 12.7GB file to your server, replace this URL with the local file path
    # (e.g., ckpt_path = "./flux1-schnell-Q8_0.gguf")

    # 1. Load the massive transformer using the new GGUF single-file loader
    transformer = FluxTransformer2DModel.from_single_file(
        IMAGE_MODEL_ID,
        quantization_config=GGUFQuantizationConfig(
            compute_dtype=torch.bfloat16),
        torch_dtype=torch.bfloat16,
    )

    print("Loading the rest of the FLUX pipeline...")

    # 2. Load the pipeline, passing in our custom GGUF transformer
    pipe = FluxPipeline.from_pretrained(
        "black-forest-labs/FLUX.1-schnell",
        transformer=transformer,
        torch_dtype=torch.bfloat16
    )

    # enable_model_cpu_offload() is highly recommended for FLUX on a 24GB RTX 3090
    pipe.to("cuda")

    return pipe


def main():
    pipe = initialize_flux_models()

    print("\nLoading dataset 'michiel/hints_of_truth'...")
    input_csv_path = os.path.join(OUTPUT_DIR, f"evons.csv")
    csv_file_path = os.path.join(
        OUTPUT_DIR, CSV_OUTPUT_NAME
    )

    df = pd.read_csv(input_csv_path, encoding='utf-8', low_memory=False)

    # Filter rows: keep only real samples with valid images
    df = df[(df['is_fake'] == 0) & (
        df['is_valid_image'] == 1)].reset_index(drop=True)

    saved_image_paths = []

    print(f"\nStarting generation of evons")

    for i in tqdm(range(len(df)), desc=f"Generating Data"):
        row = df.iloc[i]

        title = str(row['title']) if pd.notna(row['title']) else ''
        description = str(row['description']) if pd.notna(
            row['description']) else ''

        # Build text from title + description
        original_text = f"{title}. {description}".strip()
        if original_text.startswith(". "):
            original_text = original_text[2:]

        if not original_text:
            print(f"Row {i} has no text, skipping...")
            saved_image_paths.append("")
            continue

        try:
            # 1. Generate Image using FLUX.1-schnell
            with torch.inference_mode():
                image_result = pipe(
                    prompt=original_text,
                    num_inference_steps=4,
                    guidance_scale=0.0,
                    height=512,
                    width=512,
                    # max_sequence_length=256
                ).images[0]

            # Fixed filename to include the split so they don't overwrite each other
            image_filename = f"fake_img_{i:04d}.png"
            image_relative_path = os.path.join(OUTPUT_DIR, image_filename)

            # Save the image to disk FIRST so later steps can read the file
            image_result.save(image_relative_path)

            # 2. Write all data to CSV
            saved_image_paths.append(
                image_filename
            )

            # Free memory
            del image_result
            gc.collect()
            torch.cuda.empty_cache()

        except Exception as e:
            print(f"\nError processing index {i}: {e}")
            saved_image_paths.append("")
            continue

    df["fake_img_paths"] = saved_image_paths
    df.to_csv(csv_file_path, index=False)


if __name__ == "__main__":
    main()
