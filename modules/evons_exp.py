import os
import pandas as pd
import torch
import gc
from diffusers import FluxPipeline, FluxTransformer2DModel, GGUFQuantizationConfig, StableDiffusion3Pipeline, SD3Transformer2DModel, GGUFQuantizationConfig
from transformers import AutoTokenizer, AutoModelForCausalLM
from PIL import Image
from tqdm import tqdm

# --- Configuration ---
MODEL_NAME = "sd"
OUTPUT_DIR = f"evons_data"
IMAGE_OUTPUT_DIR = f"evons_qwen_{MODEL_NAME}"
CSV_OUTPUT_NAME = "evons_exp.csv"
IMAGE_MODEL_ID = "https://huggingface.co/city96/stable-diffusion-3.5-large-turbo-gguf/blob/main/sd3.5_large_turbo-Q8_0.gguf"
OLLAMA_MODEL = "llava:7b"  # Added Llava model configuration

# Local Qwen model for caption rewriting
QWEN_MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"

os.makedirs(f"{OUTPUT_DIR}/{IMAGE_OUTPUT_DIR}", exist_ok=True)


def initialize_qwen_model():
    """Initializes a local Qwen model for caption rewriting."""
    print("Loading Qwen model locally...")

    tokenizer = AutoTokenizer.from_pretrained(QWEN_MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        QWEN_MODEL_ID,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    return tokenizer, model


def rewrite_caption(tokenizer, model, original_text):
    """Rewrites a caption to be more vivid and image-generation friendly."""
    if not original_text or not original_text.strip():
        return original_text

    messages = [
        {
            "role": "system",
            "content": (
                "Rewrite the caption. Preserve the original meaning, key entities, and scene. Do not add facts not implied by the original text. Return only the rewritten caption."
            ),
        },
        {
            "role": "user",
            "content": f"Original caption: {original_text}",
        },
    ]

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=120,
            temperature=0.4,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )

    generated_ids = outputs[0][inputs["input_ids"].shape[1]:]
    rewritten = tokenizer.decode(
        generated_ids, skip_special_tokens=True
    ).strip()

    return rewritten


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

    transformer = FluxTransformer2DModel.from_single_file(
        IMAGE_MODEL_ID,
        quantization_config=GGUFQuantizationConfig(
            compute_dtype=torch.bfloat16
        ),
        torch_dtype=torch.bfloat16,
    )

    print("Loading the rest of the FLUX pipeline...")

    pipe = FluxPipeline.from_pretrained(
        "black-forest-labs/FLUX.1-schnell",
        transformer=transformer,
        torch_dtype=torch.bfloat16,
    )

    pipe.to("cuda")
    return pipe


def main():
    tokenizer, qwen_model = initialize_qwen_model()
    if MODEL_NAME == "sd":
        pipe = initialize_sd_models()
    else:
        pipe = initialize_flux_models()

    print("\nLoading dataset 'michiel/hints_of_truth'...")
    input_csv_path = os.path.join(OUTPUT_DIR, f"evons.csv")
    csv_file_path = os.path.join(
        OUTPUT_DIR, CSV_OUTPUT_NAME
    )

    df = pd.read_csv(input_csv_path, encoding='utf-8', low_memory=False)

    # Filter rows: keep only real samples with valid images
    df = df[(
        df['is_valid_image'] == 1)].reset_index(drop=True)

    saved_image_paths, titles, descriptions = [], [], []

    print(f"\nStarting generation of evons")

    for i in tqdm(range(len(df)), desc=f"Generating Data"):
        row = df.iloc[i]

        title = str(row['title']) if pd.notna(row['title']) else ''
        description = str(row['description']) if pd.notna(
            row['description']) else ''

        # 1. Rewrite title with local Qwen
        rewritten_title = rewrite_caption(
            tokenizer, qwen_model, title)

        # 2. Rewrite description with local Qwen
        rewritten_description = rewrite_caption(
            tokenizer, qwen_model, description)

        original_text = f"{title}. {description}".strip()
        if original_text.startswith(". "):
            original_text = original_text[2:]

        try:
            # 1. Generate Image using FLUX.1-schnell
            with torch.inference_mode():
                image_result = pipe(
                    prompt=original_text,
                    num_inference_steps=4,
                    guidance_scale=0.0,
                    height=512,
                    width=512,
                    max_sequence_length=256
                ).images[0]

            # Fixed filename to include the split so they don't overwrite each other
            image_filename = f"fake_img_{i:04d}.png"
            image_relative_path = os.path.join(
                OUTPUT_DIR, IMAGE_OUTPUT_DIR, image_filename)

            # Save the image to disk FIRST so later steps can read the file
            image_result.save(image_relative_path)

            # 3. Write all data to CSV
            titles.append(
                rewritten_title
            )
            descriptions.append(
                rewritten_description
            )
            saved_image_paths.append(
                image_filename
            )

            # Free memory
            del image_result

            # Do cleanup occasionally, not every image
            if i % 50 == 0:
                gc.collect()

        except Exception as e:
            print(f"\nError processing index {i}: {e}")
            saved_image_paths.append("")
            torch.cuda.empty_cache()
            gc.collect()
            continue

    df["qwen_rewritten_title"] = titles
    df["qwen_rewritten_description"] = descriptions
    df["fake_img_paths"] = saved_image_paths
    df.to_csv(csv_file_path, index=False)


if __name__ == "__main__":
    main()
