import os
import csv
import torch
import gc
import pandas as pd
from datasets import load_dataset
from diffusers import (
    FluxPipeline,
    FluxTransformer2DModel,
    GGUFQuantizationConfig,
    StableDiffusion3Pipeline,
    SD3Transformer2DModel,
)
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm

# --- Configuration ---
MODEL_NAME = "flux"
OUTPUT_DIR = f"hints_of_truth_{MODEL_NAME}_exp"
CSV_OUTPUT_NAME = "generated_images_{split}.csv"
IMAGE_MODEL_ID = "https://huggingface.co/city96/stable-diffusion-3.5-large-turbo-gguf/blob/main/sd3.5_large_turbo-Q8_0.gguf"

# Local Qwen model for caption rewriting
QWEN_MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"

os.makedirs(OUTPUT_DIR, exist_ok=True)


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

    transformer = SD3Transformer2DModel.from_single_file(
        IMAGE_MODEL_ID,
        quantization_config=GGUFQuantizationConfig(
            compute_dtype=torch.bfloat16
        ),
        torch_dtype=torch.bfloat16,
    )

    print("Loading the rest of the SD3.5 pipeline...")

    pipe = StableDiffusion3Pipeline.from_pretrained(
        "stabilityai/stable-diffusion-3.5-large-turbo",
        transformer=transformer,
        torch_dtype=torch.bfloat16,
    )

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
    full_dataset = load_dataset("michiel/hints_of_truth")

    for split in ['dev1', 'dev2', 'test']:
        csv_file_path = os.path.join(
            OUTPUT_DIR, CSV_OUTPUT_NAME.format(split=split)
        )

        csv_file = pd.read_csv(csv_file_path)
        saved_flux_image_path = []
        # csv_file = open(csv_file_path, mode='w', newline='', encoding='utf-8')
        # csv_writer = csv.writer(csv_file)
        # csv_writer.writerow(
        #     ['original_index', 'original_text',
        #         'rewritten_text', 'saved_sd_image_path']
        # )

        print(f"\nStarting generation of {split} split")
        dataset = full_dataset[split]
        print(dataset)
        print(len(dataset))

        for i in tqdm(range(len(csv_file)), desc=f"Generating {split} Data"):
            original_text = csv_file.iloc[i]['rewritten_text']

            if not original_text:
                print(f"Row {i} has no text, skipping...")
                saved_flux_image_path.append('FAILED_NO_TEXT')
                # csv_writer.writerow([i, '', '', 'FAILED_NO_TEXT'])
                continue

            try:
                # 1. Rewrite caption with local Qwen
                # rewritten_text = rewrite_caption(
                #     tokenizer, qwen_model, original_text)
                rewritten_text = original_text

                # 2. Generate image using SD3.5 from rewritten caption
                with torch.inference_mode():
                    image_result = pipe(
                        prompt=rewritten_text,
                        num_inference_steps=4,
                        guidance_scale=0.0,
                        height=512,
                        width=512,
                    ).images[0]

                image_filename = f"{split}_img_{i:04d}.png"
                image_relative_path = os.path.join(OUTPUT_DIR, image_filename)

                image_result.save(image_relative_path)

                # 3. Write all data to CSV
                # csv_writer.writerow(
                #     [i, original_text, rewritten_text, image_filename]
                # )
                saved_flux_image_path.append(image_filename)

                del image_result
                gc.collect()
                torch.cuda.empty_cache()

            except Exception as e:
                print(f"\nError processing index {i}: {e}")
                # csv_writer.writerow(
                #     [i, original_text, '', 'FAILED_GPU_ERROR']
                # )
                saved_flux_image_path.append('FAILED_NO_TEXT')
                continue

        # csv_file.close()
        csv_file.to_csv(csv_file_path, index=False)


if __name__ == "__main__":
    main()
