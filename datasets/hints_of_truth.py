import os
import csv
import torch
import ollama
from datasets import load_dataset
from diffusers import FluxPipeline, FluxTransformer2DModel, GGUFQuantizationConfig
from PIL import Image
from tqdm import tqdm

# --- Configuration ---
OUTPUT_DIR = "generated_fake_images"
CSV_OUTPUT_NAME = "generated_images_{split}.csv"
IMAGE_MODEL_ID = "https: // huggingface.co/city96/FLUX.1-schnell-gguf/blob/main/flux1-schnell-Q8_0.gguf"
OLLAMA_MODEL = "llava:7b"  # Added Llava model configuration

os.makedirs(OUTPUT_DIR, exist_ok=True)


def initialize_models():
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
    pipe.enable_model_cpu_offload()

    return pipe


def get_llava_caption(image_path, original_text):
    """Uses Llava to generate a realistic caption based on the image and original text."""
    try:
        response = ollama.chat(
            model=OLLAMA_MODEL,
            messages=[
                {
                    'role': 'user',
                    'content': f"Here is the original text that inspired this image: '{original_text}'. Please provide a realistic, concise, and descriptive caption for what is actually shown in the image.",
                    # Ollama Python client accepts file paths directly
                    'images': [image_path]
                }
            ]
        )
        return response['message']['content'].strip()
    except Exception as e:
        print(f"\nError calling Ollama Llava: {e}")
        return "FAILED_LLAVA_ERROR"


def main():
    pipe = initialize_models()

    print("\nLoading dataset 'michiel/hints_of_truth'...")
    full_dataset = load_dataset("michiel/hints_of_truth")

    for split in ['dev1', 'dev2', 'test']:
        csv_file_path = os.path.join(
            OUTPUT_DIR, CSV_OUTPUT_NAME.format(split=split))

        csv_file = open(csv_file_path, mode='w', newline='', encoding='utf-8')
        csv_writer = csv.writer(csv_file)
        # Added llava_caption to the header
        csv_writer.writerow(
            ['original_index', 'original_text', 'saved_image_path'])

        print(f"\nStarting generation of {split} split")
        dataset = full_dataset[split]
        print(dataset)
        print(len(dataset))

        # Fixed loop to use len(dataset) instead of len(full_dataset)
        for i in tqdm(range(len(dataset)), desc=f"Generating {split} Data"):

            original_text = dataset[i]['text']

            if not original_text:
                print(f"Row {i} has no text, skipping...")
                csv_writer.writerow([i, '', 'FAILED_NO_TEXT'])
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
                        max_sequence_length=256
                    ).images[0]

                # Fixed filename to include the split so they don't overwrite each other
                image_filename = f"{split}_img_{i:04d}.png"
                image_relative_path = os.path.join(OUTPUT_DIR, image_filename)

                # Save the image to disk FIRST so Llava can read the file
                image_result.save(image_relative_path)

                # 2. Generate Caption using Llava
                # llava_caption = get_llava_caption(
                #     image_relative_path, original_text)

                # 3. Write all data to CSV
                csv_writer.writerow(
                    [i, original_text, image_filename])

                # Delete the variable so Python knows it is no longer needed
                del image_result
                # Force PyTorch to immediately sweep GPU VRAM
                torch.cuda.empty_cache()
                # --- NEW: Tell PyTorch to defragment the RAM if possible ---
                if hasattr(torch.classes, 'cpu'):
                    torch.cpu.empty_cache()

            except Exception as e:
                print(f"\nError processing index {i}: {e}")
                csv_writer.writerow(
                    [i, original_text, 'FAILED_GPU_ERROR'])
                continue

        csv_file.close()


if __name__ == "__main__":
    main()
