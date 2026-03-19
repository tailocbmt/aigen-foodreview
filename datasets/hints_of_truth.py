import os
import csv
import torch
import ollama
from datasets import load_dataset
from diffusers import FluxPipeline
from PIL import Image
from tqdm import tqdm

# --- Configuration ---
OUTPUT_DIR = "generated_fake_images"
CSV_OUTPUT_NAME = "generated_images_{split}.csv"
IMAGE_MODEL_ID = "black-forest-labs/FLUX.1-schnell"
OLLAMA_MODEL = "llava:7b"  # Added Llava model configuration

os.makedirs(OUTPUT_DIR, exist_ok=True)


def initialize_models():
    """Initializes the FLUX pipeline on GPU."""
    print(f"Loading FLUX model: {IMAGE_MODEL_ID}...")

    # FLUX requires bfloat16 for optimal memory usage and quality
    pipe = FluxPipeline.from_pretrained(
        IMAGE_MODEL_ID,
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
            ['original_index', 'original_text', 'saved_image_path', 'llava_caption'])

        print(f"\nStarting generation of {split} split")
        dataset = full_dataset[split]
        print(dataset)
        print(len(dataset))
        # Fixed loop to use len(dataset) instead of len(full_dataset)
        for i in tqdm(range(len(dataset)), desc=f"Generating {split} Data"):

            original_text = dataset[i]['text']
            print(original_text)

            if not original_text:
                print(f"Row {i} has no text, skipping...")
                csv_writer.writerow([i, '', 'FAILED_NO_TEXT'])
                continue

            try:
                # 1. Generate Image using FLUX.1-schnell
                image_result = pipe(
                    prompt=original_text,
                    num_inference_steps=4,
                    guidance_scale=0.0,
                    height=512,
                    width=512,
                    max_sequence_length=256
                ).images[0]
                print(image_result)
                image_result.save("flux-schnell.png")

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

            except Exception as e:
                print(f"\nError processing index {i}: {e}")
                csv_writer.writerow(
                    [i, original_text, 'FAILED_GPU_ERROR'])
                continue

        csv_file.close()


if __name__ == "__main__":
    main()
