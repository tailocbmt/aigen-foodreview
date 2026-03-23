import csv
import os
import pandas as pd
import torch
from transformers import pipeline
from datasets import load_dataset
from tqdm import tqdm

# --- Configuration ---
OUTPUT_DIR = "original_image_captions"
CSV_OUTPUT_NAME = "original_captions_{split}.csv"
LLAVA_MODEL_ID = "llava-hf/llava-1.5-7b-hf"
BLIP_MODEL_ID = "Salesforce/blip-image-captioning-large"

os.makedirs(OUTPUT_DIR, exist_ok=True)


def initialize_blip():
    """Initializes BLIP in standard 16-bit precision."""
    print(f"Loading {BLIP_MODEL_ID} in float16...")

    blip_pipe = pipeline(
        "image-to-text",  # Changed from image-text-to-text
        model=BLIP_MODEL_ID,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    return blip_pipe


def get_blip_caption(blip_pipe, image_path):
    """Generates a pure description using the BLIP Pipeline."""

    # BLIP doesn't need conversational instructions. You just pass the image!
    with torch.no_grad():
        out = blip_pipe(image_path, max_new_tokens=50)

    generated_data = out[0]['generated_text']

    # If the pipeline returns the full conversation history as a list:
    if isinstance(generated_data, list):
        # Grab the 'content' of the final assistant response
        return generated_data[-1]['content'].strip()

    # If the pipeline returns a standard string:
    return generated_data.strip()


def initialize_llava():
    """Initializes LLaVA 1.5 in standard 16-bit precision."""
    print(f"Loading {LLAVA_MODEL_ID} in float16...")

    llava_pipe = pipeline(
        "image-text-to-text",
        model=LLAVA_MODEL_ID,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    return llava_pipe


def get_llava_caption(llava_pipe, image_path, original_text):
    """Generates a caption using the HF Pipeline."""
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_path},
                {"type": "text", "text": f"Here is the original text that inspired this image: '{original_text}'. Please provide a realistic, concise, and descriptive caption for what is actually shown in the image."},
            ],
        },
    ]

    with torch.no_grad():
        # max_length=None added to suppress the length warning!
        out = llava_pipe(text=messages, max_new_tokens=50, max_length=None)

    generated_data = out[0]['generated_text']

    # If the pipeline returns the full conversation history as a list:
    if isinstance(generated_data, list):
        # Grab the 'content' of the final assistant response
        return generated_data[-1]['content'].strip()

    # If the pipeline returns a standard string:
    return generated_data.strip()


def main():
    llava_pipe = initialize_llava()

    # 2. Load the Original Dataset
    print("\nLoading dataset 'michiel/hints_of_truth'...")
    full_dataset = load_dataset("michiel/hints_of_truth")

    for split in ['dev1', 'dev2', 'test']:
        if split not in full_dataset:
            print(f"Split '{split}' not found in dataset. Skipping.")
            continue

        dataset = full_dataset[split]
        csv_file_path = os.path.join(
            OUTPUT_DIR, CSV_OUTPUT_NAME.format(split=split))

        print(f"\nProcessing original images for {split} split...")

        # Open a new CSV file to record the results
        with open(csv_file_path, mode='w', newline='', encoding='utf-8') as file:
            csv_writer = csv.writer(file)
            csv_writer.writerow(['index', 'original_text', 'llava_caption'])

            # Iterate through the dataset
            for i in tqdm(range(len(dataset)), desc=f"Captioning {split}"):
                item = dataset[i]

                # Extract text and the native PIL Image object from the dataset
                text = item.get('text', '')
                image = item.get('image')

                if image is None:
                    csv_writer.writerow(
                        [i, text, 'FAILED_NO_IMAGE_IN_DATASET'])
                    continue

                try:
                    # The llava pipeline natively accepts PIL Images in memory!
                    with torch.no_grad():
                        out = get_llava_caption(llava_pipe, image, text)

                    caption = out[0]['generated_text'].strip()

                    # Write to CSV
                    csv_writer.writerow([i, text, caption])

                except Exception as e:
                    print(f"\nError captioning index {i}: {e}")
                    csv_writer.writerow([i, text, 'FAILED_BLIP_ERROR'])

        print(f"Finished saving {csv_file_path}")


if __name__ == "__main__":
    main()
