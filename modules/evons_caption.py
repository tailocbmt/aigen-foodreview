import csv
import os
import pandas as pd
import torch
from transformers import pipeline
from datasets import load_dataset
from tqdm import tqdm

# --- Configuration ---
INPUT_DIR = "generated_evons"
OUTPUT_DIR = "original_image_captions"
CSV_INPUT_NAME = "evons_generated_images.csv"
CSV_OUTPUT_NAME = "evons_original_captions.csv"
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


def get_llava_caption(llava_pipe, image_path, original_title, original_description):
    """Generates a caption using the HF Pipeline."""
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_path},
                {"type": "text",
                    "text": f"You are acting as a journalist generating content for a news media source. Original Headline: '{original_title}'. Original Description: '{original_description}'. Based on the provided image and this context, please write a highly realistic, human-like news caption and a short, synthetic article paragraph (3-4 sentences). Do not simply list the tags; weave them into a natural sentence that a journalist might use."},
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
    input_csv_path = os.path.join(INPUT_DIR, f"evons.csv")
    csv_file_path = os.path.join(
        INPUT_DIR, CSV_INPUT_NAME
    )

    df = pd.read_csv(input_csv_path, encoding='utf-8', low_memory=False)

    generated_texts = []
    for i in tqdm(range(len(df)), desc=f"Generating Data"):
        item = df.iloc[i]

        # Extract text and the native PIL Image object from the dataset
        title = item.get('title', '')
        description = item.get('description', '')
        image_path = item.get('fake_img_paths')

        if image_path is "":
            generated_texts.append("FAILED_NO_IMAGE_IN_DATASET")
            continue

        try:
            # The llava pipeline natively accepts PIL Images in memory!
            with torch.no_grad():
                out = get_llava_caption(
                    llava_pipe, image_path, title, description)

            print(out)
            caption = out

            # Write to CSV
            generated_texts.append(caption)

        except Exception as e:
            print(f"\nError captioning index {i}: {e}")
            generated_texts.append('FAILED_BLIP_ERROR')

        print(f"Finished saving {csv_file_path}")

    df["fake_description"] = generated_texts
    df.to_csv(os.path.join(
        OUTPUT_DIR, CSV_OUTPUT_NAME
    ), index=False)


if __name__ == "__main__":
    main()
