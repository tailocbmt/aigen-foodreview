import os
import pandas as pd
import torch
from transformers import pipeline
from tqdm import tqdm

# --- Configuration ---
INPUT_DIR = "generated_fake_images"
INPUT_CSV_NAME = "generated_images_{split}.csv"
LLAVA_MODEL_ID = "llava-hf/llava-1.5-7b-hf"


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
                {"type": "image", "url": image_path},
                {"type": "text", "text": f"Here is the original text that inspired this image: '{original_text}'. Please provide a realistic, concise, and descriptive caption for what is actually shown in the image."},
            ],
        },
    ]

    with torch.no_grad():
        out = llava_pipe(text=messages, max_new_tokens=50)

    return out[0]['generated_text'].strip()


def main():
    llava_pipe = initialize_llava()

    for split in ['dev1', 'dev2', 'test']:
        csv_file_path = os.path.join(
            INPUT_DIR, INPUT_CSV_NAME.format(split=split))

        if not os.path.exists(csv_file_path):
            print(
                f"\nCould not find {csv_file_path}, skipping {split} split...")
            continue

        print(f"\nProcessing captions for {split} split...")

        # 1. Load the CSV into a Pandas DataFrame
        df = pd.read_csv(csv_file_path)

        # 2. Create the new column if it doesn't exist yet
        if 'llava_caption' not in df.columns:
            df['llava_caption'] = None

        # 3. Iterate through the DataFrame rows
        for index, row in tqdm(df.iterrows(), total=len(df), desc=f"Captioning {split}"):
            text = row['original_text']
            image_filename = str(row['saved_image_path'])

            # Resume Feature: Skip if it already has a valid caption
            existing_caption = str(row.get('llava_caption', ''))
            if pd.notna(row.get('llava_caption')) and existing_caption not in ['None', '', 'nan', 'FAILED_LLAVA_ERROR']:
                continue

            # Handle rows where FLUX previously failed to generate an image
            if 'FAILED' in image_filename:
                df.at[index, 'llava_caption'] = 'FAILED_NO_IMAGE'
                continue

            image_relative_path = os.path.join(INPUT_DIR, image_filename)

            try:
                caption = get_llava_caption(
                    llava_pipe, image_relative_path, text)
                # Append the newly generated caption to the specific cell
                df.at[index, 'llava_caption'] = caption
            except Exception as e:
                print(f"\nError captioning index {index}: {e}")
                df.at[index, 'llava_caption'] = 'FAILED_LLAVA_ERROR'

            # Optional: Save every 50 images so you don't lose progress if it crashes
            if index % 50 == 0:
                df.to_csv(csv_file_path, index=False)

        # 4. Final Save: Overwrite the CSV without the pandas index column
        df.to_csv(csv_file_path, index=False)
        print(f"\nFinished updating {csv_file_path}")


if __name__ == "__main__":
    main()
