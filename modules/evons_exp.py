import os
import pandas as pd
import torch
import gc
from larimar_base.model_loader import initialize_image_pipeline, initialize_text_model
from PIL import Image
from tqdm import tqdm

# --- Configuration ---
TEXT_MODEL_NAME = "qwen"
IMAGE_MODEL_NAME = "qwen_image"
OUTPUT_DIR = f"evons_data"
IMAGE_OUTPUT_DIR = f"evons_qwen_{IMAGE_MODEL_NAME}"
CSV_OUTPUT_NAME = "evons_exp.csv"

os.makedirs(f"{OUTPUT_DIR}/{IMAGE_OUTPUT_DIR}", exist_ok=True)


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


def main():
    # tokenizer, qwen_model = initialize_text_model(MODEL_NAME=TEXT_MODEL_NAME)
    pipe = initialize_image_pipeline(MODEL_NAME=IMAGE_MODEL_NAME)

    print("\nLoading dataset 'michiel/hints_of_truth'...")
    input_csv_path = os.path.join(OUTPUT_DIR, f"evons_exp.csv")
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
        # rewritten_title = rewrite_caption(
        #     tokenizer, qwen_model, title)
        rewritten_title = row['qwen_rewritten_title']

        # 2. Rewrite description with local Qwen
        # rewritten_description = rewrite_caption(
        #     tokenizer, qwen_model, description)
        rewritten_description = row['qwen_rewritten_description']

        original_text = f"{rewritten_description}".strip()

        try:
            # 1. Generate Image using FLUX.1-schnell
            with torch.inference_mode():
                image_result = pipe(
                    prompt=original_text,
                    num_inference_steps=8,
                    negative_prompt=" ",
                    true_cfg_scale=4.0,
                    # num_inference_steps=9, guidance_scale=0.0, z_image
                    # num_inference_steps=4, guidance_scale=1, normal
                    # guidance_scale=0.0,
                    height=512,
                    width=512
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

    # df["qwen_rewritten_title"] = titles
    # df["qwen_rewritten_description"] = descriptions
    df[f"{IMAGE_MODEL_NAME}_img_path"] = saved_image_paths
    df.to_csv(csv_file_path, index=False)


if __name__ == "__main__":
    main()
