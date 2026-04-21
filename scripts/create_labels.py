import os

import pandas as pd
from sklearn.model_selection import train_test_split


# =========================================================
# 1) Stratified split by is_fake
# =========================================================
def stratified_split(df, seed=42):
    train_df, temp_df = train_test_split(
        df,
        test_size=0.2,
        random_state=seed,
        stratify=df["is_fake"],
        shuffle=True
    )

    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.5,
        random_state=seed,
        stratify=temp_df["is_fake"],
        shuffle=True
    )

    return (
        train_df.reset_index(drop=True),
        val_df.reset_index(drop=True),
        test_df.reset_index(drop=True),
    )


# =========================================================
# 2) Expand dataset
# Strategy:
# - 1 x real text + real image
# - all fake texts + real image
# - real text + all fake images
# - balanced sampled fake text + fake image pairs
#
# Columns expected in df:
#   real_title
#   real_img_path
#   qwen_rewritten_title
#   llama_rewritten_title
#   mistral_rewritten_title
#   sd_img_path
#   flux_img_path
#   z_img_path
#   qwen_img_path
#   is_fake
# =========================================================
def build_fake_text_pool(row):
    pool = []

    mapping = {
        "qwen": (row.get("qwen_rewritten_title"), row.get("qwen_rewritten_description")),
        "llama": (row.get("llama_rewritten_title"), row.get("llama_rewritten_description")),
        "mistral": (row.get("mistral_rewritten_title"), row.get("mistral_rewritten_description")),
    }

    for gen_name, text_value in mapping.items():
        if pd.notna(text_value) and str(text_value).strip():
            pool.append({
                "text_generator": gen_name,
                "title": text_value[0],
                "description": text_value[1]
            })

    return pool


def build_fake_image_pool(row):
    pool = []

    mapping = {
        "sd": row.get("sd_img_path"),
        "flux": row.get("flux_img_path"),
        "z": row.get("z_img_path"),
        "qwen_image": row.get("qwen_img_path"),
    }

    for gen_name, img_value in mapping.items():
        if pd.notna(img_value) and str(img_value).strip():
            pool.append({
                "image_generator": gen_name,
                "image_path": img_value
            })

    return pool


def sampled_cross_pairs(fake_text_pool, fake_image_pool, max_pairs=None):
    """
    Balanced deterministic pairing without importing random.
    Rotates image choices across text generators.

    If max_pairs is None:
        uses min(len(fake_text_pool), len(fake_image_pool))
    """
    if not fake_text_pool or not fake_image_pool:
        return []

    if max_pairs is None:
        max_pairs = min(len(fake_text_pool), len(fake_image_pool))

    pairs = []
    num_texts = len(fake_text_pool)
    num_imgs = len(fake_image_pool)

    for i in range(max_pairs):
        t = fake_text_pool[i % num_texts]
        img = fake_image_pool[(i * 2 + 1) % num_imgs]
        pairs.append((t, img))

    # remove accidental duplicates if any
    unique_pairs = []
    seen = set()
    for t, img in pairs:
        key = (t["text_generator"], img["image_generator"])
        if key not in seen:
            seen.add(key)
            unique_pairs.append((t, img))

    return unique_pairs


def expand_dataset(
    df,
    include_all_fake_text_real_image=True,
    include_all_real_text_fake_image=True,
    max_fake_fake_pairs=4
):
    rows = []

    df = df.reset_index(drop=True)

    for source_id, row in df.iterrows():
        real_title = row["title"]
        real_des = row["description"]
        real_img = os.path.join(row["media_source"], row["image_fn"])
        original_is_fake = row["is_fake"]

        fake_text_pool = build_fake_text_pool(row)
        fake_image_pool = build_fake_image_pool(row)

        # 1) real text + real image
        if pd.notna(real_title) and str(real_title).strip() and pd.notna(real_img) and str(real_img).strip():
            rows.append({
                "source_id": source_id,
                "title": real_title,
                "description": real_des,
                "image_path": real_img,
                "text_manipulated": 0,
                "image_manipulated": 0,
                "text_generator": "real",
                "image_generator": "real",
                "combo_label": "real_text_real_img",
                "label_text": 0,
                "label_image": 0,
                "label": [0, 0],
                "is_fake": original_is_fake
            })

        # 2) all fake texts + real image
        if include_all_fake_text_real_image and pd.notna(real_img) and str(real_img).strip():
            for item in fake_text_pool:
                rows.append({
                    "source_id": source_id,
                    "title": item["title"],
                    "description": item["description"],
                    "image_path": real_img,
                    "text_manipulated": 1,
                    "image_manipulated": 0,
                    "text_generator": item["text_generator"],
                    "image_generator": "real",
                    "combo_label": "fake_text_real_img",
                    "label_text": 1,
                    "label_image": 0,
                    "label": [1, 0],
                    "is_fake": original_is_fake
                })

        # 3) real text + all fake images
        if include_all_real_text_fake_image and pd.notna(real_title) and str(real_title).strip():
            for item in fake_image_pool:
                rows.append({
                    "source_id": source_id,
                    "title": real_title,
                    "description": real_des,
                    "image_path": item["image_path"],
                    "text_manipulated": 0,
                    "image_manipulated": 1,
                    "text_generator": "real",
                    "image_generator": item["image_generator"],
                    "combo_label": "real_text_fake_img",
                    "label_text": 0,
                    "label_image": 1,
                    "label": [0, 1],
                    "is_fake": original_is_fake
                })

        # 4) sampled fake text + fake image pairs
        cross_pairs = sampled_cross_pairs(
            fake_text_pool,
            fake_image_pool,
            max_pairs=max_fake_fake_pairs
        )

        for fake_text_item, fake_img_item in cross_pairs:
            rows.append({
                "source_id": source_id,
                "title": fake_text_item["title"],
                "description": fake_text_item["description"],
                "image_path": fake_img_item["image_path"],
                "text_manipulated": 1,
                "image_manipulated": 1,
                "text_generator": fake_text_item["text_generator"],
                "image_generator": fake_img_item["image_generator"],
                "combo_label": "fake_text_fake_img",
                "label_text": 1,
                "label_image": 1,
                "label": [1, 1],
                "is_fake": original_is_fake
            })

    return pd.DataFrame(rows)


# =========================================================
# 3) Full pipeline
# =========================================================
def create_multimodal_splits(llama3_csv, input_csv, output_dir=".", seed=42):
    IMAGE_OUTPUT_DIR = "evons_qwen_{IMAGE_MODEL_NAME}"
    llama3_df = pd.read_csv(llama3_csv)
    df = pd.read_csv(input_csv)

    df['llama3_rewritten_title'] = llama3_df['llama3_rewritten_title']
    df['llama3_rewritten_description'] = llama3_df['llama3_rewritten_description']
    df = df.rename(columns={
        "qwen_rewritten_title": "qwen_old_rewritten_title",
        "qwen_rewritten_description": "qwen_old_rewritten_description",
        "qwen_new_rewritten_title": "qwen_rewritten_title",
        "qwen_new_rewritten_description": "qwen_rewritten_description",
        "mixtral_rewritten_title": "mistral_rewritten_title",
        "mixtral_rewritten_description": "mistral_rewritten_description",
        "llama3_rewritten_title": "llama_rewritten_title",
        "llama3_rewritten_description": "llama_rewritten_description",
        "fake_img_paths": "sd_img_path",
        "z_image_img_path": "z_img_path"
    })

    df["sd_img_path"] = df["sd_img_path"].apply(
        lambda x: os.path.join(IMAGE_OUTPUT_DIR.format(IMAGE_MODEL_NAME="sd"), x))
    df["flux_img_path"] = df["flux_img_path"].apply(
        lambda x: os.path.join(IMAGE_OUTPUT_DIR.format(IMAGE_MODEL_NAME="flux"), x))
    df["z_img_path"] = df["z_img_path"].apply(
        lambda x: os.path.join(IMAGE_OUTPUT_DIR.format(IMAGE_MODEL_NAME="z_image"), x))
    df["sdxl_img_path"] = df["sdxl_img_path"].apply(
        lambda x: os.path.join(IMAGE_OUTPUT_DIR.format(IMAGE_MODEL_NAME="sdxl"), x))
    df = df.drop(columns=["Unnamed: 0"])

    cols = [
        "qwen_rewritten_title",
        "llama_rewritten_title",
        "mistral_rewritten_title",
        "qwen_rewritten_description",
        "llama_rewritten_description",
        "mistral_rewritten_description",
    ]
    df[cols] = df[cols].apply(
        lambda col: col.str.replace(r'^"""|"""$', '', regex=True))

    required_cols = [
        "title",
        "description",
        "media_source",
        "image_fn",
        "qwen_rewritten_title",
        "llama_rewritten_title",
        "mistral_rewritten_title",
        "qwen_rewritten_description",
        "llama_rewritten_description",
        "mistral_rewritten_description",
        "sd_img_path",
        "flux_img_path",
        "z_img_path",
        "sdxl_img_path",
        "is_fake",
    ]

    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    # split first to avoid leakage
    train_df, val_df, test_df = stratified_split(df, seed=seed)

    # expand each split
    train_expanded = expand_dataset(
        train_df,
        include_all_fake_text_real_image=True,
        include_all_real_text_fake_image=True,
        max_fake_fake_pairs=4
    )

    val_expanded = expand_dataset(
        val_df,
        include_all_fake_text_real_image=True,
        include_all_real_text_fake_image=True,
        max_fake_fake_pairs=4
    )

    test_expanded = expand_dataset(
        test_df,
        include_all_fake_text_real_image=True,
        include_all_real_text_fake_image=True,
        max_fake_fake_pairs=4
    )

    # save
    train_path = f"{output_dir}/train_multilabel.csv"
    val_path = f"{output_dir}/val_multilabel.csv"
    test_path = f"{output_dir}/test_multilabel.csv"

    print(train_expanded.head(10))
    print(val_expanded.head(10))
    print(test_expanded.head(10))

    train_expanded.to_csv(train_path, index=False)
    val_expanded.to_csv(val_path, index=False)
    test_expanded.to_csv(test_path, index=False)

    # print summary
    print("=== Original split sizes ===")
    print("train:", len(train_df))
    print("val  :", len(val_df))
    print("test :", len(test_df))

    print("\n=== Original is_fake distribution ===")
    print("train")
    print(train_df["is_fake"].value_counts(normalize=True).sort_index())
    print("val")
    print(val_df["is_fake"].value_counts(normalize=True).sort_index())
    print("test")
    print(test_df["is_fake"].value_counts(normalize=True).sort_index())

    print("\n=== Expanded split sizes ===")
    print("train:", len(train_expanded))
    print("val  :", len(val_expanded))
    print("test :", len(test_expanded))

    print("\n=== Combo distribution ===")
    print("train")
    print(train_expanded["combo_label"].value_counts())
    print("val")
    print(val_expanded["combo_label"].value_counts())
    print("test")
    print(test_expanded["combo_label"].value_counts())

    print("\nSaved to:")
    print(train_path)
    print(val_path)
    print(test_path)

    return train_expanded, val_expanded, test_expanded


# =========================================================
# 4) Example run
# =========================================================
if __name__ == "__main__":
    train_expanded, val_expanded, test_expanded = create_multimodal_splits(
        llama3_csv="evons_data/evons_exp_llama3.csv",
        input_csv="evons_data/evons_exp.csv",
        output_dir="evons_data",
        seed=42
    )
