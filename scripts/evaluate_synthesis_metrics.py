import os
import json
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM
from modules.utils import text_metrics, perplexity_gptneo, image_metrics

IMAGE_OUTPUT_DIR = "evons_qwen_{IMAGE_MODEL_NAME}"
MODEL_NAME = "EleutherAI/gpt-neo-125M"  # or "EleutherAI/gpt-neo-1.3B"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
model.eval()

df = pd.read_csv("evons_data/evons_exp.csv")
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
df = df[df["image_fn"] != "mwn5977-t.jpg"]

fake_title_cols = [
    "qwen_rewritten_title",
    "llama_rewritten_title",
    "mistral_rewritten_title",
]

fake_des_cols = [
    "qwen_rewritten_description",
    "llama_rewritten_description",
    "mistral_rewritten_description",
]

fake_img_cols = [
    "sd_img_path",
    "flux_img_path",
    "z_img_path",
    "sdxl_img_path",
]

results = []

for _, row in df.iterrows():
    row_id = row.index

    real_title = str(row["title"])
    real_des = str(row["description"])

    real_img_path = os.path.join(
        "images",
        row["media_source"],
        row["image_fn"].replace("usas", "usasup"),
    )

    # ---- REAL ----
    real_title_m = text_metrics(real_title)
    real_title_m["PPL"] = perplexity_gptneo(tokenizer, model, real_title)

    real_des_m = text_metrics(real_des)
    real_des_m["PPL"] = perplexity_gptneo(tokenizer, model, real_des)

    real_img_m = image_metrics(real_img_path)

    row_metrics = {
        "id": row_id,
        "is_fake": row['is_fake'],
        "real": {
            "title": real_title_m,
            "description": real_des_m,
            "image": real_img_m,
        },
        "generated": {}
    }

    # ---- GENERATED TEXT ----
    for title_col, des_col in zip(fake_title_cols, fake_des_cols):
        model_name = title_col.replace("_rewritten_title", "")

        gen_title = str(row[title_col])
        gen_des = str(row[des_col])

        gen_title_m = text_metrics(gen_title)
        gen_title_m["PPL"] = perplexity_gptneo(tokenizer, model, gen_title)

        gen_des_m = text_metrics(gen_des)
        gen_des_m["PPL"] = perplexity_gptneo(tokenizer, model, gen_des)

        if model_name not in row_metrics["generated"]:
            row_metrics["generated"][model_name] = {}

        row_metrics["generated"][model_name]["title"] = gen_title_m
        row_metrics["generated"][model_name]["description"] = gen_des_m

    # ---- GENERATED IMAGES ----
    for gen_img_col in fake_img_cols:
        model_name = gen_img_col.replace("_img_path", "")
        gen_img_path = row[gen_img_col]

        gen_img_m = image_metrics(gen_img_path)

        if model_name not in row_metrics["generated"]:
            row_metrics["generated"][model_name] = {}

        row_metrics["generated"][model_name]["image"] = gen_img_m

    results.append(row_metrics)

# ---- SAVE JSON ----
with open("evons_metrics_output.json", "w") as f:
    json.dump(results, f, indent=2)
