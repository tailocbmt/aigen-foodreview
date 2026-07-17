import string

import pandas as pd  # Useful for checking punctuation
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import torch
import os
import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from transformers import AutoImageProcessor, AutoTokenizer, CLIPProcessor, CLIPModel, FlavaProcessor, FlavaModel
from torch.utils.data import DataLoader, Subset
from modules.dataset import EvonsMultimodalDataset, EvonsOfflineMultimodalDataset, EvonsOfflineMultimodalWDctDataset, HintsOfTruthMultimodalDataset, AIGenFoodMultimodalDataset, RAIDMultimodalDataset, DefactifyMultimodalDataset, SemEval2024MultimodalDataset
from larimar_base.base_models import CLIPDetector, CLIPDetectorWMemory, FLAVADetector, FLAVADetectorWMemory
from larimar_base.multi_label_models import CLIPDetectorWMemoryCoAttention, FakeNewsMultimodal, FakeNewsMultimodalCoAttention, FakeNewsMultimodalWMemory, FakeNewsMultimodalWMemoryCoAttention, FakeNewsSeparate, NetShareFusionCLIP


class MultimodalWrapper(torch.nn.Module):
    def __init__(self, base_model, text_inputs):
        super().__init__()
        self.base_model = base_model
        self.text_inputs = text_inputs

    def forward(self, pixel_values):
        # Reconstruct the dictionary that your multimodal model expects
        full_inputs = {'pixel_values': pixel_values}
        full_inputs.update(self.text_inputs)

        # Pass the full dictionary to the real model
        return self.base_model(full_inputs)


# CONFIG
folder = "visual_fakenews_mem"

# Define the path to your config file
config_path = 'configs/multimodal_config.json'

# Open and read the JSON file
with open(config_path, 'r') as file:
    config = json.load(file)

# Extract values into variables
# Options for model_name: 'clip', 'flava'
SEED = config.get('seed', 42)
model_name = config.get('model_name', 'clip')
use_memory = config.get('use_memory', 'linear')
MODEL_MODE = config.get('mode', "w/o memory")
MEMORY_SIZE = config.get('memory_size', 1280)
FEATURE_DIM = config.get('feature_dim', 512)
TEXT_BACKBONE = config.get('text_backbone', "bert-base-uncased")
VISION_BACKBONE = config.get('vision_backbone', "microsoft/resnet-50")

# Note: for 'clip', max length should be 77
dataset = config.get('dataset', 'food_review')
MAX_LENGTH = config.get('MAX_LENGTH', 512)
train_file = config.get('train_file', '')
test_file = config.get('test_file', '')
output_dir = config.get('output_dir', '')
image_dir = config.get('image_dir', '')
BATCH_SIZE = 1

device = 'cuda' if torch.cuda.is_available() else 'cpu'
available_models = ['clip', 'flava', 'fakenews',
                    'netsharefusion', 'fakenews_separate']
best_acc = 0

# MODEL SELECTION
if model_name not in available_models:
    raise ValueError(f'{model_name} not in {available_models}.')

# WEIGHTS
weights = os.listdir(output_dir)
weights = sorted(weights, key=lambda x: int(x.split('-')[1].split('.')[0]))
weights = weights[-1]
weights_dir = os.path.join(output_dir, weights)

output_dim = 1
if 'evons_multimodal' or 'foodreview' or 'defactify' or 'semeval2024' in dataset:
    output_dim = 2

if model_name == 'clip':
    backbone = CLIPModel.from_pretrained("openai/clip-vit-base-patch16")
    processor = CLIPProcessor.from_pretrained('openai/clip-vit-base-patch16')

    if MODEL_MODE == "w/o mem":
        model = CLIPDetector(backbone, processor, out_dim=output_dim)
    elif MODEL_MODE == "w/ mem":
        model = CLIPDetectorWMemory(
            backbone, processor, out_dim=output_dim, use_memory=use_memory)
    elif MODEL_MODE == "w/ coatt + mem":
        model = CLIPDetectorWMemoryCoAttention(
            backbone, processor, out_dim=output_dim, use_memory=use_memory)

elif model_name == 'flava':
    backbone = FlavaModel.from_pretrained("facebook/flava-full")
    processor = FlavaProcessor.from_pretrained("facebook/flava-full")

    if MODEL_MODE == "w/o mem":
        model = FLAVADetector(backbone, processor, out_dim=output_dim)
    elif MODEL_MODE == "w/ mem":
        model = FLAVADetectorWMemory(
            backbone, processor, out_dim=output_dim, use_memory=use_memory)

elif model_name == 'fakenews':
    processor = []
    processor.append(AutoImageProcessor.from_pretrained(VISION_BACKBONE))
    processor.append(AutoTokenizer.from_pretrained(TEXT_BACKBONE))

    if MODEL_MODE == "w/o mem":
        model = FakeNewsMultimodal(output_dim=output_dim)
    elif MODEL_MODE == "w/ mem":
        model = FakeNewsMultimodalWMemory(
            out_dim=output_dim,
            use_memory=use_memory,
            memory_size=MEMORY_SIZE,
            feature_dim=FEATURE_DIM,
            text_backbone=TEXT_BACKBONE,
            vision_backbone=VISION_BACKBONE
        )
    elif MODEL_MODE == "w/ coatt":
        model = FakeNewsMultimodalCoAttention(output_dim=output_dim)
    elif MODEL_MODE == "w/ coatt + mem":
        model = FakeNewsMultimodalWMemoryCoAttention(
            out_dim=output_dim, use_memory=use_memory)
elif model_name == 'netsharefusion':
    processor = []
    processor.append(AutoImageProcessor.from_pretrained(VISION_BACKBONE))
    processor.append(AutoTokenizer.from_pretrained(TEXT_BACKBONE))

    model = NetShareFusionCLIP(
        num_labels=output_dim)
elif model_name == 'fakenews_separate':
    text_weights_dir = "./evons_fakenews_text"
    image_weights_dir = "./evons_fakenews_vision"
    text_weights = sorted(os.listdir(text_weights_dir),
                          key=lambda x: int(x.split('-')[1].split('.')[0]))[-1]
    image_weights = sorted(os.listdir(image_weights_dir),
                           key=lambda x: int(x.split('-')[1].split('.')[0]))[-1]

    processor = []
    processor.append(AutoImageProcessor.from_pretrained(VISION_BACKBONE))
    processor.append(AutoTokenizer.from_pretrained(TEXT_BACKBONE))

    model = FakeNewsSeparate(
        os.path.join(text_weights_dir, text_weights),
        os.path.join(image_weights_dir, image_weights),
        output_dim=output_dim)
else:
    pass

if model_name != 'fakenews_separate':
    model.load_state_dict(torch.load(weights_dir))

model = model.to(device)
print(f'Model {model_name} loaded at weights: {weights}.')

# DATA
if dataset == "hints_of_truth":
    test = HintsOfTruthMultimodalDataset(
        test_file, image_dir, "test", processor, MAX_LENGTH)
elif dataset == "evons":
    real_image_dir = config.get('real_image_dir')
    full_data = EvonsMultimodalDataset(
        test_file, real_image_dir, image_dir, processor, MAX_LENGTH)

    # indices of dataset
    indices = list(range(len(full_data)))
    labels = [full_data[i]['label']
              for i in indices]  # adjust based on your dataset

    # 1. Train (80%) vs temp (20%)
    train_idx, temp_idx, train_labels, temp_labels = train_test_split(
        indices,
        labels,
        test_size=0.2,
        stratify=labels,
        random_state=42
    )

    # 2. Split temp → val (10%) + test (10%)
    val_idx, test_idx = train_test_split(
        temp_idx,
        test_size=0.5,
        stratify=temp_labels,
        random_state=42
    )
    test = Subset(full_data, test_idx)
elif dataset == "fakenews":
    test = EvonsOfflineMultimodalDataset(
        test_file, image_dir, processor, MAX_LENGTH)
elif dataset == "foodreview":
    test = AIGenFoodMultimodalDataset(
        test_file, image_dir, processor, MAX_LENGTH)
elif dataset == "RAID":
    test = RAIDMultimodalDataset(
        processor, MAX_LENGTH)
elif dataset == "semeval2024":
    test = SemEval2024MultimodalDataset(
        processor, MAX_LENGTH)
elif dataset == "defactify":
    test = DefactifyMultimodalDataset(
        processor, MAX_LENGTH)
else:
    test = EvonsOfflineMultimodalDataset(
        test_file, image_dir, processor, MAX_LENGTH)

train_df = pd.read_csv(train_file)
test_dataloader = DataLoader(test, BATCH_SIZE)
print(f'Loaded Testing File: {test_file}.')

pred_val = []
labels_val = []

# ==========================================
# VISUALIZATION & INTERPRETABILITY
# ==========================================

# Only run for our custom FakeNews models as target layers differ for CLIP/FLAVA


def plot_multimodal_tsne(model, train_df, folder, SEED=42):
    # 1. Extract the frozen memory matrix and both label sets
    text_labels, image_labels = [], []

    memory_matrix = model.episodic_memory.memory.cpu().numpy()
    memory_index = model.episodic_memory.memory_labels.cpu().numpy().tolist()

    subset_df = train_df.loc[memory_index]
    for _, row in subset_df.iterrows():
        text_labels.append(row['text_generator'])
        image_labels.append(row['image_generator'])

    text_labels = np.array(text_labels)
    image_labels = np.array(image_labels)

    # 2. Split the memory matrix into Text (312 dim) and Image (512 dim)
    # Assuming the vector format is [text_features, image_features]
    text_matrix = memory_matrix[:, :312]
    image_matrix = memory_matrix[:, 312:824]  # 312 + 512 = 824

    # 3. Apply t-SNE to Joint, Text-Only, and Image-Only representations
    tsne = TSNE(n_components=2, perplexity=30, random_state=SEED)

    joint_2d = tsne.fit_transform(memory_matrix)
    text_2d = tsne.fit_transform(text_matrix)
    image_2d = tsne.fit_transform(image_matrix)

    # 4. Setup dictionaries mapping your IDs to names
    txt_names = {
        "real": "Real Text", "qwen": "Qwen",
        "llama": "LLaMA3", "mistral": "Mistral"
    }
    img_names = {
        "real": "Real Image", "sd": "Stable Diff.",
        "flux": "FLUX", "z": "Z-Image"
    }

    # 5. Create a 2x2 grid figure
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    fig.suptitle("t-SNE of Episodic Memory Slots",
                 fontsize=18, fontweight='bold')

    ax_joint_txt = axes[0, 0]
    ax_joint_img = axes[0, 1]
    ax_split_txt = axes[1, 0]
    ax_split_img = axes[1, 1]

    # --- Plot A: Joint colored by Text Source ---
    for label_id, name in txt_names.items():
        idx = (text_labels == label_id)
        ax_joint_txt.scatter(joint_2d[idx, 0], joint_2d[idx, 1],
                             label=name, alpha=0.7, s=40)
    ax_joint_txt.set_title("Joint Memory (Colored by Text Source)")
    ax_joint_txt.legend()
    ax_joint_txt.grid(True, linestyle='--', alpha=0.5)

    # --- Plot B: Joint colored by Image Source ---
    for label_id, name in img_names.items():
        idx = (image_labels == label_id)
        ax_joint_img.scatter(joint_2d[idx, 0], joint_2d[idx, 1],
                             label=name, alpha=0.7, s=40)
    ax_joint_img.set_title("Joint Memory (Colored by Image Source)")
    ax_joint_img.legend()
    ax_joint_img.grid(True, linestyle='--', alpha=0.5)

    # --- Plot C: Text-Only Split colored by Text Source ---
    for label_id, name in txt_names.items():
        idx = (text_labels == label_id)
        ax_split_txt.scatter(text_2d[idx, 0], text_2d[idx, 1],
                             label=name, alpha=0.7, s=40)
    ax_split_txt.set_title("Text-Only Split (Colored by Text Source)")
    ax_split_txt.legend()
    ax_split_txt.grid(True, linestyle='--', alpha=0.5)

    # --- Plot D: Image-Only Split colored by Image Source ---
    for label_id, name in img_names.items():
        idx = (image_labels == label_id)
        ax_split_img.scatter(image_2d[idx, 0], image_2d[idx, 1],
                             label=name, alpha=0.7, s=40)
    ax_split_img.set_title("Image-Only Split (Colored by Image Source)")
    ax_split_img.legend()
    ax_split_img.grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()

    # Save
    save_dir = f"./evons_data/{folder}"
    os.makedirs(save_dir, exist_ok=True)

    plt.savefig(f"{save_dir}/test_tnse.png", bbox_inches='tight', dpi=300)
    plt.close()
