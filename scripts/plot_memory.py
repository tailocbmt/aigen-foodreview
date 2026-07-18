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
folder = "./evons_data/test"

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
model.set_memory_mode("read")
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
# 4. Setup dictionaries mapping your IDs to names
txt_names = {
    "real": "Real Text", "qwen": "Qwen",
    "llama": "LLaMA3", "mistral": "Mistral"
}
img_names = {
    "real": "Real Image", "sd": "Stable Diff.",
    "flux": "FLUX", "z": "Z-Image"
}

# Only run for our custom FakeNews models as target layers differ for CLIP/FLAVA
# 1. Extract the frozen memory matrix and both label sets
TEXT_INDEX = 312
text_labels, image_labels = [], []

memory_matrix = model.episodic_memory.memory.detach().cpu().numpy()
memory_index = model.episodic_memory.memory_labels.detach().cpu().numpy().tolist()

subset_df = train_df.loc[memory_index]
for _, row in subset_df.iterrows():
    text_labels.append(row['text_generator'])
    image_labels.append(row['image_generator'])

text_labels = np.array(text_labels)
image_labels = np.array(image_labels)

# 2. Split the memory matrix into Text (312 dim) and Image (512 dim)
# Assuming the vector format is [text_features, image_features]
text_matrix = memory_matrix[:, :TEXT_INDEX]
image_matrix = memory_matrix[:, TEXT_INDEX:]  # 312 + 512 = 824


def plot_multimodal_tsne(folder, SEED=42):
    # 3. Apply t-SNE to Joint, Text-Only, and Image-Only representations
    tsne = TSNE(n_components=2, perplexity=30, random_state=SEED)

    joint_2d = tsne.fit_transform(memory_matrix)
    text_2d = tsne.fit_transform(text_matrix)
    image_2d = tsne.fit_transform(image_matrix)

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
    plt.savefig(f"{folder}/test_tnse.png", bbox_inches='tight', dpi=300)
    plt.close()


def plot_mismatch_example(folder, SEED=42):
    for i in range(len(test)):
        print(f"Processing Example {i}/{len(test)}...")
        sample = test[i]

        # 2. Forward pass for this specific sample
        inputs_vis = {key: tensor.unsqueeze(0).squeeze(1).to(device)
                      for key, tensor in sample['inputs'].items()}
        input_label = sample["label"].unsqueeze(0).squeeze(1).cpu().numpy()

        # We don't track gradients for visualization
        with torch.no_grad():
            output, retrieved = model(inputs_vis, return_memory=True)
            probs = torch.sigmoid(output)
            preds = (probs > 0.5).int().cpu().numpy()
            results = (input_label == preds)

            is_correct = bool(results.item()) if hasattr(
                results, 'item') else bool(results[0])

        # Retrieve Top-K indices
        # Squeeze to handle potential batch dimensions like [1, 10] -> [10]
        topk_index = retrieved["top10_labels"].squeeze().cpu().numpy().tolist()

        # Map the Top-K indices to integer positions (0-511) to plot them correctly
        topk_positions = [memory_index.index(idx) for idx in topk_index]

        # 3. Extract x_input, x_output and prepare for t-SNE
        x_input = retrieved["x_input"].squeeze().cpu().numpy()  # Shape: (824,)
        x_text = x_input[:TEXT_INDEX]
        x_image = x_input[TEXT_INDEX:]

        x_output = retrieved["x_output"].squeeze(
        ).cpu().numpy()  # Shape: (824,)
        x_text_output = x_output[:TEXT_INDEX]
        x_image_output = x_output[TEXT_INDEX:]

        # Append x_input to the end of the memory matrices
        joint_data = np.vstack([memory_matrix, x_input, x_output])
        text_data = np.vstack([text_matrix, x_text, x_text_output])
        image_data = np.vstack([image_matrix, x_image, x_image_output])

        # 4. Apply t-SNE
        # (Must be done per loop so x_input gets properly mapped into the space)
        tsne = TSNE(n_components=2, perplexity=30, random_state=SEED)
        joint_2d_all = tsne.fit_transform(joint_data)
        text_2d_all = tsne.fit_transform(text_data)
        image_2d_all = tsne.fit_transform(image_data)

        # 5. Correctly separate the points
        # Base memory is everything EXCEPT the last two points
        joint_2d = joint_2d_all[:-2]
        joint_x_in = joint_2d_all[-2]
        joint_x_out = joint_2d_all[-1]

        text_2d = text_2d_all[:-2]
        text_x_in = text_2d_all[-2]
        text_x_out = text_2d_all[-1]

        image_2d = image_2d_all[:-2]
        image_x_in = image_2d_all[-2]
        image_x_out = image_2d_all[-1]

        # 6. Create the 2x2 grid figure
        fig, axes = plt.subplots(2, 2, figsize=(16, 14))

        # Set dynamic title based on prediction correctness
        pred_text = "CORRECT" if is_correct else "INCORRECT"
        title_color = "darkgreen" if is_correct else "darkred"
        fig.suptitle(f"t-SNE Retrieval Match (Test Sample {i}) | Prediction: {pred_text}",
                     fontsize=18, fontweight='bold', color=title_color)

        # Helper function to plot each subplot cleanly
        def plot_axis(ax, memory_pts, input_pt, output_pt, labels, names_dict, title):
            # A) Plot standard memory slots
            for label_id in np.unique(labels):
                idx = (labels == label_id)
                display_name = names_dict.get(label_id, str(label_id))
                ax.scatter(memory_pts[idx, 0], memory_pts[idx, 1],
                           label=display_name, alpha=0.3, s=40)

            # B) Highlight the Top-K retrieved memory slots (Cyan Diamonds)
            ax.scatter(memory_pts[topk_positions, 0], memory_pts[topk_positions, 1],
                       marker='D', facecolors='cyan', edgecolors='black', s=100, linewidths=1.2,
                       label='Top-10 Retrieved', zorder=5)

            # Add Text Labels for Top-K points
            for pos in topk_positions:
                x, y = memory_pts[pos, 0], memory_pts[pos, 1]
                raw_label = labels[pos]
                display_label = names_dict.get(
                    raw_label, str(raw_label)) + f" {pos}"

                # Annotate with a small offset and a readable background box
                ax.annotate(display_label, (x, y), xytext=(6, 6),
                            textcoords='offset points', fontsize=9, fontweight='bold',
                            bbox=dict(boxstyle="round,pad=0.2",
                                      fc="white", alpha=0.7, ec="gray"),
                            zorder=6)

            # C) Plot the actual query input (Large Red Star)
            ax.scatter(input_pt[0], input_pt[1],
                       marker='*', color='red', s=400, edgecolor='black', linewidths=1,
                       label='Input Query (x_input)', zorder=7)

            # D) Plot the output query (Large Gold Triangle)
            ax.scatter(output_pt[0], output_pt[1],
                       marker='^', color='gold', s=350, edgecolor='black', linewidths=1,
                       label='Output (x_output)', zorder=7)

            # E) Draw an arrow pointing from x_input to x_output
            ax.annotate("",
                        # Target (where the arrow points)
                        xy=(output_pt[0], output_pt[1]),
                        # Origin (where the arrow starts)
                        xytext=(input_pt[0], input_pt[1]),
                        arrowprops=dict(
                            arrowstyle="->", color="black", lw=2.5, ls="--", shrinkA=8, shrinkB=8),
                        zorder=6)  # Kept just below the markers so it doesn't cross over them

            ax.set_title(title)
            ax.legend(loc='best')
            ax.grid(True, linestyle='--', alpha=0.5)

        # Plot all 4 panels passing both in and out points
        plot_axis(axes[0, 0], joint_2d, joint_x_in, joint_x_out, text_labels,
                  txt_names, "Joint Memory (Colored by Text Source)")
        plot_axis(axes[0, 1], joint_2d, joint_x_in, joint_x_out, image_labels,
                  img_names, "Joint Memory (Colored by Image Source)")
        plot_axis(axes[1, 0], text_2d, text_x_in, text_x_out, text_labels,
                  txt_names, "Text-Only Split (Colored by Text Source)")
        plot_axis(axes[1, 1], image_2d, image_x_in, image_x_out, image_labels,
                  img_names, "Image-Only Split (Colored by Image Source)")

        plt.tight_layout()

        # Save with dynamic filename so they don't overwrite each other
        save_path = os.path.join(folder, f"example_{i}_tsne.png")
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.close()


plot_multimodal_tsne(folder=folder, SEED=SEED)
plot_mismatch_example(folder=folder, SEED=SEED)
