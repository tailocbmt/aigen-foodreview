"""
Experiment 6 Integration: Cross-Modal Coherence Analysis
Designed to plug directly into your existing inference loop.

This script accumulates features during inference, then computes:
- Pre-memory cosine similarity (raw image vs raw text)
- Post-memory cosine similarity (sliced from x_output)
- Delta per sample, stratified by multi-label composition type
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ttest_rel
from tqdm import tqdm
import os
import torch
from typing import Dict, List, Tuple

from sklearn.model_selection import train_test_split
from sklearn.cross_decomposition import CCA
import json
from transformers import AutoImageProcessor, AutoTokenizer, CLIPProcessor, CLIPModel, FlavaProcessor, FlavaModel
from torch.utils.data import DataLoader, Subset
from modules.dataset import EvonsMultimodalDataset, EvonsOfflineMultimodalDataset, EvonsOfflineMultimodalWDctDataset, HintsOfTruthMultimodalDataset, AIGenFoodMultimodalDataset, RAIDMultimodalDataset, DefactifyMultimodalDataset, SemEval2024MultimodalDataset
from larimar_base.base_models import CLIPDetector, CLIPDetectorWMemory, FLAVADetector, FLAVADetectorWMemory
from larimar_base.multi_label_models import CLIPDetectorWMemoryCoAttention, FakeNewsMultimodal, FakeNewsMultimodalCoAttention, FakeNewsMultimodalWMemory, FakeNewsMultimodalWMemoryCoAttention, FakeNewsSeparate, NetShareFusionCLIP

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


# =============================================================================
# PART 1: ACCUMULATOR CLASS (To integrate into your existing loop)
# =============================================================================


class CoherenceAccumulator:
    """
    Accumulates features and labels during inference.
    Now uses explicit image_dim and text_dim.
    """

    def __init__(self, image_dim: int = 512, text_dim: int = 312):
        self.image_dim = image_dim
        self.text_dim = text_dim
        self.x_image_list = []      # Pre-memory image features
        self.x_text_list = []       # Pre-memory text features
        self.o_image_list = []      # Post-memory image features
        self.o_text_list = []       # Post-memory text features
        self.text_labels = []       # [0 or 1]
        self.image_labels = []      # [0 or 1]
        self.preds_text = []
        self.preds_image = []

    def add_sample(self, x_input: np.ndarray, x_output: np.ndarray,
                   label: np.ndarray, preds: np.ndarray = None):
        """
        x_input and x_output are concatenated vectors of shape (image_dim + text_dim,)
        """
        # Slice pre-memory
        x_image = x_input[:self.image_dim]
        x_text = x_input[self.image_dim:]

        # Slice post-memory
        o_image = x_output[:self.image_dim]
        o_text = x_output[self.image_dim:]

        self.x_image_list.append(x_image)
        self.x_text_list.append(x_text)
        self.o_image_list.append(o_image)
        self.o_text_list.append(o_text)
        self.text_labels.append(label[0])
        self.image_labels.append(label[1])

        if preds is not None:
            self.preds_text.append(preds[0])
            self.preds_image.append(preds[1])

    def get_results(self) -> Dict[str, np.ndarray]:
        return {
            'x_image': np.array(self.x_image_list),
            'x_text': np.array(self.x_text_list),
            'o_image': np.array(self.o_image_list),
            'o_text': np.array(self.o_text_list),
            'text_label': np.array(self.text_labels),
            'image_label': np.array(self.image_labels),
        }

# =============================================================================
# PART 2: MODIFIED INFERENCE LOOP (Integrate this into your existing script)
# =============================================================================


def run_inference_and_collect(
    model,
    dataloader,
    device,
    image_dim: int = 512,
    text_dim: int = 312,
    max_samples: int = None
) -> CoherenceAccumulator:
    """
    Run inference and collect pre/post features.
    """
    model.eval()
    accumulator = CoherenceAccumulator(image_dim=image_dim, text_dim=text_dim)

    count = 0
    with torch.no_grad():
        for sample in tqdm(dataloader, desc="Collecting Coherence Features"):
            inputs_vis = {key: tensor.unsqueeze(0).squeeze(1).to(device)
                          for key, tensor in sample['inputs'].items()}
            input_label = sample["label"].unsqueeze(0).squeeze(1).cpu().numpy()

            output, retrieved = model(inputs_vis, return_memory=True)
            probs = torch.sigmoid(output)
            preds = (probs > 0.5).int().cpu().numpy().flatten()

            x_input = retrieved["x_input"].squeeze(
            ).cpu().numpy()   # shape (824,)
            x_output = retrieved["x_output"].squeeze(
            ).cpu().numpy()  # shape (824,)

            accumulator.add_sample(
                x_input=x_input,
                x_output=x_output,
                label=input_label.flatten(),
                preds=preds
            )

            count += 1
            if max_samples and count >= max_samples:
                break

    return accumulator


# =============================================================================
# PART 3: ANALYSIS AND VISUALIZATION (Same as before, with small updates)
# =============================================================================

def compute_cca_similarity(img_feats: np.ndarray, txt_feats: np.ndarray, n_components=10) -> np.ndarray:
    """
    Per‑sample cosine similarity after projecting via CCA.
    - img_feats: (N, D_img) – e.g., (N, 512)
    - txt_feats: (N, D_txt) – e.g., (N, 312)
    - Returns: (N,) cosine similarities in the common CCA space.
    """
    N = img_feats.shape[0]
    if N < 2:
        return np.zeros(N)

    # Use at most min(D_img, D_txt, N-1) components
    n_comp = min(n_components, img_feats.shape[1], txt_feats.shape[1], N - 1)
    if n_comp < 1:
        return np.zeros(N)

    cca = CCA(n_components=n_comp)
    cca.fit(img_feats, txt_feats)
    img_trans, txt_trans = cca.transform(img_feats, txt_feats)

    # Cosine similarity in the projected space
    img_norm = img_trans / \
        (np.linalg.norm(img_trans, axis=1, keepdims=True) + 1e-8)
    txt_norm = txt_trans / \
        (np.linalg.norm(txt_trans, axis=1, keepdims=True) + 1e-8)
    return np.sum(img_norm * txt_norm, axis=1)


def get_group_id(text_label: np.ndarray, image_label: np.ndarray) -> np.ndarray:
    """
    Multi-label to group ID mapping:
        0: Real-Real   [0, 0]
        1: Real-Fake   [0, 1]
        2: Fake-Real   [1, 0]
        3: Fake-Fake   [1, 1]
    """
    return (text_label * 2 + image_label).astype(np.int32)


def get_group_name(gid: int) -> str:
    mapping = {
        0: "Real-Real [0,0]",
        1: "Real-Fake [0,1]",
        2: "Fake-Real [1,0]",
        3: "Fake-Fake [1,1]"
    }
    return mapping.get(gid, f"Unknown Group {gid}")


def analyze_accumulated_results(
    accumulator: CoherenceAccumulator,
    output_dir: str = "./experiment_6_results"
):
    """Run statistical analysis and generate plots (aggregated across all samples)."""
    os.makedirs(output_dir, exist_ok=True)

    # === Convert accumulator to results dict ===
    results = accumulator.get_results()

    # === Compute similarities (using your chosen metric) ===
    # IMPORTANT: If you are still using separate CCA, this may be flawed.
    # Consider using the fixed-CCA approach (fit on pre, transform both).
    # For now, we keep your existing compute_cca_similarity.
    pre_sim = compute_cca_similarity(results['x_image'], results['x_text'])
    post_sim = compute_cca_similarity(results['o_image'], results['o_text'])
    delta = post_sim - pre_sim

    # === Build DataFrame (no grouping) ===
    df = pd.DataFrame({
        'pre_sim': pre_sim,
        'post_sim': post_sim,
        'delta': delta
    })

    # Save full results
    csv_path = os.path.join(output_dir, 'coherence_results_aggregated.csv')
    df.to_csv(csv_path, index=False)
    print(f"✅ Saved per-sample results to {csv_path}")

    # === Overall Statistics ===
    n = len(df)
    pre_mean = df['pre_sim'].mean()
    post_mean = df['post_sim'].mean()
    delta_mean = df['delta'].mean()
    delta_std = df['delta'].std()
    t_stat, p_val = ttest_rel(df['pre_sim'], df['post_sim'])

    print("\n" + "=" * 70)
    print("📊 AGGREGATED STATISTICS (All Samples)")
    print("=" * 70)
    print(f"N = {n}")
    print(
        f"Pre-Memory Similarity (mean ± std) = {pre_mean:.4f} ± {df['pre_sim'].std():.4f}")
    print(
        f"Post-Memory Similarity (mean ± std) = {post_mean:.4f} ± {df['post_sim'].std():.4f}")
    print(f"Delta (Post - Pre) = {delta_mean:.4f} ± {delta_std:.4f}")
    print(
        f"Paired t-test: t = {t_stat:.4f}, p = {p_val:.2e} (significant: {p_val < 0.001})")

    # Save statistics
    stats_df = pd.DataFrame([{
        'N': n,
        'Pre-Mean': pre_mean,
        'Pre-Std': df['pre_sim'].std(),
        'Post-Mean': post_mean,
        'Post-Std': df['post_sim'].std(),
        'Delta-Mean': delta_mean,
        'Delta-Std': delta_std,
        't-statistic': t_stat,
        'p-value': p_val,
        'Significant (p<0.001)': p_val < 0.001
    }])
    stats_csv = os.path.join(output_dir, 'coherence_statistics_aggregated.csv')
    stats_df.to_csv(stats_csv, index=False)

    # === Plot 1: Boxplot (Pre vs Post) - TWO BARS ===
    plt.figure(figsize=(6, 8))
    # Prepare data for boxplot
    data_to_plot = [df['pre_sim'].values, df['post_sim'].values]
    labels = ['Pre-Memory', 'Post-Memory']

    bp = plt.boxplot(data_to_plot, labels=labels, patch_artist=True,
                     medianprops=dict(linewidth=2, color='black'),
                     whiskerprops=dict(linewidth=1.5),
                     capprops=dict(linewidth=1.5),
                     boxprops=dict(linewidth=1.5))

    # Color the boxes
    colors = ['#4C72B0', '#DD8452']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    # Add delta annotation
    plt.text(0.5, 0.95, f'Δ = {delta_mean:.4f} (p={p_val:.2e})',
             ha='center', va='center', fontsize=12, fontweight='bold',
             transform=plt.gca().transAxes,
             bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray'))

    plt.title(
        'Cross-Modal Similarity: Pre-Memory vs Post-Memory', fontsize=14)
    plt.ylabel('Similarity', fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.3)
    plt.tight_layout()
    boxplot_path = os.path.join(output_dir, 'coherence_boxplot_aggregated.png')
    plt.savefig(boxplot_path, dpi=300)
    print(f"✅ Saved aggregated boxplot to {boxplot_path}")

    # === Plot 2: Single Delta Bar with Error ===
    plt.figure(figsize=(6, 6))
    plt.bar(['Δ (Post - Pre)'], [delta_mean], yerr=delta_std,
            capsize=10, color='#2E86AB', alpha=0.8,
            edgecolor='black', linewidth=1.5)
    plt.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.5)
    # Annotate value
    plt.text(0, delta_mean + 0.01 if delta_mean >= 0 else delta_mean - 0.01,
             f'{delta_mean:.4f}', ha='center', va='bottom' if delta_mean >= 0 else 'top',
             fontsize=12, fontweight='bold')
    plt.title('Mean Delta (Post - Pre) Across All Samples', fontsize=14)
    plt.ylabel('Δ Similarity', fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.3)
    plt.tight_layout()
    barplot_path = os.path.join(output_dir, 'coherence_delta_aggregated.png')
    plt.savefig(barplot_path, dpi=300)
    print(f"✅ Saved aggregated delta bar chart to {barplot_path}")

    # Optional: Scatter plot of Pre vs Post (to see individual shifts)
    plt.figure(figsize=(6, 6))
    plt.scatter(df['pre_sim'], df['post_sim'], alpha=0.3, s=5, color='gray')
    # Diagonal line
    min_val = min(df['pre_sim'].min(), df['post_sim'].min())
    max_val = max(df['pre_sim'].max(), df['post_sim'].max())
    plt.plot([min_val, max_val], [min_val, max_val],
             'r--', linewidth=2, label='y=x')
    plt.xlabel('Pre-Memory Similarity')
    plt.ylabel('Post-Memory Similarity')
    plt.title('Per-Sample Shift: Pre vs Post Memory')
    plt.legend()
    plt.grid(alpha=0.3)
    scatter_path = os.path.join(output_dir, 'coherence_scatter_aggregated.png')
    plt.savefig(scatter_path, dpi=300)
    print(f"✅ Saved scatter plot to {scatter_path}")

    return stats_df, df


# =============================================================================
# PART 4: MAIN WRAPPER - CALL THIS FROM YOUR SCRIPT
# =============================================================================

def run_experiment_6_from_model(
    model,
    dataloader,
    device,
    image_dim: int = 512,
    text_dim: int = 312,
    output_dir: str = "./evons_data/experiment_6_results"
):
    """
    Full pipeline for Experiment 6.
    """
    print("=" * 70)
    print("🔬 EXPERIMENT 6: Cross-Modal Coherence Analysis (Multi-Label)")
    print("=" * 70)

    # Step 1: Run inference and collect features
    accumulator = run_inference_and_collect(
        model=model,
        dataloader=dataloader,
        device=device,
        image_dim=image_dim,
        text_dim=text_dim,
        max_samples=None   # set to e.g. 500 for quick test
    )

    print(f"✅ Collected {len(accumulator.x_image_list)} samples")

    # Step 2: Analyze
    stats_df, df = analyze_accumulated_results(
        accumulator, output_dir=output_dir)

    # Final summary
    print("\n" + "=" * 70)
    print("📌 KEY FINDINGS FOR YOUR PROFESSOR:")
    print("=" * 70)
    for _, row in stats_df.iterrows():
        sig = "✅ SIGNIFICANT" if row['Significant (p < 0.001)'] else "❌ NOT significant"
        print(f"{row['Group']}:")
        print(
            f"  Delta = {row['Delta (Post - Pre)']:.4f} (p={row['p-value']:.2e}) {sig}")

    print(f"\n📁 All results saved to: {output_dir}")
    return stats_df, df


# =============================================================================
# HOW TO CALL IT (at the bottom of your script)
# =============================================================================

if __name__ == "__main__":
    # ... (your existing model and dataloader loading code) ...

    # Run the experiment
    stats_df, df = run_experiment_6_from_model(
        model=model,
        dataloader=test,   # use your test loader
        device=device,
        image_dim=512,                # image feature dimension
        text_dim=312,                 # text feature dimension
        output_dir="./evons_data/experiment_6_results"
    )
