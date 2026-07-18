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
    Replace your current results storage with this.
    """

    def __init__(self):
        self.x_image_list = []      # Pre-memory image features
        self.x_text_list = []       # Pre-memory text features
        # Post-memory image features (from x_output)
        self.o_image_list = []
        self.o_text_list = []       # Post-memory text features (from x_output)
        self.text_labels = []       # [0 or 1]
        self.image_labels = []      # [0 or 1]
        self.preds_text = []        # (Optional: for accuracy tracking)
        self.preds_image = []       # (Optional: for accuracy tracking)

    def add_sample(self, x_input: np.ndarray, x_output: np.ndarray,
                   label: np.ndarray, preds: np.ndarray = None):
        """
        Add a single sample to the accumulator.

        Args:
            x_input: Concatenated pre-memory features [2*D]
            x_output: Concatenated post-memory features [2*D]
            label: Multi-label array [text_label, image_label] (0=Real, 1=Fake)
            preds: (Optional) Predictions array [text_pred, image_pred]
        """
        D = 312  # Infer the dimension from the input shape

        # Slice pre-memory
        x_image = x_input[:D]
        x_text = x_input[D:]

        # Slice post-memory
        o_image = x_output[:D]
        o_text = x_output[D:]

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
        """Convert lists to numpy arrays for analysis."""
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
    text_index: int = 412  # Your D value (824/2). Set appropriately.
) -> CoherenceAccumulator:
    """
    Run inference on the dataloader and collect pre/post features.

    This replaces your current inference loop. It retains all your existing
    visualization logic but additionally accumulates features for Experiment 6.
    """
    model.eval()
    accumulator = CoherenceAccumulator()

    # Store memory indices for visualization (optional)

    curr = 0
    max_count = 100
    with torch.no_grad():
        for sample in tqdm(dataloader, desc="Collecting Coherence Features"):
            # === YOUR EXISTING INPUT PREPARATION ===
            inputs_vis = {key: tensor.unsqueeze(0).squeeze(1).to(device)
                          for key, tensor in sample['inputs'].items()}
            input_label = sample["label"].unsqueeze(0).squeeze(1).cpu().numpy()

            # === FORWARD PASS (with memory retrieval) ===
            output, retrieved = model(inputs_vis, return_memory=True)
            probs = torch.sigmoid(output)
            preds = (probs > 0.5).int().cpu().numpy().flatten()

            # === EXTRACT FEATURES FROM RETRIEVED ===
            x_input = retrieved["x_input"].squeeze().cpu().numpy()   # [2*D]
            x_output = retrieved["x_output"].squeeze().cpu().numpy()  # [2*D]

            # === STORE IN ACCUMULATOR ===
            accumulator.add_sample(
                x_input=x_input,
                x_output=x_output,
                label=input_label.flatten(),
                preds=preds
            )

            curr += 1
            if curr > max_count:
                break

    return accumulator


# =============================================================================
# PART 3: ANALYSIS AND VISUALIZATION (Same as before, with small updates)
# =============================================================================

def compute_cosine_similarity(img_feats: np.ndarray, txt_feats: np.ndarray) -> np.ndarray:
    """Compute pairwise cosine similarity."""
    img_norm = img_feats / \
        (np.linalg.norm(img_feats, axis=1, keepdims=True) + 1e-8)
    txt_norm = txt_feats / \
        (np.linalg.norm(txt_feats, axis=1, keepdims=True) + 1e-8)
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
        0: "Real-Real [0,0] (Coherent, Auth.)",
        1: "Real-Fake [0,1] (Mismatched)",
        2: "Fake-Real [1,0] (Mismatched)",
        3: "Fake-Fake [1,1] (Coherent, Synth.)"
    }
    return mapping.get(gid, f"Unknown Group {gid}")


def analyze_accumulated_results(
    accumulator: CoherenceAccumulator,
    output_dir: str = "./experiment_6_results"
):
    """Run statistical analysis and generate plots from accumulated features."""
    os.makedirs(output_dir, exist_ok=True)

    # === Convert accumulator to results dict ===
    results = accumulator.get_results()

    # === Compute similarities ===
    pre_sim = compute_cosine_similarity(results['x_image'], results['x_text'])
    post_sim = compute_cosine_similarity(results['o_image'], results['o_text'])
    delta = post_sim - pre_sim

    # === Build DataFrame ===
    group_id = get_group_id(results['text_label'], results['image_label'])

    df = pd.DataFrame({
        'group_id': group_id,
        'text_label': results['text_label'],
        'image_label': results['image_label'],
        'pre_sim': pre_sim,
        'post_sim': post_sim,
        'delta': delta
    })
    df['group_name'] = df['group_id'].apply(get_group_name)

    # Save full results
    csv_path = os.path.join(output_dir, 'coherence_results_full.csv')
    df.to_csv(csv_path, index=False)
    print(f"✅ Saved per-sample results to {csv_path}")

    # === Statistical Summary ===
    stats = []
    for gid in sorted(df['group_id'].unique()):
        subset = df[df['group_id'] == gid]
        n = len(subset)
        pre_mean = subset['pre_sim'].mean()
        post_mean = subset['post_sim'].mean()
        delta_mean = subset['delta'].mean()
        t_stat, p_val = ttest_rel(subset['pre_sim'], subset['post_sim'])
        stats.append({
            'Group': get_group_name(gid),
            'N': n,
            'Pre-Sim (Mean)': pre_mean,
            'Post-Sim (Mean)': post_mean,
            'Delta (Post - Pre)': delta_mean,
            't-statistic': t_stat,
            'p-value': p_val,
            'Significant (p < 0.001)': p_val < 0.001
        })

    stats_df = pd.DataFrame(stats)
    stats_csv = os.path.join(output_dir, 'coherence_statistics.csv')
    stats_df.to_csv(stats_csv, index=False)

    print("\n" + "=" * 70)
    print("📊 STATISTICAL SUMMARY")
    print("=" * 70)
    print(stats_df.to_string(index=False))

    # === Plot 1: Boxplot (Pre vs Post) ===
    plt.figure(figsize=(12, 6))
    df_melt = df.melt(
        id_vars=['group_id', 'group_name'],
        value_vars=['pre_sim', 'post_sim'],
        var_name='stage',
        value_name='cosine_similarity'
    )
    df_melt['stage'] = df_melt['stage'].map(
        {'pre_sim': 'Pre-Memory', 'post_sim': 'Post-Memory'})

    ax = sns.boxplot(
        data=df_melt,
        x='group_name',
        y='cosine_similarity',
        hue='stage',
        palette={'Pre-Memory': '#4C72B0', 'Post-Memory': '#DD8452'},
        linewidth=1.2
    )

    # Add delta annotations
    for i, gid in enumerate(sorted(df['group_id'].unique())):
        delta_mean = df[df['group_id'] == gid]['delta'].mean()
        ax.text(i, 0.92, f'Δ = {delta_mean:.3f}',
                ha='center', va='center', fontsize=10, fontweight='bold',
                bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray'))

    plt.title('Cross-Modal Cosine Similarity: Pre-Memory vs Post-Memory\n(Multi-Label Stratification)', fontsize=14)
    plt.xlabel('Composition Type', fontsize=12)
    plt.ylabel('Cosine Similarity', fontsize=12)
    plt.xticks(rotation=15, ha='right')
    plt.legend(title='Feature Stage', loc='upper left')
    plt.grid(axis='y', linestyle='--', alpha=0.3)
    plt.tight_layout()
    boxplot_path = os.path.join(output_dir, 'coherence_boxplot.png')
    plt.savefig(boxplot_path, dpi=300)
    plt.show()
    print(f"✅ Saved boxplot to {boxplot_path}")

    # === Plot 2: Delta Bar Chart ===
    plt.figure(figsize=(10, 6))
    group_names = [get_group_name(gid)
                   for gid in sorted(df['group_id'].unique())]
    means = df.groupby('group_id')['delta'].mean()
    stds = df.groupby('group_id')['delta'].std()
    colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D']

    bars = plt.bar(group_names, means, yerr=stds, capsize=8,
                   color=colors, alpha=0.8, edgecolor='black', linewidth=1.2)

    plt.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.5)

    for bar, mean_val in zip(bars, means):
        plt.text(bar.get_x() + bar.get_width()/2,
                 bar.get_height() + 0.01 if bar.get_height() >= 0 else bar.get_height() - 0.03,
                 f'{mean_val:.3f}',
                 ha='center', va='bottom' if mean_val >= 0 else 'top',
                 fontsize=11, fontweight='bold')

    plt.title('Δ Cross-Modal Similarity: Post-Memory - Pre-Memory\n(Positive = Enhanced Coherence, Negative = Exposed Mismatch)', fontsize=14)
    plt.ylabel('Δ Cosine Similarity (Post - Pre)', fontsize=12)
    plt.xticks(rotation=15, ha='right')
    plt.grid(axis='y', linestyle='--', alpha=0.3)
    plt.tight_layout()
    barplot_path = os.path.join(output_dir, 'coherence_delta_barchart.png')
    plt.savefig(barplot_path, dpi=300)
    plt.show()
    print(f"✅ Saved delta bar chart to {barplot_path}")

    return stats_df, df


# =============================================================================
# PART 4: MAIN WRAPPER - CALL THIS FROM YOUR SCRIPT
# =============================================================================

def run_experiment_6_from_model(
    model,
    dataloader,
    device,
    text_index: int = 312,  # Your D value (half of x_input dimension)
    output_dir: str = "./experiment_6_results"
):
    """
    Full pipeline for Experiment 6.

    Call this after training, using your test/validation dataloader.

    Args:
        model: Your trained model
        dataloader: Test/Val dataloader (must return dict with 'inputs' and 'label')
        device: torch.device
        text_index: The dimension D of each modality (x_input shape is 2*D)
        output_dir: Where to save results

    Example:
        >>> model = MyModel()
        >>> model.load_state_dict(torch.load('best.pth'))
        >>> test_loader = get_test_loader()
        >>> stats, df = run_experiment_6_from_model(model, test_loader, device)
    """
    print("=" * 70)
    print("🔬 EXPERIMENT 6: Cross-Modal Coherence Analysis (Multi-Label)")
    print("=" * 70)

    # Step 1: Run inference and collect features
    accumulator = run_inference_and_collect(
        model=model,
        dataloader=dataloader,
        device=device,
        text_index=text_index
    )

    print(f"✅ Collected {len(accumulator.x_image_list)} samples")

    # Step 2: Analyze
    stats_df, df = analyze_accumulated_results(
        accumulator, output_dir=output_dir)

    # Step 3: Final summary
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
# PART 5: HOW TO USE IN YOUR EXISTING SCRIPT
# =============================================================================

"""
HOW TO INTEGRATE INTO YOUR CURRENT SCRIPT:

1. Copy this entire file into your project as `experiment_6.py`

2. In your main script, import and call:

   from experiment_6 import run_experiment_6_from_model

   # After training or loading your model:
   model.eval()
   stats_df, df = run_experiment_6_from_model(
       model=model,
       dataloader=test_loader,  # your test dataloader
       device=device,
       text_index=412,          # YOUR D value. If x_input is 824, D=412
       output_dir="./experiment_6_results"
   )

3. That's it! The code will:
   - Reuse your existing inference loop
   - Extract pre-memory (x_input) and post-memory (x_output) features
   - Compute cosine similarity before and after memory
   - Stratify by your 4 multi-label groups
   - Generate publication-ready figures

ASSUMPTIONS (Matches your code):
- sample is a dict with keys: 'inputs' and 'label'
- inputs is a dict of tensors (image, text, etc.)
- label is a tensor of shape [2] -> [text_label, image_label]
- model(inputs, return_memory=True) returns (output, retrieved)
- retrieved contains 'x_input', 'x_output', and optionally 'top10_labels'
"""


# =============================================================================
# QUICK SANITY CHECK
# =============================================================================

if __name__ == "__main__":
    # # Dummy test with your exact data shapes
    # print("Running integration test with dummy data...")

    # # Simulate your retrieved outputs
    # D = 412
    # N_samples = 100

    # class DummyRetrieved:
    #     def __init__(self):
    #         self.x_input = torch.randn(1, 2*D)
    #         self.x_output = torch.randn(1, 2*D)
    #         self.top10_labels = torch.randint(0, 512, (1, 10))

    # # Simulate accumulator
    # acc = CoherenceAccumulator()
    # for i in range(N_samples):
    #     label = np.random.randint(0, 2, 2)
    #     preds = np.random.randint(0, 2, 2)
    #     x_input = np.random.randn(2*D)
    #     x_output = np.random.randn(2*D)
    #     acc.add_sample(x_input, x_output, label, preds)

    # # Analyze
    # stats, df = analyze_accumulated_results(
    #     acc, output_dir="./test_experiment_6")
    # print("✅ Integration test passed!")
    run_experiment_6_from_model(
        model=model,
        dataloader=test_dataloader,
        device=device
    )
