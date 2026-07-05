import string  # Useful for checking punctuation
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
import numpy as np
from sklearn.model_selection import train_test_split
import torch
import os
import json
from transformers import AutoImageProcessor, AutoTokenizer, CLIPProcessor, CLIPModel, FlavaProcessor, FlavaModel
from sklearn.metrics import accuracy_score, hamming_loss, precision_score, recall_score, f1_score
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


# Define the path to your config file
config_path = 'configs/multimodal_config.json'

# Open and read the JSON file
with open(config_path, 'r') as file:
    config = json.load(file)

# Extract values into variables
# Options for model_name: 'clip', 'flava'
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
test_file = config.get('test_file', '')
output_dir = config.get('output_dir', '')
image_dir = config.get('image_dir', '')
BATCH_SIZE = 512

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

test_dataloader = DataLoader(test, BATCH_SIZE)
print(f'Loaded Testing File: {test_file}.')

pred_val = []
labels_val = []

# ==========================================
# VISUALIZATION & INTERPRETABILITY
# ==========================================

# Only run for our custom FakeNews models as target layers differ for CLIP/FLAVA
if 'fakenews' in model_name:
    print("\n--- Generating Interpretability Visualizations ---")

    # We must ensure we are OUTSIDE of torch.no_grad()
    # We set model back to train to allow gradient tracking, but disable dropout behavior manually if needed,
    # or rely on pytorch-grad-cam to handle it.
    model.train()
    model.set_memory_mode("read")

    # 1. Define the target layer for ResNet-50 Grad-CAM
    # In Hugging Face's ResNet50, the last conv block is usually stages[-1]
    if hasattr(model, 'image_encoder'):
        target_layers = [model.image_encoder.resnet.encoder.stages[-1]]
    else:
        print("Warning: Could not find image_encoder on this model architecture.")
        target_layers = None

    # 2. Select a few specific samples (e.g., the first 3 in the test set)
    num_samples_to_visualize = 350

    for i in range(num_samples_to_visualize):
        sample = test[i]  # Get raw sample from dataset

        # Format inputs for the model (batch size of 1)
        # Note: Your DataLoader squeezed dim 1, we do the same here
        inputs_vis = {key: tensor.unsqueeze(0).squeeze(1).to(
            device) for key, tensor in sample['inputs'].items()}

        # --- A. IMAGE GRAD-CAM ---
        if target_layers is not None:
            # Generate Grad-CAM map
            # Separate the image tensor from the text dictionary
            image_tensor = inputs_vis['pixel_values']
            text_dict = {k: v for k, v in inputs_vis.items() if k !=
                         'pixel_values'}

            # Wrap the model for this specific sample
            wrapper_model = MultimodalWrapper(model, text_dict)

            # Initialize Grad-CAM with the wrapper
            cam = GradCAM(model=wrapper_model, target_layers=target_layers)

            # Pass ONLY the image tensor to Grad-CAM
            grayscale_cam = cam(input_tensor=image_tensor, targets=None)[0, :]

            # Reconstruct the original image from the normalized tensor
            # Hugging Face usually normalizes with ImageNet mean/std. We must reverse this for display.
            img_tensor = inputs_vis['pixel_values'].squeeze(
            ).cpu().numpy().transpose(1, 2, 0)
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            rgb_img = std * img_tensor + mean
            # Ensure values are bounded [0, 1]
            rgb_img = np.clip(rgb_img, 0, 1)

            # Overlay heatmap
            visualization = show_cam_on_image(
                rgb_img, grayscale_cam, use_rgb=True)

            plt.figure(figsize=(6, 6))
            plt.imshow(visualization)
            plt.title(
                f"Sample {i}: ResNet Grad-CAM Focus\nTrue Label: {sample['label']}")
            plt.axis('off')

            img_save_path = os.path.join(
                "evons_data", "visual", f"{model_name}_{dataset}_sample_{i}_gradcam.png")
            plt.savefig(img_save_path, bbox_inches='tight', dpi=300)
            plt.close()
            print(f"Saved Image Grad-CAM to: {img_save_path}")

        # --- B. TEXT ATTENTION ---
        # Run forward pass explicitly asking for interpretability
        # (Requires the modified feature_extractor we discussed before)
        try:
            _, text_attn = model.feature_extractor(
                inputs_vis, return_interpretability=True)

            if text_attn is not None:
                tokenizer = processor[1] if isinstance(
                    processor, list) else processor
                input_ids = inputs_vis['input_ids'][0].cpu().numpy()
                tokens = tokenizer.convert_ids_to_tokens(input_ids)

                weights = text_attn[0].detach().cpu().numpy()

                # Filter out [PAD] tokens for clean visualization
                # Get IDs for special tokens we want to hide
# Get valid length to drop all [PAD] tokens immediately
                pad_id = tokenizer.pad_token_id
                valid_len = (input_ids != pad_id).sum().item()

                valid_tokens = tokens[:valid_len]
                valid_weights = weights[:valid_len]

                filtered_tokens = []
                filtered_weights = []

                for tok, w in zip(valid_tokens, valid_weights):
                    # 1. Skip the [SEP] token
                    if tok == tokenizer.sep_token:
                        continue

                    # 2. Skip tokens that consist entirely of punctuation
                    # (This catches '.', ',', '!', '"', '-', '?', etc.)
                    if all(char in string.punctuation for char in tok):
                        continue

                    filtered_tokens.append(tok)
                    filtered_weights.append(w)

                plt.figure(figsize=(12, 2))
                sns.heatmap(
                    [filtered_weights], xticklabels=filtered_tokens, yticklabels=False, cmap="Reds")
                plt.title(f"Sample {i}: BERT [CLS] Token Attention")

                text_save_path = os.path.join(
                    "evons_data", "visual", f"{model_name}_{dataset}_sample_{i}_textattn.png")
                plt.savefig(text_save_path, bbox_inches='tight', dpi=300)
                plt.close()
                print(f"Saved Text Attention to: {text_save_path}")

        except TypeError:
            print("Note: feature_extractor does not accept 'return_interpretability'. Update the model class to view text attention.")
