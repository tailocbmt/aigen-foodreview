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
from modules.utils import multilabel_accuracy

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
    text_weights_dir = ""
    image_weights_dir = ""
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
    test = EvonsOfflineMultimodalWDctDataset(
        test_file, image_dir, processor, MAX_LENGTH)

test_dataloader = DataLoader(test, BATCH_SIZE)
print(f'Loaded Testing File: {test_file}.')

pred_val = []
labels_val = []

model.eval()
model.set_memory_mode("read")
with torch.no_grad():
    print('Validating..')
    for j, batchv in enumerate(test_dataloader):
        inputs_val = batchv['inputs']
        inputs_val = {key: tensor.squeeze(1).to(
            device) for key, tensor in inputs_val.items()}
        labels = batchv['label']

        outputs = model(inputs_val)

        probs = torch.sigmoid(outputs)
        preds = (probs > 0.5).int()

        pred_val.extend(preds.cpu().numpy())
        labels_val.extend(labels.cpu().numpy())

    # Convert to numpy
    y_true = np.array(labels_val)
    y_pred = np.array(pred_val)

    acc = accuracy_score(y_true, y_pred)
    multi_acc = multilabel_accuracy(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average='macro')
    rec = recall_score(y_true, y_pred, average='macro')
    macro_f1 = f1_score(y_true, y_pred, average="macro")
    micro_f1 = f1_score(y_true, y_pred, average="micro")
    samples_f1 = f1_score(y_true, y_pred, average="samples")
    hamming = hamming_loss(y_true, y_pred)

    print(f'Accuracy on the test set: {acc}')
    print(f'Multilabel Accuracy: {multi_acc}')
    print(f'Precision on the test set: {prec}')
    print(f'Recall on the test set: {rec}')
    print(f'Macro F1-score on the test set: {macro_f1}')
    print(f'Micro F1-score on the test set: {micro_f1}')
    print(f'Sample F1-score on the test set: {samples_f1}')
    print(f'Hamming on the test set: {hamming}')

    results = []
    for i in range(len(y_true)):
        results.append({
            "sample_id": i,
            "true_labels": y_true[i].tolist(),
            "pred_labels": y_pred[i].tolist(),
        })

    # Save predictions
    with open(f'evons_data/{model_name}.json', 'w') as f:
        json.dump(results, f, indent=2)
