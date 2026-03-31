import json

from transformers import AutoImageProcessor, ViTForImageClassification, ResNetForImageClassification
import torch
import os
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from torch.utils.data import DataLoader
from modules.dataset import HintsOfTruthVisionDataset, VisionDataset

# Define the path to your config file
config_path = 'configs/multimodal_mem_config.json'

# Open and read the JSON file
with open(config_path, 'r') as file:
    config = json.load(file)

# CONFIG
model_name = 'resnet'  # resnet, vit

dataset = config.get('dataset', 'food_review')
test_file = config.get('test_file', '')
image_dir = config.get('image_dir', '')
output_dir = config.get('output_dir', '')
BATCH_SIZE = config.get('BATCH_SIZE', 16)
MAX_LENGTH = config.get('MAX_LENGTH', 512)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
available_models = ['vit', 'resnet']

# WEIGHTS
weights = sorted(os.listdir(output_dir))[-1]
weights_dir = os.path.join(output_dir, weights)

if model_name == 'vit':
    model = ViTForImageClassification.from_pretrained(
        weights_dir, num_labels=2)
    tokenizer = AutoImageProcessor.from_pretrained(
        'google/vit-base-patch16-224')
elif model_name == 'resnet':
    model = ResNetForImageClassification.from_pretrained(
        weights_dir, num_labels=2)
    tokenizer = AutoImageProcessor.from_pretrained("microsoft/resnet-50")
else:
    pass
model = model.to(device)
print(f'Model {model_name} loaded.')

# DATA
if dataset == "hints_of_truth":
    test = HintsOfTruthVisionDataset(
        test_file, image_dir, "test", MAX_LENGTH, tokenizer)
else:
    test = VisionDataset(test_file, image_dir, tokenizer)

test_dataloader = DataLoader(test, BATCH_SIZE)
print(f'Loaded Testing File: {test_file}.')

pred_val = []
labels_val = []

model.eval()
with torch.no_grad():
    print('Validating..')
    for j, batchv in enumerate(test_dataloader):
        inputs_val = batchv['input'].to(device)
        inputs_val['pixel_values'] = inputs_val['pixel_values'].squeeze(1)
        label_val = batchv['label'].numpy().tolist()

        output_val = model(**inputs_val)
        output_val = torch.softmax(output_val.logits, dim=-1)
        predictions = torch.argmax(output_val, dim=-1).detach().cpu().numpy()
        predictions = np.minimum(predictions, 1).tolist()
        pred_val.extend(predictions)
        labels_val.extend(label_val)

    acc = accuracy_score(pred_val, labels_val)
    prec = precision_score(pred_val, labels_val)
    rec = recall_score(pred_val, labels_val)
    f1 = f1_score(pred_val, labels_val)
    print(f'Accuracy on the test set: {acc}')
    print(f'Precision on the test set: {prec}')
    print(f'Recall on the test set: {rec}')
    print(f'F1-score on the test set: {f1}')
