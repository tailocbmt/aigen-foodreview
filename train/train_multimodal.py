import json
import numpy as np
from sklearn.model_selection import train_test_split
from transformers import AutoImageProcessor, AutoTokenizer, CLIPProcessor, CLIPModel, FlavaProcessor, FlavaModel
import torch
import os
import logging
from torch.optim import AdamW
from sklearn.metrics import accuracy_score, f1_score, hamming_loss, precision_score, recall_score
from torch.utils.data import DataLoader, Subset
from modules.dataset import EvonsMultimodalDataset, EvonsOfflineMultimodalDataset, MultimodalDataset, HintsOfTruthMultimodalDataset
from larimar_base.base_models import CLIPDetector, CLIPDetectorWMemory, FLAVADetector, FLAVADetectorWMemory
from larimar_base.multi_label_models import CLIPDetectorWMemoryCoAttention, FakeNewsMultimodal, FakeNewsMultimodalCoAttention, FakeNewsMultimodalWMemory, FakeNewsMultimodalWMemoryCoAttention, NetShareFusionCLIP
import torch.nn as nn
from modules.utils import multilabel_accuracy
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

# CONFIG
# Define the path to your config file
config_path = 'configs/multimodal_config.json'

# Open and read the JSON file
with open(config_path, 'r') as file:
    config = json.load(file)

# --- Extract values into variables ---

# Options for model_name: 'clip', 'flava'
model_name = config.get('model_name', 'flava')
use_memory = config.get('use_memory', 'linear')
MODEL_MODE = config.get('mode', "w/o memory")
# Note: for 'clip', max length should be 77
MAX_LENGTH = config.get('MAX_LENGTH', 512)
# File paths
dataset = config.get('dataset', 'food_review')
train_file = config.get('train_file', '')
val_file = config.get('val_file', '')
logging_file = config.get('logging_file', '')
output_dir = config.get('output_dir', '')
image_dir = config.get('image_dir', '')
# Training hyperparameters
EPOCHS = config.get('EPOCHS', 100)
BATCH_SIZE = config.get('BATCH_SIZE', 16)
LR = config.get('LR', 0.0001)
EARLY_STOP = config.get('EARLY_STOP', 10)

# wandb config
use_wandb = config.get('use_wandb', True)
wandb_project = config.get(
    'wandb_project', 'Multimodal synthesis data detection')
wandb_run_name = config.get('wandb_run_name', f'{model_name}-{dataset}')
wandb_mode = config.get('wandb_mode', 'online')  # online, offline, disabled
if config.get("api_key"):
    os.environ["WANDB_API_KEY"] = config["api_key"]

device = 'cuda' if torch.cuda.is_available() else 'cpu'
available_models = ['clip', 'flava', 'fakenews']
best_acc = 0

# Create output directory if needed
os.makedirs(output_dir, exist_ok=True)

# Initialize wandb
if use_wandb and WANDB_AVAILABLE:
    wandb.init(
        project=wandb_project,
        name=wandb_run_name,
        mode=wandb_mode,
        config=config
    )
    print('wandb initialized.')
elif use_wandb and not WANDB_AVAILABLE:
    print('wandb is not installed. Continuing without wandb.')

# MODEL SELECTION
if model_name not in available_models:
    raise ValueError(f'{model_name} not in {available_models}.')

output_dim = 1
if dataset == 'evons_multimodal':
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
    processor.append(AutoImageProcessor.from_pretrained("microsoft/resnet-50"))
    processor.append(AutoTokenizer.from_pretrained('bert-base-uncased'))

    if MODEL_MODE == "w/o mem":
        model = FakeNewsMultimodal(output_dim=output_dim)
    elif MODEL_MODE == "w/ mem":
        model = FakeNewsMultimodalWMemory(
            out_dim=output_dim, use_memory=use_memory)
    elif MODEL_MODE == "w/ coatt":
        model = FakeNewsMultimodalCoAttention(output_dim=output_dim)
    elif MODEL_MODE == "w/ coatt + mem":
        model = FakeNewsMultimodalWMemoryCoAttention(
            out_dim=output_dim, use_memory=use_memory)
elif model_name == 'netsharefusion':
    backbone = CLIPModel.from_pretrained("openai/clip-vit-base-patch16")
    processor = CLIPProcessor.from_pretrained('openai/clip-vit-base-patch16')

    model = NetShareFusionCLIP(
        backbone=backbone,
        num_labels=output_dim)
else:
    pass

model = model.to(device)
print(f'Model {model_name} loaded.')

if use_wandb and WANDB_AVAILABLE:
    wandb.watch(model, log="all", log_freq=100)

# DATA
if dataset == "hints_of_truth":
    train = HintsOfTruthMultimodalDataset(
        train_file, image_dir, "dev1", processor, MAX_LENGTH)
    val = HintsOfTruthMultimodalDataset(
        val_file, image_dir, "dev2", processor, MAX_LENGTH)
elif dataset == "evons":
    real_image_dir = config.get('real_image_dir')
    full_data = EvonsMultimodalDataset(
        train_file, real_image_dir, image_dir, processor, MAX_LENGTH)

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
    train = Subset(full_data, train_idx)
    val = Subset(full_data, val_idx)
else:
    train = EvonsOfflineMultimodalDataset(
        train_file, image_dir, processor, MAX_LENGTH)
    val = EvonsOfflineMultimodalDataset(
        val_file, image_dir, processor, MAX_LENGTH)

train_dataloader = DataLoader(train, BATCH_SIZE, shuffle=True)
print(f'Loaded Traininig File: {train_file}.')
val_dataloader = DataLoader(val, BATCH_SIZE, shuffle=False)
print(f'Loaded Validation File: {val_file}.')
print('Data loaded.')

# logging
logging.basicConfig(filename=logging_file, level=logging.INFO, filemode='a+')
print('Log file initialized.')

# OPTIMIZER
optimiser = AdamW(model.parameters(), lr=LR)
criterion = nn.BCEWithLogitsLoss()

# OPTIMIZATION
print('Training..')
count = 0
for epoch in range(1, EPOCHS):
    # reset before training
    model.reset_memory()
    model.train()
    model.set_memory_mode("read_write")

    pred_val = []
    labels_val = []
    train_loss = 0.0

    print(f'Epoch: {epoch}')
    for i, batch in enumerate(train_dataloader):
        torch.cuda.empty_cache()
        optimiser.zero_grad()

        if i % 1000 == 0:
            print(f'{i}th batch..')
        inputs, labels = batch['inputs'], batch['label']
        inputs = {key: tensor.squeeze(1).to(
            device) for key, tensor in inputs.items()}

        labels = torch.tensor(batch['label'], dtype=torch.float64)
        labels = labels.to(device)

        output = model(inputs).squeeze(1).to(torch.float64)

        loss = criterion(output, labels)
        loss.backward()
        optimiser.step()

        train_loss += loss.item()
    #     # break

    avg_train_loss = train_loss / len(train_dataloader)

    val_loss = 0.0

    model.reset_memory()
    model.eval()
    model.set_memory_mode("read")

    with torch.no_grad():
        print('Validating..')
        for j, batchv in enumerate(val_dataloader):
            inputs_val = batchv['inputs']
            inputs_val = {key: tensor.squeeze(1).to(
                device) for key, tensor in inputs_val.items()}

            labels = batchv['label'].to(device).float()

            outputs = model(inputs_val)

            loss = criterion(outputs, labels)
            val_loss += loss.item()

            probs = torch.sigmoid(outputs)
            preds = (probs > 0.5).int()

            pred_val.extend(preds.cpu().numpy())
            labels_val.extend(labels.cpu().numpy())
            # break

        avg_val_loss = val_loss / len(val_dataloader)
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

        logging.info(
            f'Epoch: {epoch}, Accuracy: {acc}, LR: {LR}, Batch Size: {BATCH_SIZE}.')
        print(f'# Train Loss: {avg_train_loss}')
        print(f'# Val Loss: {avg_val_loss}')
        print(f'# Accuracy: {acc}')
        print(f'# Multilabel Accuracy: {multi_acc}')
        print(f'# Precision: {prec}')
        print(f'# Recall: {rec}')
        print(f'# F1-score Macro: {macro_f1}')
        print(f'# F1-score Micro: {micro_f1}')
        print(f'# F1-score Sample: {samples_f1}')
        print(f'# Hamming loss: {hamming}')

        if use_wandb and WANDB_AVAILABLE:
            wandb.log({
                "epoch": epoch,
                "train/loss": avg_train_loss,
                "val/loss": avg_val_loss,
                "val/accuracy": acc,
                "val/multi_accuracy": multi_acc,
                "val/precision": prec,
                "val/recall": rec,
                "val/macro_f1_score": macro_f1,
                "val/micro_f1_score": micro_f1,
                "val/sample_f1_score": samples_f1,
                "val/hamming_loss": hamming,
                "lr": optimiser.param_groups[0]['lr']
            })

        if macro_f1 > best_acc:
            best_acc = macro_f1
            save_path = os.path.join(output_dir, f'weight-{epoch}.pt')

            torch.save(model.state_dict(), save_path)
            print('Saved model.')

            if use_wandb and WANDB_AVAILABLE:
                wandb.log({
                    "best_macro_f1": best_acc,
                    "best_val_accuracy": best_acc,
                    "best_epoch": epoch
                })
                wandb.save(save_path)

            count = 0
        else:
            count += 1

        if count == 10:
            print(f'Stopping at epoch: {epoch}')
            break
    # break

if use_wandb and WANDB_AVAILABLE:
    wandb.finish()
