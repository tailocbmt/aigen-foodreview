import json

from transformers import AutoImageProcessor, ViTForImageClassification, ResNetForImageClassification
import torch
import os
import logging
from torch.optim import AdamW
from torch.optim.lr_scheduler import StepLR
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from torch.utils.data import DataLoader
from modules.dataset import HintsOfTruthVisionDataset, VisionDataset

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

# CONFIG
config_path = 'configs/multimodal_mem_config.json'

with open(config_path, 'r') as file:
    config = json.load(file)

model_name = 'resnet'  # resnet, vit
dataset = config.get('dataset', 'food_review')   # food_review, hints_of_truth
train_file = config.get('train_file', '')
val_file = config.get('val_file', '')
logging_file = config.get('logging_file', '')
output_dir = config.get('output_dir', '')
image_dir = config.get('image_dir', '')

EPOCHS = config.get('EPOCHS', 100)
BATCH_SIZE = config.get('BATCH_SIZE', 16)
LR = config.get('LR', 0.0001)
EARLY_STOP = config.get('EARLY_STOP', 5)

LR_STEP_SIZE = config.get('LR_STEP_SIZE', 5)
LR_GAMMA = config.get('LR_GAMMA', 0.5)

use_wandb = config.get('use_wandb', True)
wandb_project = config.get(
    'wandb_project', 'Multimodal synthesis data detection')
wandb_run_name = config.get('wandb_run_name', f'{model_name}-{dataset}-vision')
wandb_mode = config.get('wandb_mode', 'online')  # online, offline, disabled

if config.get("api_key"):
    os.environ["WANDB_API_KEY"] = config["api_key"]

device = 'cuda' if torch.cuda.is_available() else 'cpu'
available_models = ['vit', 'resnet']
best_acc = 0
if model_name == 'vit':
    model = ViTForImageClassification.from_pretrained(
        "google/vit-base-patch16-224")
    tokenizer = AutoImageProcessor.from_pretrained(
        'google/vit-base-patch16-224')
elif model_name == 'resnet':
    model = ResNetForImageClassification.from_pretrained("microsoft/resnet-50")
    tokenizer = AutoImageProcessor.from_pretrained("microsoft/resnet-50")
else:
    pass
model = model.to(device)
print(f'Model {model_name} loaded.')

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


# DATA
if dataset == "hints_of_truth":
    train = HintsOfTruthVisionDataset(train_file, image_dir, "dev1", tokenizer)
    val = HintsOfTruthVisionDataset(val_file, image_dir, "dev2", tokenizer)
else:
    train = VisionDataset(train_file, image_dir, tokenizer)
    val = VisionDataset(val_file, image_dir, tokenizer)

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
scheduler = StepLR(optimiser, step_size=LR_STEP_SIZE, gamma=LR_GAMMA)


# OPTIMIZATION
print('Training..')
count = 0
for epoch in range(1, EPOCHS + 1):
    model.train()
    pred_train = []
    labels_train = []
    train_loss = 0.0

    print(f'Epoch: {epoch}')
    for i, batch in enumerate(train_dataloader):
        torch.cuda.empty_cache()
        optimiser.zero_grad()
        if i % 100 == 0:
            print(f'{i}th batch..')
        inputs = batch['input'].to(device)
        inputs['pixel_values'] = inputs['pixel_values'].squeeze(1)

        labels = torch.tensor(batch['label'])
        labels = labels.to(device)

        output = model(**inputs, labels=labels)
        loss = output.loss
        loss.backward()
        optimiser.step()

        train_loss += loss

    avg_train_loss = train_loss / len(train_dataloader)

    pred_val = []
    labels_val = []
    val_loss = 0.0

    model.eval()
    with torch.no_grad():
        print('Validating..')
        for j, batchv in enumerate(val_dataloader):
            inputs_val = batchv['input'].to(device)
            inputs_val['pixel_values'] = inputs_val['pixel_values'].squeeze(1)
            label_val = batchv['label'].numpy().tolist()

            output_val = model(**inputs_val)
            output_val = torch.softmax(output_val.logits, dim=-1)
            predictions = torch.argmax(
                output_val, dim=-1).detach().cpu().numpy().tolist()
            pred_val.extend(predictions)
            labels_val.extend(label_val)

        avg_val_loss = val_loss / len(val_dataloader)
        acc = accuracy_score(labels_val, pred_val)
        prec = precision_score(labels_val, pred_val)
        rec = recall_score(labels_val, pred_val)
        f1 = f1_score(labels_val, pred_val)

        logging.info(
            f'Epoch: {epoch}, '
            f'Train Loss: {avg_train_loss}, Val Loss: {avg_val_loss}, '
            f'Val Acc: {acc}, '
            f'Precision: {prec}, Recall: {rec}, F1: {f1}, '
            f'LR: {optimiser.param_groups[0]["lr"]}, Batch Size: {BATCH_SIZE}.'
        )

        print(f'# Train Loss: {avg_train_loss}')

        print(f'# Val Loss: {avg_val_loss}')
        print(f'# Accuracy: {acc}')
        print(f'# Precision: {prec}')
        print(f'# Recall: {rec}')
        print(f'# F1-score: {f1}')

        if use_wandb and WANDB_AVAILABLE:
            wandb.log({
                "epoch": epoch,
                "train/loss": avg_train_loss,
                "val/loss": avg_val_loss,
                "val/accuracy": acc,
                "val/precision": prec,
                "val/recall": rec,
                "val/f1_score": f1,
                "lr": optimiser.param_groups[0]['lr']
            })

        if acc > best_acc:
            best_acc = acc
            save_path = os.path.join(output_dir, f'weight-{epoch}')
            model.save_pretrained(save_path)
            tokenizer.save_pretrained(save_path)
            print('Saved model.')

            if use_wandb and WANDB_AVAILABLE:
                wandb.log({
                    "best_val_accuracy": best_acc,
                    "best_epoch": epoch
                })
            count = 0
        else:
            count += 1

        if count == EARLY_STOP:
            print(f'Stopping at epoch: {epoch}')
            break

    scheduler.step()
    print()

if use_wandb and WANDB_AVAILABLE:
    wandb.finish()
