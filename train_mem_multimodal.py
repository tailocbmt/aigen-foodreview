import json
from transformers import CLIPProcessor, CLIPModel, FlavaProcessor, FlavaModel
import torch
import os
import logging
from torch.optim import AdamW
from torch.optim.lr_scheduler import StepLR
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from torch.utils.data import DataLoader
from modules.dataset import MultimodalDataset, HintsOfTruthMultimodalDataset
from larimar_base.base_models import CLIPDetectorWMemory, FLAVADetectorWMemory
from larimar_base.exp_models import CLIPDetectorSeparateMemory
import torch.nn as nn
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

# CONFIG
# Define the path to your config file
config_path = 'configs/multimodal_mem_config.json'

# Open and read the JSON file
with open(config_path, 'r') as file:
    config = json.load(file)

# --- Extract values into variables ---

# Options for model_name: 'clip', 'flava'
model_name = config.get('model_name', 'flava')
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

# LR decay config
LR_STEP_SIZE = config.get('LR_STEP_SIZE', 5)   # decay every 5 epochs
LR_GAMMA = config.get('LR_GAMMA', 0.5)         # multiply lr by 0.5

# wandb config
use_wandb = config.get('use_wandb', True)
wandb_project = config.get(
    'wandb_project', 'Multimodal synthesis data detection')
wandb_run_name = config.get('wandb_run_name', f'{model_name}-{dataset}-epo')
wandb_mode = config.get('wandb_mode', 'online')  # online, offline, disabled
if config.get("api_key"):
    os.environ["WANDB_API_KEY"] = config["api_key"]

device = 'cuda' if torch.cuda.is_available() else 'cpu'
available_models = ['clip', 'flava']
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

if model_name == 'clip':
    backbone = CLIPModel.from_pretrained("openai/clip-vit-base-patch16")
    processor = CLIPProcessor.from_pretrained('openai/clip-vit-base-patch16')
    model = CLIPDetectorWMemory(backbone, processor)
elif model_name == 'flava':
    backbone = FlavaModel.from_pretrained("facebook/flava-full")
    processor = FlavaProcessor.from_pretrained("facebook/flava-full")
    model = FLAVADetectorWMemory(backbone, processor)
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
else:
    train = MultimodalDataset(train_file, image_dir, processor, MAX_LENGTH)
    val = MultimodalDataset(val_file, image_dir, processor, MAX_LENGTH)

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
criterion = nn.BCEWithLogitsLoss()

# OPTIMIZATION
print('Training..')
count = 0
for epoch in range(1, EPOCHS):
    # reset before training
    if hasattr(model, "episodic_memory") and model.episodic_memory is not None:
        model.episodic_memory.reset_memory()

    model.train()
    model.memory_mode = "read_write"   # if you still use that design

    pred_val = []
    labels_val = []
    train_loss = 0.0

    print(f'Epoch: {epoch}')
    for i, batch in enumerate(train_dataloader):
        torch.cuda.empty_cache()
        optimiser.zero_grad()
        if i % 100 == 0:
            print(f'{i}th batch..')
        inputs, labels = batch['inputs'], batch['label']
        inputs = {key: tensor.squeeze(1).to(device)
                  for key, tensor in inputs.items()}
        labels = batch['label'].to(
            device=device, dtype=torch.float32).view(-1, 1)

        output = model(inputs).squeeze(1).to(torch.float64)

        loss = criterion(output, labels)
        loss.backward()
        optimiser.step()

        train_loss += loss.item()
        # break

    avg_train_loss = train_loss / len(train_dataloader)

    # reset again before validation
    if hasattr(model, "episodic_memory") and model.episodic_memory is not None:
        model.episodic_memory.reset_memory()

    val_loss = 0.0
    model.eval()
    model.memory_mode = "read"   # or "off"

    with torch.no_grad():
        print('Validating..')
        for j, batchv in enumerate(val_dataloader):
            inputs_val = batchv['inputs']
            inputs_val = {key: tensor.squeeze(1).to(
                device) for key, tensor in inputs_val.items()}
            label_val = batchv['label'].numpy().tolist()

            label_val_tensor = torch.tensor(
                batchv['label'], dtype=torch.float64).to(device)

            output_val = model(inputs_val).squeeze(1).to(torch.float64)

            loss_val = criterion(output_val, label_val_tensor)
            val_loss += loss_val.item()

            predictions = torch.sigmoid(output_val)
            predictions = torch.where(
                predictions > 0.5, 1, 0).detach().cpu().numpy().tolist()

            pred_val.extend(predictions)
            labels_val.extend(label_val)
            # break

        avg_val_loss = val_loss / len(val_dataloader)
        acc = accuracy_score(pred_val, labels_val)
        prec = precision_score(pred_val, labels_val)
        rec = recall_score(pred_val, labels_val)
        f1 = f1_score(pred_val, labels_val)

        logging.info(
            f'Epoch: {epoch}, Accuracy: {acc}, LR: {LR}, Batch Size: {BATCH_SIZE}.')
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
            save_path = os.path.join(output_dir, f'weight-{epoch}.pt')

            torch.save(model.state_dict(), save_path)
            print('Saved model.')

            if use_wandb and WANDB_AVAILABLE:
                wandb.log({
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
    # step lr decay after each epoch
    scheduler.step()

if use_wandb and WANDB_AVAILABLE:
    wandb.finish()
