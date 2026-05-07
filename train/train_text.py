import json
import numpy as np
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, BertForSequenceClassification
from transformers import GPTNeoForSequenceClassification
import torch
import os
import logging
from torch.optim import AdamW
from modules.utils import multilabel_accuracy
from torch.optim.lr_scheduler import StepLR
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, hamming_loss
from torch.utils.data import DataLoader, Subset

# Note: Added EvonsTextDataset assumption based on your multimodal structure
from modules.dataset import EvonsOfflineTextDataset, HintsOfTruthTextDataset
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

model_name = 'bert'  # bert, gpt
MAX_LENGTH = 512

dataset = config.get('dataset', 'food_review')
train_file = config.get('train_file', '')
val_file = config.get('val_file', '')
logging_file = config.get('logging_file', '')
output_dir = config.get('output_dir', '')
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
available_models = ['bert', 'gpt']
best_acc = 0

# Create output directory if needed
os.makedirs(output_dir, exist_ok=True)

# MODEL SELECTION
if model_name == 'bert':
    model = BertForSequenceClassification.from_pretrained("bert-base-uncased")
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
elif model_name == 'gpt':
    model = GPTNeoForSequenceClassification.from_pretrained(
        "EleutherAI/gpt-neo-125M")
    model.config.pad_token_id = model.config.eos_token_id
    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-125M")
    tokenizer.pad_token = tokenizer.eos_token
else:
    pass
model = model.to(device)
print(f'Model {model_name} loaded.')

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
    train = HintsOfTruthTextDataset(
        train_file, "dev1", tokenizer, MAX_LENGTH)
    val = HintsOfTruthTextDataset(
        val_file, "dev2", tokenizer, MAX_LENGTH)
else:
    train = EvonsOfflineTextDataset(train_file, tokenizer, MAX_LENGTH)
    val = EvonsOfflineTextDataset(val_file, tokenizer, MAX_LENGTH)

train_dataloader = DataLoader(train, BATCH_SIZE, shuffle=True)
print(f'Loaded Training File: {train_file}.')
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
for epoch in range(1, EPOCHS):
    model.train()

    pred_val = []
    labels_val = []
    train_loss = 0.0

    print(f'Epoch: {epoch}')
    for i, batch in enumerate(train_dataloader):
        torch.cuda.empty_cache()
        optimiser.zero_grad()
        if i % 1000 == 0:
            print(f'{i}th batch..')

        inputs = batch['input'].to(device)
        input_ids = inputs['input_ids'].squeeze(1)
        attention_mask = inputs['attention_mask'].squeeze(1)
        labels = torch.tensor(batch['label'])
        labels = labels.to(device)

        # output = model(**inputs, labels=labels)
        output = model(input_ids=input_ids,
                       attention_mask=attention_mask, labels=labels)
        loss = output.loss

        loss.backward()
        optimiser.step()

        train_loss += loss.item()

    avg_train_loss = train_loss / len(train_dataloader)

    val_loss = 0.0
    model.eval()
    with torch.no_grad():
        print('Validating..')
        for j, batchv in enumerate(val_dataloader):
            inputs_val = batchv['input'].to(device)
            input_ids_val = inputs_val['input_ids'].squeeze(1)
            attention_mask_val = inputs_val['attention_mask'].squeeze(1)
            label_val = batchv['label'].numpy().tolist()

            # Pass labels for validation loss evaluation
            labels_val_tensor = torch.tensor(label_val).to(device)
            
            output_val = model(input_ids=input_ids_val,
                               attention_mask=attention_mask_val)

            loss_val = output.loss
            val_loss += loss_val.item()

            output_val = torch.softmax(output_val.logits, dim=-1)
            predictions = torch.argmax(
                output_val, dim=-1).detach().cpu().numpy().tolist()

            pred_val.extend(predictions)
            labels_val.extend(label_val)

        avg_val_loss = val_loss / len(val_dataloader)
        # Convert to numpy for comprehensive metrics calculation
        y_true = np.array(labels_val)
        y_pred = np.array(pred_val)

        acc = accuracy_score(y_true, y_pred)
        multi_acc = multilabel_accuracy(y_true, y_pred)
        prec = precision_score(y_true, y_pred, average='macro')
        rec = recall_score(y_true, y_pred, average='macro')
        macro_f1 = f1_score(y_true, y_pred, average="macro")
        micro_f1 = f1_score(y_true, y_pred, average="micro")
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

if use_wandb and WANDB_AVAILABLE:
    wandb.finish()
